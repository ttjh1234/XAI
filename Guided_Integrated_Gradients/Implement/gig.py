import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict as ddict

import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import json
from tqdm import tqdm

# A very small number for comparing floating point values.
EPSILON = 1E-9

def l1_distance(x1,x2):
    return torch.abs(x1-x2).sum()

def translate_x_to_alpha(x,x_input,x_baseline):    
    return torch.where(x_input-x_baseline !=0, (x-x_baseline)/(x_input-x_baseline),torch.ones_like(x,dtype=torch.float32)*torch.tensor(float('nan')))

def translate_alpha_to_x(alpha,x_input,x_baseline):
    assert 0 <= alpha <= 1.0        
    return x_baseline + (x_input-x_baseline) * alpha


def compute_gradients(images,model,label):
    model.eval()
    model.zero_grad()
    output=model(images)[:,label].sum()
    
    input_grad=torch.autograd.grad(output,images)[0]
    
    return input_grad


def guided_ig(x_input, x_baseline, label ,model, steps=128, fraction=0.25, max_dist=0.02,device='cuda:1'):
    
    x = x_baseline.clone().detach()
    l1_total = l1_distance(x_input, x_baseline)
    attr = torch.zeros_like(x_input,dtype=torch.float32)
    
    total_diff = x_input - x_baseline
    
    if torch.abs(total_diff).sum() == 0:
        return attr
    
    for step in range(steps):
        
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=device)
        # Calculate gradients and make a copy.
        grad_actual = compute_gradients(x, model, label)
        grad = grad_actual.clone().detach()
        
        # Calculate current step alpha and the ranges of allowed values for this step.
        
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)

        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        
        # The goal of every step is to reduce L1 distance to the input.
        l1_target = l1_total * (1 - (step+1)/steps)     

        gamma = torch.tensor(float('inf'))
        
        while gamma > 1.0:
            x_old = x.clone().detach()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha = torch.where(torch.isnan(x_alpha),alpha_max,x_alpha)
            
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            
            x = torch.where(x_alpha<alpha_min, x_min, x)

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            
            if torch.isclose(l1_target, l1_current, rtol=EPSILON, atol=EPSILON):
                attr += (x - x_old) * grad_actual
                break
            
            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = torch.tensor(float('inf'))
            
            threshold = torch.quantile(torch.abs(grad), fraction, interpolation='lower')
            s = torch.logical_and(torch.abs(grad) <= threshold, ~torch.isinf(grad))
            
            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (torch.abs(x - x_max) * s).sum()
    
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = torch.tensor(float('inf'))

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual
    
    return attr
            




