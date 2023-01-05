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

def interpolate_images(baseline,image,alphas,device):
    alphas_x = alphas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    baseline_x = torch.unsqueeze(baseline, dim=0)
    input_x = torch.unsqueeze(image, dim=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    images = torch.tensor(images, requires_grad=True, dtype=torch.float32, device=device)
    return images

def compute_gradients(images,model,label):
    model.eval()
    model.zero_grad()
    output=model(images)[:,label].sum()
    
    input_grad=torch.autograd.grad(output,images)[0]
    
    return input_grad

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / 2.0
  integrated_gradients = torch.mean(grads, dim=0)
  return integrated_gradients

def integrated_gradients(baseline,image,model,label,m_steps=300,batch_size=32,device='cuda'):
    # 1. Generate alphas
    m_steps=300
    device='cuda'
    batch_size=32

    alphas = torch.linspace(start=0, end= 1,steps=m_steps).to(device)

    # Accumulate gradients across batches
    integrated_gradients = 0.0
    
    # Batch alpha images
    ds = DataLoader(alphas, batch_size=batch_size, shuffle=False, drop_last=False)
    
    for batch in ds:

        # 2. Generate interpolated image
               
        batch_interpolated_inputs = interpolate_images(baseline=baseline,
                                                    image=image,
                                                    alphas=batch,
                                                    device=device)
        # 3. Compute gradients between model outputs and interpolated inputs
        
        batch_gradients = compute_gradients(images=batch_interpolated_inputs,model=model,label=label)

        # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

    # 5. Scale integrated gradients with respect to input
    scaled_integrated_gradients = (image - baseline) * integrated_gradients
    return scaled_integrated_gradients

def scaled_attribute(attribute):
    attribute=np.fabs(attribute)
    scaled_attribute = attribute / np.max(attribute)
    scaled_attribute=np.transpose(scaled_attribute,[1,2,0])
    return scaled_attribute