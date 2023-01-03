import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import torchtext
from torchtext.datasets import IMDB
from tqdm import tqdm
from NLP.transformers_module import *
from copy import deepcopy
from collections import defaultdict as ddict

import matplotlib as mpl
from konlpy.tag import Okt
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import json
from tqdm import tqdm

def interpolate_language(input,baseline,alphas,embedding_layer):
    alphas_x = alphas.unsqueeze(1).unsqueeze(2)
    baseline=embedding_layer(baseline)
    baseline_x = torch.unsqueeze(baseline, dim=0)
    
    input_x=embedding_layer(input)
    input_x=torch.unsqueeze(input_x,axis=0)
    delta = input_x - baseline_x
    sentence = baseline_x +  alphas_x * delta
    
    return sentence

def compute_gradients(sentence,model):
    model.eval()
    model.zero_grad()
    output=model(sentence).sum()
    input_grad=torch.autograd.grad(output,sentence)[0]

    return input_grad

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = torch.mean(grads, dim=0)
    return integrated_gradients

def integrated_gradients(baseline,input,model,model_embed,m_steps=300,batch_size=32,device='cuda'):
    # 0. Take Embedding Layer
    embedding=model_embed
    
    # 1. Generate alphas
    alphas = torch.linspace(start=0, end= 1,steps=m_steps).to(device)

    # Accumulate gradients across batches
    integrated_gradients = 0.0

    # Batch alpha images
    ds = DataLoader(alphas, batch_size=batch_size, shuffle=False, drop_last=False)
    
    for batch in ds:

    # 2. Generate interpolated sentence
        batch_interpolated_inputs = interpolate_language(input,baseline=baseline,alphas=batch,embedding_layer=embedding)

    # 3. Compute gradients between model outputs and interpolated inputs
        batch_gradients = compute_gradients(sentence=batch_interpolated_inputs,model=model)

    # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

    
    # 5. Scale integrated gradients with respect to input
    
    scale=embedding(input)-embedding(baseline) 
    scaled_integrated_gradients = scale * integrated_gradients
    return scaled_integrated_gradients
