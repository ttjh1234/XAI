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

def discretized_integrated_gradients(path,model,batch_size=32,device='cuda'):
        
    # Accumulate gradients across batches
    d_integrated_gradients = 0.0

    # 2. Batch embedding path 
    ds = DataLoader(path, batch_size=batch_size, shuffle=False, drop_last=False)
    
    for batch in ds:


    # 3. Compute gradients between model outputs and interpolated inputs
        batch_gradients = compute_gradients(sentence=batch,model=model)

    # 4. Average integral approximation. Summing integrated gradients across batches.
        d_integrated_gradients += integral_approximation(gradients=batch_gradients)

    
    # 5. Scale discretized integrated gradients with respect to input
    
    scale=path[-1]-path[0]
    scaled_integrated_gradients = scale * d_integrated_gradients
    return scaled_integrated_gradients
