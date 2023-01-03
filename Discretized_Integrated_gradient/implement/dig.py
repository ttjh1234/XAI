import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import kneighbors_graph
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import torchtext
from torchtext.datasets import IMDB
from tqdm import tqdm
from NLP.transformers_module import *
from copy import deepcopy
from collections import defaultdict as ddict
from module.knn_embedding import *
from module.path import *
import matplotlib as mpl
from konlpy.tag import Okt
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import json
from tqdm import tqdm
from XAI.Integrated_gradient.implement.ig_pytorch import *
from module.visualization import *

# 1. Data Fetch & Define Pretrained model 

train_iter,valid_iter,text_iter,INPUT_DIM,TEXT_PAD_IDX, TEXT=imdb_pytorch_load()

## Vocab dict

word2index=TEXT.vocab.stoi
index2word=TEXT.vocab.itos

## Model Hyperparameter

HID_DIM = 128
ENC_LAYERS = 3
ENC_HEADS = 8
ENC_PF_DIM = 512
ENC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

model=sentiment_classification(enc,HID_DIM,src_pad_idx=TEXT_PAD_IDX,device=device).to(device)    

## Model load
model.load_state_dict(torch.load('./assets/transformer_sentiment.pt'))

## Model Performance Check
criterion = nn.BCELoss()
test_loss = evaluate(model, text_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
accuracy_score(model,train_iter,valid_iter,text_iter)

# 2. Integrated Gradients 

## Get Test Sample
 
for i in train_iter:
    break

test_data=i.text[1]
test_label=i.label[1]

## Get token embedding layer of model
emb=model.encoder.tok_embedding

## Define baseline & set device
baseline=torch.ones(500).type(torch.long).to(device)
test_data=i.text[1].to(device)

## Calculate IG attribute 
ig=integrated_gradients(baseline=baseline,input=test_data,model=model,model_embed=emb)

## Sum over embedding dim
ig=torch.sum(ig,dim=1)

## Change Token to Words  
sample_text=''
for i in test_data.to('cpu').numpy():
    sample_text=sample_text+' '+index2word[i]

## Attribute Scaling (Bound value : Maximum Absolute Value)
ig2=ig/torch.max(torch.abs(ig.max()),torch.abs(ig.min()))

## Plot Results
print_html_language(test_data.to('cpu'),ig2,index2word)

# 3. Discretized Integrated Gradients




