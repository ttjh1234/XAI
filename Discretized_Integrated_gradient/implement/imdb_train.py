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

train_iter,valid_iter,text_iter,INPUT_DIM,TEXT_PAD_IDX, TEXT=imdb_pytorch_load()

# 단어 사전
word2index=TEXT.vocab.stoi
index2word=TEXT.vocab.itos

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

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

N_EPOCHS = 100
CLIP = 1
patient=0

best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    if valid_loss < best_valid_loss:
        best_epoch=epoch
        print('Best Epoch : ',best_epoch)
        
        patient=0
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './assets/transformer_sentiment.pt')
    else:
        print('Best Epoch : ',best_epoch)
        patient+=1
        if patient>10:
            break
    
model.load_state_dict(torch.load('./assets/transformer_sentiment.pt'))
test_loss = evaluate(model, text_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
accuracy_score(model,train_iter,valid_iter,text_iter)