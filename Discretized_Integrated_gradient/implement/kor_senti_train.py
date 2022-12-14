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

### Korean Dataset test.

class CustomDataset(Dataset): 
        def __init__(self):
            with open("./assets/word2idx.json", "r") as f:
                self.word2idx = json.load(f)
            with open("./assets/word2freq.json", "r") as f:
                self.word2freq = json.load(f)
            with open("./assets/word2prob.json", "r") as f:
                self.word2prob = json.load(f)
            with open("./assets/idx2word.json", "r") as f:
                self.idx2word = json.load(f)
            self.vocab_size = len(self.word2idx)
            
            idx2prob = {}
            for key, value in self.word2prob.items():
                idx2prob[self.word2idx.get(key)] = value
            self.idx2prob = idx2prob
            
            self.contexts = np.load('./.data/sentence.npy')
            self.targets = np.load('./.data/labels.npy')
            
            
        def __len__(self): 
            return len(self.targets)

        def __getitem__(self, idx): 
            x = torch.tensor(self.contexts[idx], dtype=torch.long)
            y = torch.tensor(self.targets[idx], dtype=torch.long)
            return x, y

dataset = CustomDataset()

datasetsize= len(dataset)
train_size = int(datasetsize * 0.8)
valid_size = int(datasetsize * 0.1)
test_size = datasetsize - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(valid_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
valid_iter = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=False)
test_iter = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

INPUT_DIM=dataset.vocab_size

HID_DIM = 150
ENC_LAYERS = 3
ENC_HEADS = 5
ENC_PF_DIM = 256
ENC_DROPOUT = 0.1
TEXT_PAD_IDX=0

device='cuda'

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT,
              device,
              max_length=50, 
            )

model=sentiment_classification(enc,HID_DIM,src_pad_idx=TEXT_PAD_IDX,device=device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

N_EPOCHS = 1000
CLIP = 1
patient=0
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train_kor(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate_kor(model, valid_iter, criterion)
    
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
        torch.save(model.state_dict(), './assets/transformer_kor_sentiment.pt')
    else:
        print('Best Epoch : ',best_epoch)
        patient+=1
        if patient>20:
            break

model.load_state_dict(torch.load('./assets/transformer_kor_sentiment.pt'))
test_loss = evaluate_kor(model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score_kor(model,train_iter,valid_iter,test_iter)

# Comparison - Simple GRU Model.

class simpleGRU(nn.Module):
    def __init__(self,input_dim,emb_hidden,hidden_size,num_layers,num_classes=1):
        super().__init__()
        self.num_layers=num_layers
        self.embedding=nn.Embedding(input_dim,emb_hidden)
        self.gru=nn.GRU(emb_hidden,hidden_size,num_layers,batch_first=True)
        #self.fc1=nn.Linear(hidden_size,hidden_size)
        self.fc1=nn.Linear(hidden_size,num_classes)
        #self.fc2=nn.Linear(hidden_size,num_classes)
        #self.activation1=nn.ReLU()
        self.activation2 = nn.Sigmoid()
        
    def forward(self, input):
        if input.dim()==2:
            out=self.embedding(input)
        else:
            out=input
        out,_=self.gru(out)
        out= out[:,-1,:]
        out=self.activation2(self.fc1(out))
        #out=self.activation2(self.fc2(out))
        
        return out

INPUT_DIM=dataset.vocab_size
HID_DIM = 128
NUM_LAYERS = 4

comp_model=simpleGRU(input_dim=INPUT_DIM,emb_hidden=HID_DIM,hidden_size=HID_DIM,num_layers=NUM_LAYERS).to('cuda')

optimizer = torch.optim.Adam(comp_model.parameters())
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1,last_epoch=-1,verbose=True)
criterion = nn.BCELoss()

N_EPOCHS = 100
CLIP = 1
patient=0
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    train_loss = train_kor(comp_model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate_kor(comp_model, valid_iter, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    if valid_loss < best_valid_loss:
        patient=0
        best_valid_loss = valid_loss
        torch.save(comp_model.state_dict(), './assets/grn_kor_sentiment.pt')
    else:
        patient+=1
        if patient>10:
            break

comp_model.load_state_dict(torch.load('./assets/grn_kor_sentiment.pt'))
test_loss = evaluate_kor(comp_model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

accuracy_score_kor(comp_model,train_iter,valid_iter,test_iter)

