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


train_iter,valid_iter,text_iter,INPUT_DIM,TEXT_PAD_IDX, TEXT=imdb_pytorch_load()

# 단어 사전
word2index=TEXT.vocab.stoi
index2word=TEXT.vocab.itos

HID_DIM = 256
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

modeldir=os.getcwd().split("XAI")[0]

model=sentiment_classification(enc,HID_DIM,src_pad_idx=TEXT_PAD_IDX,device=device).to(device)
model.load_state_dict(torch.load(modeldir+'NLP/model/transformer-sentiment.pt'))

# Plan to implement

# Discretized Integrated Gradients

def discretized_integrated_gradients(x,baseline,model,method='greedy'):
     
    return None

# Test Code

# test code
for i in train_iter:
    i
    break
i
text_data=i.text[3]
text_label=i.label[3]


sentence=[]
for w_index in text_data:
    if index2word[w_index]!='<pad>':
        sentence.append(index2word[w_index])
        
print(' '.join(sentence))

device='cpu'
model=model.to(device)

tok_embedding_layer=model.encoder.tok_embedding

x=text_data

adj=create_knn_matrix(tok_embedding_layer)
word_features=tok_embedding_layer.weight.cpu().detach().numpy()
word_idx_map=word2index



baseline=torch.ones(500).type(torch.int)

path=making_interpolation_path(x, baseline, 'cuda', (word_idx_map, word_features, adj), steps=30, strategy='greedy')

model=model.to('cuda')

model.eval()
#model.encoder.
model.zero_grad()

output=model(path).sum()

path=path.to('cuda')
path.requires_grad=True
output.shape

output.backward()

print(path.grad)

path_grad=path.grad
path=path.cpu().detach()
path_grad=path_grad.cpu().detach()





path_grad.shape

path.shape
tuple(path)[0][1:].shape



shifted_inputs_tpl	= tuple(torch.cat([scaled_features[1:], scaled_features[-1].unsqueeze(0)]) for scaled_features in tuple(path))
steps				= tuple(shifted_inputs_tpl[i] - tuple(path)[i] for i in range(len(shifted_inputs_tpl)))
scaled_grads		= tuple(path_grad[i] * steps[i] for i in range(path_grad.shape[0]))


for i in scaled_grads:
    print(i.shape)


att=[]
for i in scaled_grads:
    att.append(torch.sum(i,dim=1))

np.array(att).shape
    
np.array(att)



np.array(att).shape

scaled_grads[0].shape

path.shape

multiply=torch.zeros(0,500,256).to('cuda')

baseline_emb=tok_embedding_layer.weight[1]

baseline_emb=baseline_emb.to('cpu')

dig_vec=(path-baseline_emb/32)*path_grad

dig_vec=dig_vec.reshape(500,32,256)

dig=torch.sum(dig_vec,dim=1)

final_dig=torch.sum(dig,dim=1)

final_dig

att=final_dig.detach().numpy()

multiply.shape

path.shape[0]

def get_color(attr):
    if attr > 0:
        g = int(128*attr) + 127
        b = 128 - int(64*attr)
        r = 128 - int(64*attr)
    else:
        g = 128 + int(64*attr)
        b = 128 + int(64*attr)
        r = int(-128*attr) + 127
    return r,g,b

final_dig.detach().numpy()
np.min(final_dig.detach().numpy())

att=final_dig/torch.norm(final_dig)
att=att.to('cpu').detach().numpy()
scaled_att = (att - np.min(att)) / (np.max(att)- np.min(att)) * 2 - 1

cmap_bound=np.abs(att).max()

att2=att/cmap_bound

att2=(att.max()-att)/(att.max()-att.min())

att3=att2*2-1

rgblist=[]

for i in scaled_att:
    rgblist.append(get_color(i))

str=""
for i,c in zip(sentence,rgblist):
    if i!="<pad>":
        str=str+"\033[38;2;{};{};{}m {} \033[0m".format(c[0],c[1],c[2],i)
        
print(str)

def colorize(attrs,cmap='PiYG'):
    cmap_bound=np.abs(attrs).max()
    norm=mpl.colors.Normalize(vmin=-cmap_bound,vmax=cmap_bound)
    cmap=mpl.cm.get_cmap(cmap)
    
    colors=list(map(lambda x:mpl.colors.to_rgb(cmap(norm(x))),attrs))
    return colors

att=final_dig.detach().numpy()


colors=colorize(att)

list(map(lambda x : mpl.colors.hex2color(x),colors))
colors[0]*255

color_rgb = tuple([int(c*255) for c in colors])


### 한국어로 모델 재학습 및 test.

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

for i in train_iter:
    break

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

# 비교군 : simple GRU Model 

class simpleGRN(nn.Module):
    def __init__(self,input_dim,emb_hidden,hidden_size,num_layers,num_classes=1):
        super().__init__()
        self.num_layers=num_layers
        self.embedding=nn.Embedding(input_dim,emb_hidden)
        self.gru=nn.GRU(emb_hidden,hidden_size,num_layers,batch_first=True)
        self.fc1=nn.Linear(hidden_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,num_classes)
        self.activation1=nn.ReLU()
        self.activation2 = nn.Sigmoid()
        
    def forward(self, input):
        out=self.embedding(input)
        out,_=self.gru(out)
        out= out[:,-1,:]
        out=self.activation1(self.fc1(out))
        out=self.activation2(self.fc2(out))
        
        return out

INPUT_DIM=dataset.vocab_size
HID_DIM = 256
NUM_LAYERS = 2

comp_model=simpleGRN(input_dim=INPUT_DIM,emb_hidden=HID_DIM*4,hidden_size=HID_DIM,num_layers=NUM_LAYERS).to('cuda')

optimizer = torch.optim.Adam(comp_model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.BCELoss()

optimizer.state_dict().keys()
optimizer.state_dict()['param_groups']

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

