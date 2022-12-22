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

train_iter,valid_iter,text_iter,INPUT_DIM,TEXT_PAD_IDX, TEXT=imdb_pytorch_load()

# 단어 사전
TEXT.vocab.stoi

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




# KNN w.r.t Words Embeddings

def k_nearest_neighbor(tok_embedding_layer,x,k=10,flag=0):
    
    # Generate 0 ~ Vocab_size integer.
    cand=torch.arange(0,10002,step=1).to(device)

    # Calculate Word Embedding.
    with torch.no_grad():
        emb_cand=tok_embedding_layer(cand)
    
    # Extract Sentence Word Embedding.
    # flag==1 : x is word token, o.w x is word embedding.
    if flag==1:
        emb_words=emb_cand[x,:]
    else:
        emb_words=x
    
    # K Neighbor words index of Sentence Word Embeddings.
    
    # This operation is very heavy. How can I ease this op? -> scikit learn - Use kneighbors_graph  

    # words_neighbor=torch.argsort(torch.sum((emb_words-emb_cand)**2,dim=2))[:,1:(k+1)]
    
    distance_mat=kneighbors_graph(emb_cand,10,mode='distance',p=2)
    distance_mat=torch.Tensor(distance_mat.toarray())
    words_neighbor=torch.argsort(distance_mat,dim=1,descending=True)[:,:k]

    # K Neighbor words embedding of Sentence Word Embeddings.
    
    sentence_index=[int(torch.where(i==tok_embedding_layer.weight)[0][0]) for i in emb_words.squeeze(0)]
    
    words_neighbor=words_neighbor[sentence_index]
    
    words_neighbor_embedding=emb_cand[words_neighbor]
    
    # Pad index : 1
    baseline_embed=emb_cand[[1]]

    baseline_embed=baseline_embed.repeat(1,500,1)
    
    return emb_words,baseline_embed,words_neighbor,words_neighbor_embedding


# monotonize algorithm

def monotonize_anchor(x,baseline,anchor,perturb_value):
    
    if anchor.dim()==3:
        anchor=torch.transpose(anchor,1,0)
        # K, Sentence length, Embedding dim
    else:
        anchor=anchor.unsqueeze(0)

    upper_state=torch.where((baseline<=anchor)&(x>=anchor),1,-1)
    lower_state=torch.where((baseline>=anchor)&(x<=anchor),1,-1)
    
    perturb_embed=torch.where((upper_state==1)|(lower_state==1),anchor,perturb_value)

    words=perturb_embed.to(device)

    
    return words



# anchor search algorithm

def anchor_greedy(x,baseline,neighbor,perturb_value):
    
    # First, Calculate monotonize_embed of KNN(w).
    monotone_cand=monotonize_anchor(x,baseline,neighbor,perturb_value).transpose(1,0)
    anchor_index=torch.argmin(torch.sum((neighbor-monotone_cand)**2,dim=2),dim=1)
    
    anchor_index=anchor_index.unsqueeze(1)
    nwords=torch.arange(0,500,step=1).unsqueeze(1)
    
    index_tensor=torch.cat((nwords,anchor_index),dim=1)
        
    # 500,10,256 Tensor -> 500,256 Tensor 
    
    anchor=monotone_cand[index_tensor[:,0],index_tensor[:,1],:]
    next_words_embed=neighbor[index_tensor[:,0],index_tensor[:,1],:].unsqueeze(0)
    
    return anchor,next_words_embed


def anchor_max_count(x,baseline,neighbor):
    
    neighbor=torch.transpose(neighbor,1,0)
    
    upper_state=torch.where((baseline<=neighbor)&(x>=neighbor),1,-1)
    lower_state=torch.where((baseline>=neighbor)&(x<=neighbor),1,-1)
    
    n_state=torch.where((upper_state==1)|(lower_state==1),1,0)

    highest_num_index=torch.argmax(torch.sum(n_state,dim=2),dim=0).unsqueeze(1)
    nwords=torch.arange(0,500,step=1).unsqueeze(1)
    index_tensor=torch.cat((nwords,highest_num_index),dim=1)
    
    neighbor=torch.transpose(neighbor,1,0)
        
    anchor=neighbor[index_tensor[:,0],index_tensor[:,1],:]

    return anchor


def anchor_search(tok_embedding_layer, x, m=30,method='greedy'):
    words=torch.zeros(0,500,256).to(device)
    
    # Calculate neighbor of x.
    input_emb, baseline_emb, _, neighbor=k_nearest_neighbor(tok_embedding_layer, x, k=10,flag=1)
    
    # Calculate Perturb Value
    perturb_value=input_emb-(input_emb-baseline_emb)/m
    
    # Greedy Heuristic Algorithm.
    if method=='greedy':
        for _ in tqdm(range(m)):
            anchor_temp,next_words_embed=anchor_greedy(input_emb, baseline_emb, neighbor, perturb_value)
            input_emb, _, _, neighbor=k_nearest_neighbor(tok_embedding_layer, next_words_embed, k=10)
            anchor_temp=anchor_temp.unsqueeze(0)
            words=torch.cat((words,anchor_temp),dim=0)

    
    # MaxCount Algorithm.
    else:
        for _ in tqdm(range(m)):
            anchor=anchor_max_count(input_emb, baseline_emb, neighbor)
            anchor_temp=monotonize_anchor(input_emb, baseline_emb, anchor, perturb_value)
            input_emb, _, _, neighbor=k_nearest_neighbor(tok_embedding_layer, anchor, k=10)
            
            words=torch.cat((words,anchor_temp),dim=0)

    return words



# Discretized Integrated Gradients

def discretized_integrated_gradients(x,baseline,model,method='greedy'):
    
    # 1. Anchor Search.
    anchor=anchor_search(x,baseline,method)
    
    # 2. Monotonize
    candidate=monotonize_anchor(x,baseline,anchor)
    
    # 3. Calculate input-embedding gradient w.r.t output
    # ...
    
    # 4. Riemann Approximation
    # 
    # each_score = grad*(x_{k+1}-x_{k}))
    # sum each_score  
    
    return None


# Metrics

def word_approximation_error(w,word_embedding):
    value=0
    return value

def log_odds(x,model,k):
    
    '''    
    LO Score is defined as the average difference of the negative logarithmic probabilities on the
    predicted class before and after masking the top k% features with zero padding.
    Lower scores are better.
    '''
    
    # First, calculate predicted class prob of original sentence.
    # And, calculate attribute power each words & each sentence.
    # Suppose, given N sentence, we obtain N X sentence_length tensor.
    # But, we need top k% words which attribute each sentence.
    # Finally, need : N tensor (predicted class prob of sentence), N X (sentence_length*K/100) Tensor (attribute words).   
    
    predict_prob=model(x)
    
    #implement plan
    # DIG=dig...
    
    # Input sentence perturb.
    # all x, if index=attribute_words_index then pad, o.w remain.
    # return perturb_sentence
    
    # perturb_prob=model(perturb_sentence)
    
    # log_odds=torch.mean(torch.log(perturb_prob)/torch.log(predict_prob))
    
    log_odds=0
    return log_odds

def comp_score(x,model,k):
    
    '''

    Comprehensiveness score is the average difference of the change in predictd class probability
    before and after removing the top k% features. Similar to Log-odds, this measures the influence of the top-attributed
    words on the model's prediction. 
    Higher scores are better.

    '''
    
    # First, calculate predicted class prob of original sentence.
    # And, calculate attribute power each words & each sentence.
    # Suppose, given N sentence, we obtain N X sentence_length tensor.
    # But, we need top k% words which attribute each sentence.
    # Finally, need : N tensor (predicted class prob of sentence), N X (sentence_length*K/100) Tensor (attribute words).   
    
    predict_prob=model(x)
    
    #implement plan
    # DIG=dig...
    
    # Input sentence perturb.
    # all x, if index=attribute_words_index then pad, o.w remain.
    # return perturb_sentence
    
    # perturb_prob=model(perturb_sentence)
    
    # comp=torch.mean(perturb_prob-predict_prob)
    
    comp=0
    return comp

def sufficiency(x,model):
    
    '''

    Sufficiency score is defined as the average difference of the change in predicted class
    probability before and after keeping only the top k% features.
    This measures the adequacy of the top k% attribute for model's prediction.
    Lower scores are better.
    
    '''
    
    # First, calculate predicted class prob of original sentence.
    # And, calculate attribute power each words & each sentence.
    # Suppose, given N sentence, we obtain N X sentence_length tensor.
    # But, we need top k% words which attribute each sentence.
    # Finally, need : N tensor (predicted class prob of sentence), N X (sentence_length*K/100) Tensor (attribute words).   
    
    predict_prob=model(x)
    
    #implement plan
    # DIG=dig...
    
    # Input sentence perturb.
    # all x, if index=attribute_words_index then remain, o.w pad.
    # return perturb_sentence
    
    # perturb_prob=model(perturb_sentence)
    
    # suff=torch.mean(perturb_prob-predict_prob)
    
    suff=0
    return suff



'''
# Test
for i in train_iter:
    i
    break

text_data=i.text[[0]]

text_data.shape

device='cpu'
model=model.to(device)

d=anchor_search(model.encoder.tok_embedding, text_data, m=30,method='greedy')

d2=anchor_search(model.encoder.tok_embedding, text_data, m=30,method='max_count')

words=torch.zeros(0,500,256).to(device)

tok_embedding_layer=model.encoder.tok_embedding
x=text_data
# Calculate neighbor of x.
input_emb, baseline_emb, _, neighbor=k_nearest_neighbor(tok_embedding_layer, x, k=10,flag=1)

input_emb.shape

baseline_emb.shape

t1,t2=anchor_greedy(input_emb,baseline_emb,neighbor,30)

t1.shape

t2.shape

x=input_emb
baseline=baseline_emb
neighbor=neighbor
iter=30


tdb=d[:,0,:]

tdb.shape

tdb[1]<=tdb[2]
tdb[2]<=tdb[3]

words=torch.zeros(0,500,256).to(device)

'''





