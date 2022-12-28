import torch
import numpy as np
import os
from sklearn.neighbors import kneighbors_graph

def create_knn_matrix(tok_embedding_layer,k=500):
    
    '''
    
    Make KNN Matrix w.r.t Embedding Vec.
    The matrix has K neighbor word index every word index.
    This matrix is well used, so after initial call, we reuse it.  
    
    '''
    
    # Get Model Word Embedding Vec.
    
    emb_mat=tok_embedding_layer.weight.cpu().detach().numpy()
       
    # Get K Neighbor matrix w.r.t each words using l2 distance between Word Embeddings.
    # scikit learn - Use kneighbors_graph  
    
    knn_mat=kneighbors_graph(emb_mat,k,mode='distance',p=2)
    
    return knn_mat