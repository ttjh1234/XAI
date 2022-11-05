import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os


# Plan to Implement

# first anchor search algorithm

def word_approximation_error(w,word_embedding):
    value=0
    return value


def anchor_search(x,baseline,method='greedy'):
    anchor=[]
    if method=='greedy':
        a=0
    else:
        a=1
    
    return anchor

# second monotonize algorithm

def monotonize_anchor(x,baseline,anchor):
    anchor=[]
    return anchor

def log_odds(x,model):
    
    '''
    
    LO Score is defined as the average difference of the negative logarithmic probabilities on the
    predicted class before and after masking the top k% features with zero padding.
    Lower scores are better.
    
    '''
    value=0
    return value

def comp_score(x,model):
    
    '''

    Comprehensiveness score is the average difference of the change in predictd class probability
    before and after removing the top k% features. Similar to Log-odds, this measures the influence of the top-attributed
    words on the model's prediction. 
    Higher scores are better.

    '''
    
    value=0
    return value

def sufficiency(x,model):
    
    '''

    Sufficiency score is defined as the average difference of the change in predicted class
    probability before and after keeping only the top k% features.
    This measures the adequacy of the top k% attribute for model's prediction.
    Lower scores are better.
    
    '''

    value=0
    return value