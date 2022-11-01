import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from module.data import *
import tensorflow_datasets as tfds
import os
import datasets
import csv

def generate_examples(file_paths, data_filename):
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename == data_filename:
            with open(file_path, encoding="utf8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for idx, row in enumerate(reader):
                    yield idx, {
                        "idx": row["index"] if "index" in row else idx,
                        "sentence": row["sentence"],
                        "label": int(row["label"]) if "label" in row else -1,
                    }

os.getcwd()

os.listdir(r"C:\Users\UOS\Desktop\data")

for i in generate_examples(os.listdir(r"C:\Users\UOS\Desktop\data"),'train.tsv'):
    print(i)

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