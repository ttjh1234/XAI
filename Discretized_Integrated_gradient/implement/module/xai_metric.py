import torch
import pandas as pd
import numpy as np

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
