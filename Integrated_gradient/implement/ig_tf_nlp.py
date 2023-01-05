'''
XAI Implement

Axiomatic Attribution for Deep Networks-IG

Choi Sung Su

Natural Language Processing.

Implement Dataset : IMDB Review Datasets.

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from webcolors import rgb_to_name
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)


try:
    from Discretized_Integrated_gradient.implement.module.model import *
except:
    os.chdir(r'c:\Users\UOS\Desktop\Sungsu\github\XAI')
    from Discretized_Integrated_gradient.implement.module.model import pretrained_model,imdb_data_fetch


(X_train,y_train), (X_test,y_test), word_index, id_to_word=imdb_data_fetch()

model=pretrained_model(r".\Discretized_Integrated_gradient\model\GRU_model.h5")

yhat=model.predict(X_train[0].reshape(1,-1))

sentence=[]
for i in X_train[0]:
    if i not in [0,1,2,3]:
        sentence.append(id_to_word[i])

' '.join(sentence)

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

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    _, index = kdt_db.query(rgb_tuple)
    return names[index]



def interpolate_language(input,baseline,alphas,embedding_layer):
    alphas_x = alphas[:,tf.newaxis,tf.newaxis]
    baseline=embedding_layer(baseline)
    baseline_x = tf.expand_dims(baseline, axis=0)
    
    input_x=embedding_layer(input)
    input_x=tf.expand_dims(input_x,axis=0)
    delta = input_x - baseline_x
    sentence = baseline_x +  alphas_x * delta
    return sentence


def compute_gradients(sentence,model):
    with tf.GradientTape() as tape:
        tape.watch(sentence)
        probs = model(sentence)
    return tape.gradient(probs, sentence)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def integrated_gradients(baseline,input,model,m_steps=300,batch_size=32):
    predict_model=tf.keras.models.Sequential()
    layer_names = [layer.name for layer in model.layers]
    for n,i in enumerate(layer_names):
        if n==0:
            embed=model.get_layer(i)
        else:
            predict_model.add(model.get_layer(i))
    
    # 1. Generate alphas
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

    # Accumulate gradients across batches
    integrated_gradients = 0.0

    # Batch alpha images
    ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

    for batch in ds:

    # 2. Generate interpolated sentence
        batch_interpolated_inputs = interpolate_language(input,baseline=baseline,alphas=batch,embedding_layer=embed)

    # 3. Compute gradients between model outputs and interpolated inputs
        batch_gradients = compute_gradients(sentence=batch_interpolated_inputs,model=predict_model)

    # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

    
    # 5. Scale integrated gradients with respect to input
    
    scale=embed(input)-embed(baseline) 
    scaled_integrated_gradients = scale * integrated_gradients
    return scaled_integrated_gradients


input=X_train[0]
baseline=np.zeros(500)

ig=integrated_gradients(baseline,input,model,m_steps=300,batch_size=32)
ig2=tf.reduce_sum(ig,axis=1)

d=np.array(ig2)

rgblist=[]

for i in d:
    rgblist.append(get_color(i))


sentence=[]
for i in X_train[0]:
    sentence.append(id_to_word[i])

' '.join(sentence)
sentence[0]


rgblist
str=""
for i,c in zip(sentence,rgblist):
    if i!="<pad>":
        str=str+"\033[38;2;{};{};{}m {} \033[0m".format(c[0],c[1],c[2],i)
        
print(str)

# zero imbedding vector ?

model.summary()

test_model=tf.keras.models.Sequential()

layer_names = [layer.name for layer in model.layers]

for n,i in enumerate(layer_names):
    if n==0:
        embed=model.get_layer(i)
    else:
        test_model.add(model.get_layer(i))



