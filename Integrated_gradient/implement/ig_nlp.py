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

try:
    from Integrated_gradient.implement import *
except:
    os.chdir(r'c:\\Users\\UOS\\Desktop\\Sungsu\\github\\XAI\\')
    from Integrated_gradient.implement import pretrained_model,imdb_data_fetch

try:
    from NLP.utils import *
except:
    os.chdir(r'c:\\Users\\UOS\\Desktop\\Sungsu\\github\\XAI\\')
    from Integrated_gradient.implement import pretrained_model,imdb_data_fetch



(X_train,y_train), (X_test,y_test), word_index, id_to_word=imdb_data_fetch()

os.getcwd()

model=pretrained_model(r".\Discretized_Integrated_gradient\model\GRU_model.h5")

model.summary()

yhat=model.predict(X_train[0].reshape(1,-1))

sentence=[]
for i in X_train[0]:
    if i not in [0,1,2,3]:
        sentence.append(id_to_word[i])

' '.join(sentence)


model.get_layer('embedding')(X_train[0])




def interpolate_language(input,baseline,alphas,embedding_layer):
    alphas_x = alphas[:,tf.newaxis,tf.newaxis]
    baseline=embedding_layer(baseline)
    baseline_x = tf.expand_dims(baseline, axis=0)
    c=tf.constant([500,1], tf.int32)
    baseline_x =tf.tile(baseline_x, c)
    baseline_x=tf.expand_dims(baseline_x,axis=0)
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
    embed=model.get_layer('embedding')(X_train[0])
    
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
        batch_gradients = compute_gradients(sentence=batch_interpolated_inputs,model=model)

    # 4. Average integral approximation. Summing integrated gradients across batches.
        integrated_gradients += integral_approximation(gradients=batch_gradients)

  # 5. Scale integrated gradients with respect to input
    scaled_integrated_gradients = (input - baseline) * integrated_gradients
    return scaled_integrated_gradients



alphas = tf.linspace(start=0.0, stop=1.0, num=30)
input=X_train[0]
baseline=0
embedding_layer=model.get_layer('embedding')
sen=interpolate_language(X_train[0],0,alphas,model.get_layer('embedding'))