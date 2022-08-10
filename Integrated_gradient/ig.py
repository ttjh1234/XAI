'''
XAI Implement

Axiomatic Attribution for Deep Networks

Choi Sung Su

First. Baseline Visualization

Second. Why prior Gradients Attribution Method doesn't Sensitivity

Thrid. Integrated Gradients Method Implement at Image Recognition Task  

Implement Dataset : Cifar10

'''

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Data import and preprocess

dataset,info=tfds.load("cifar10",with_info=True)
info.features['label'].names

cf_train,cf_test=dataset['train'],dataset['test']

def data_preprocess(feature):
    img=feature['image']
    label=feature['label']
    image=tf.image.resize(img,[224,224])
    label=tf.one_hot(label,depth=10)

    return image,label

cf_train=cf_train.map(data_preprocess)
cf_test=cf_test.map(data_preprocess)

cf_train2=cf_train.batch(32).prefetch(1)
cf_test2=cf_test.batch(32).prefetch(1)

# Model Define

feature_extract=VGG16(include_top=False, input_shape=(224, 224, 3))
flat=tf.keras.layers.Flatten()(feature_extract.output)
fc1=tf.keras.layers.Dense(4096,activation='relu')(flat)
fc2=tf.keras.layers.Dense(4096,activation='relu')(fc1)
cls_layer=tf.keras.layers.Dense(10,activation='softmax')(fc2)

model=tf.keras.Model(inputs=feature_extract.input,outputs=cls_layer)

# Model Compile 

early_cb=tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
opt= tf.keras.optimizers.Adam()
myloss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(loss=myloss,optimizer=opt)

history=model.fit(cf_train2,epochs=1000,validation_data=cf_test2,callbacks=[early_cb])


# Baseline Visualization

black=tf.zeros((32,32,3))

for n,i in enumerate(cf_train):
    if n==1:
        data=i
        break

plt.figure(figsize=(8,3))
plt.imshow(data['image'])
plt.imshow(black)


# Why Gradient Attribution Method doen't satisfy Sensitivity
x=np.arange(-1,10,step=0.01)
y=1-np.maximum(1-x,np.zeros(1100))

plt.plot(x,y,'black')
plt.title("F(x)=1-ReLU(1-x)")
plt.axhline(0,-1,10,color='black',linestyle='--')
plt.axvline(0,-1,4,color='black',linestyle='--')
plt.plot(0,0,'o',c='blue')
plt.plot(2,1,'o',c='red')
plt.annotate('',xy=(2.1,1.1),xytext=(2.5,1.5),xycoords='data',arrowprops=dict(arrowstyle='->',color='black',lw=2))
plt.annotate('Input x=2, but gradient=0',xy=(2.5,1.6))
plt.annotate('Gradients Attribution Method : xF\'(x)=0',xy=(2.5,2.2))
plt.annotate('F(x)-F(x\')=1',xy=(2.5,1.9))
plt.annotate('',xy=(0.1,-0.1),xytext=(0.5,-0.5),xycoords='data',arrowprops=dict(arrowstyle='->',color='black',lw=2))
plt.annotate('Baseline x\'=0',xy=(0.6,-0.6))
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((-1,10))
plt.ylim((-1,4))
plt.show()

# implement Integrated Gradients
# First Model Test
# load weights trained at Surver.

model.load_weights("C:/Users/UOS/Desktop/Sungsu/github/XAI/Integrated_gradient/model/0810.h5")

def data_preprocess_test(feature):
    img=feature['image']
    label=feature['label']
    image=tf.image.resize(img,[224,224])

    return image,label

tcf_train=cf_train.map(data_preprocess_test)
tcf_test=cf_test.map(data_preprocess_test)
tcf_train2=tcf_train.batch(1).prefetch(1)
tcf_test2=tcf_test.batch(1).prefetch(1)

train_label=[]
pred_train=[]

for i,j in tcf_train2:
    train_label.append(j)
    pred_train.append(np.argmax(model.predict(i)))

test_label=[]
pred_test=[]

for i,j in tcf_test:
    test_label.append(j)
    pred_test.append(np.argmax(model.predict(i)))

len(train_label)
train_label
len(test_label)


 
 