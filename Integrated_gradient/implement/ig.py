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
from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import pydotplus as pydot

# Data import and preprocess

#dataset,info=tfds.load("cifar10",with_info=True)
#info.features['label'].names

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

bindex,_=np.where(y_train==2)
cindex,_=np.where(y_train==3)

bxtrain=x_train[bindex]
cxtrain=x_train[cindex]
bytrain=y_train[bindex]
cytrain=y_train[cindex]

xtrain=np.concatenate([bxtrain,cxtrain],axis=0)
ytrain=np.concatenate([bytrain,cytrain],axis=0)

btindex,_=np.where(y_test==2)
ctindex,_=np.where(y_test==3)

bxtest=x_test[btindex]
cxtest=x_test[ctindex]
bytest=y_test[btindex]
cytest=y_test[ctindex]

xtest=np.concatenate([bxtest,cxtest],axis=0)
ytest=np.concatenate([bytest,cytest],axis=0)

mean=[0,0,0]
std=[0,0,0]
newx_tr = np.ones(xtrain.shape)
newx_ts = np.ones(xtest.shape)

for i in range(3):
    mean[i] = np.mean(xtrain[:,:,:,i])
    std[i] = np.std(xtrain[:,:,:,i])

for i in range(3):
    newx_tr[:,:,:,i] = xtrain[:,:,:,i] - mean[i]
    newx_tr[:,:,:,i] = newx_tr[:,:,:,i] / std[i]
    newx_ts[:,:,:,i] = xtest[:,:,:,i] - mean[i]
    newx_ts[:,:,:,i] = newx_ts[:,:,:,i] / std[i]

xtrain=newx_tr
xtest=newx_ts
x_t,x_v,y_t,y_v=train_test_split(xtrain,ytrain,test_size=0.1,stratify=ytrain,random_state=42)

x_t.shape
x_v.shape
y_t2=np.int64(y_t)
y_v2=np.int64(y_v)


# Model Compile 

x_t=tf.convert_to_tensor(x_t)
x_v=tf.convert_to_tensor(x_v)
y_t=tf.convert_to_tensor(y_t2)
y_v=tf.convert_to_tensor(y_v2)
y_t=tf.where(y_t==2,0,1)
y_v=tf.where(y_v==2,0,1)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.0)


datagen.fit(x_t)

# model define

model = Sequential()
model.add(Conv2D(16, 3,use_bias=False,padding='same',activation='tanh',input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, 3,use_bias=False, activation='tanh',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, 3,use_bias=False, activation='tanh',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,use_bias=False,activation='relu'))
model.add(tf.keras.layers.Dense(1,use_bias=False,activation='sigmoid'))

model.summary()

tf.keras.utils.plot_model(model,to_file='c://Users/UOS/Desktop/model.png')

early_cb=tf.keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)
opt= tf.keras.optimizers.Adam()
myloss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
accuracy=tf.keras.metrics.Accuracy()
model.compile(loss=myloss,optimizer=opt)
history=model.fit(datagen.flow(x_t,y_t,batch_size=32),epochs=1000,validation_data=(x_v,y_v),callbacks=[early_cb])

pred_train=model.predict(x_t)
pt=tf.where(pred_train>=0.5,1,0)

pred_valid=model.predict(x_v)
pv=tf.where(pred_valid>=0.5,1,0)

print("Train accuracy : ",accuracy_score(pt,y_t)," Valid accuracy : ", accuracy_score(pv,y_v))

os.getcwd()
os.chdir("../")

# experiment1 16 64 64 32
model.save_weights("./model/experiment1.h5")
# experiment1 16 64 128 64
model.save_weights("./model/experiment2.h5")
# experiment1 16 64 128 64 no bias but strange Normarization
model.save_weights("./model/experiment3.h5")
# experiment1 16 64 128 64 no bias & Normarization
model.save_weights("./model/experiment4.h5")
# experiment1 16 64 128 64 no bias & Normarization No bias
model.save_weights("./model/experiment5.h5")

# experiment1 16 64 128 64 no bias & Normarization No bias all layers
model.save_weights("./model/experiment6.h5")

model.load_weights(r"C:\Users\UOS\Desktop\Sungsu\github\XAI\Integrated_gradient\model\experiment6.h5")
# Baseline Visualization
black=tf.zeros((32,32,3))
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

baseline=tf.zeros((32,32,3))
 
pred_train=model.predict(x_t)
pt=tf.where(pred_train>=0.5,1,0)
pt
accuracy_score(pt,y_t)

baseline11=baseline.numpy()
for i in range(3):
  baseline11[:,:,i]=baseline11[:,:,i]-mean[i]
  baseline11[:,:,i]=baseline11[:,:,i]/std[i]


model.predict(tf.expand_dims(baseline,axis=0))


y_t[0]
cat=x_t[1]
bird=x_t[2]
plt.imshow(tf.reshape(cat,(32,32,3)))
plt.imshow(tf.reshape(bird,(32,32,3)))


cat=tf.cast(cat,tf.float32)
bird=tf.cast(bird,tf.float32)

def interpolate_images(baseline,image,alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images

def compute_gradients(images):
  with tf.GradientTape() as tape:
    tape.watch(images)
    probs = model(images)
  return tape.gradient(probs, images)

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

def integrated_gradients(baseline,
                         image,
                         m_steps=300,
                         batch_size=32):
  # 1. Generate alphas
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

  # Accumulate gradients across batches
  integrated_gradients = 0.0

  # Batch alpha images
  ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

  for batch in ds:

    # 2. Generate interpolated images
    batch_interpolated_inputs = interpolate_images(baseline=baseline,
                                                   image=image,
                                                   alphas=batch)

    # 3. Compute gradients between model outputs and interpolated inputs
    batch_gradients = compute_gradients(images=batch_interpolated_inputs)

    # 4. Average integral approximation. Summing integrated gradients across batches.
    integrated_gradients += integral_approximation(gradients=batch_gradients)

  # 5. Scale integrated gradients with respect to input
  scaled_integrated_gradients = (image - baseline) * integrated_gradients
  return scaled_integrated_gradients

baseline=tf.zeros((32,32,3))

result1=integrated_gradients(baseline,cat,m_steps=300,batch_size=32)
result2=integrated_gradients(baseline,bird,m_steps=300,batch_size=32)


result11=result1.numpy()
for i in range(3):
  result11[:,:,i]=result11[:,:,i]*std[i]
  result11[:,:,i]=result11[:,:,i]+mean[i]

result22=result2.numpy()
for i in range(3):
  result22[:,:,i]=result22[:,:,i]*std[i]
  result22[:,:,i]=result22[:,:,i]+mean[i]

plt.imshow(result1)
plt.imshow(result2)

plt.imshow(result11/255)
plt.imshow(result22/255)


cat2=cat.numpy()
for i in range(3):
  cat2[:,:,i]=cat2[:,:,i]*std[i]
  cat2[:,:,i]=cat2[:,:,i]+mean[i]

plt.imshow(cat2/255)

bird2=bird.numpy()
for i in range(3):
  bird2[:,:,i]=bird2[:,:,i]*std[i]
  bird2[:,:,i]=bird2[:,:,i]+mean[i]

plt.imshow(bird2/255)

fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.imshow(cat2/255)
ax1.axis("off")
ax2.imshow(result11/255)
ax2.axis("off")
plt.show()


fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.imshow(bird2/255)
ax1.axis("off")
ax2.imshow(result22/255)
ax2.axis("off")
plt.show()



