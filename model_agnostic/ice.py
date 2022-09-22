# Individual Conditional Expectation (ICE)

'''

ICE plots display one line per instance that shows how the instance's prediction changes when a feature changes.
The equivalent to a PDP for individual data instances is called individual conditional expectation (ICE) plot (Goldstein et al. 2017)

'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from data import data_preprocess
import matplotlib.pyplot as plt

data=data_preprocess(r"C:\Users\UOS\Desktop\data\Bike-Sharing-Dataset\day.csv")

feat_list=[i for n,i in enumerate(data.columns) if n!=10]
x=data.loc[:,feat_list]
y=data["cnt"]
x=pd.get_dummies(x)

reg=RandomForestRegressor(random_state=42)
reg.fit(x,y)

yhat=reg.predict(x)

plt.plot(y, color='green',linewidth=5, alpha=0.5, label='True')
plt.plot(yhat,color="yellow",alpha=0.5,label="Predict")
plt.xlabel('Instance')
plt.ylabel('Cnt')
plt.title("Model Result")
plt.legend()
plt.show()

data=x
feature="temp"
model=reg

def individual_conditional_expectation_plot(data,feature,model):
    interest_feature=np.sort(data[feature].unique())
    data2=data.copy()
    
    plt.figure(figsize=(8,15))
    for i in range(data2.shape[0]):
        temp=data2.iloc[[i],:]
        xprime=temp
        for _ in range(interest_feature.shape[0]-1):
            xprime=pd.concat([xprime,temp],axis=0)
        
        xprime[feature]=interest_feature
        subresult=model.predict(xprime)
       
        plt.plot(interest_feature,subresult,color="black",linewidth=0.1)
        
    plt.xlabel(feature)
    #plt.ylim(min_value-iqr,max_value+iqr)
    plt.ylabel("Model Predict")
    plt.title("Individual Conditional Expecataion about {}".format(feature))
    plt.show()

def centered_individual_conditional_expectation_plot(data,feature,model,anchor_point):
        
    interest_feature=np.sort(data[feature].unique())
    data2=data.copy()
    
    plt.figure(figsize=(6.4,9.6))
    for i in range(data2.shape[0]):
        temp=data2.iloc[[i],:]
        xprime=temp.copy()
        anchor=temp.copy()
        
        for _ in range(interest_feature.shape[0]-1):
            xprime=pd.concat([xprime,temp],axis=0)
        
        anchor[feature]=anchor_point
        
        xprime[feature]=interest_feature
        subresult=model.predict(xprime)
        anchorvalue=model.predict(anchor)
        subresult=subresult-anchorvalue
       
        plt.plot(interest_feature,subresult,color="black",linewidth=0.1)
        
    plt.xlabel(feature)
    #plt.ylim(min_value-iqr,max_value+iqr)
    plt.ylabel("Model Predict")
    plt.title("Centered Individual Conditional Expecataion about {}".format(feature))
    plt.show()


individual_conditional_expectation_plot(x,"temp",reg)
centered_individual_conditional_expectation_plot(x,"temp",reg,-5.220871)

