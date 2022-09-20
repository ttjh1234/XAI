# Partial Dependence Plot (PDP)

'''

PDP shows the marginal effect one or two features have on the predicted outcome of a ml model. (J.H. Friedman 2001)

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


def partial_dependence_plot(data,feature,model):
    interest_feature=np.sort(data[feature].unique())    
    fhat=[]
    data2=data.copy()
    for i in interest_feature:
        data2[feature]=i
        fhat.append(np.mean(model.predict(data2)))
    
    max_value=np.max(fhat)
    min_value=np.min(fhat)
    q3=np.quantile(fhat,q=0.75)
    q1=np.quantile(fhat,q=0.25)
    iqr=q3-q1
    
    plt.plot(list(interest_feature),fhat,color="black")
    plt.xlabel(feature)
    plt.ylim(min_value-iqr,max_value+iqr)
    plt.ylabel("Model Predict")
    plt.title("Partial Dependence Plot about {}".format(feature))
    plt.show() 
    

partial_dependence_plot(x,"temp",reg)
partial_dependence_plot(x,"hum",reg)
partial_dependence_plot(x,"windspeed",reg)

