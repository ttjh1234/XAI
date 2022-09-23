# Accumulated Local Effects Plot (ALE)

'''

Accumulated local effects describe how features influence the prediction of a machine learning model on average.
ALE Plots are a faster and unbiased alternative to partial dependence plots.

'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
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

def bin_generate(x,n):
    bin_list=[]
    temp=np.min(x)
    inter=(np.max(x)-np.min(x))/(n-1)

    bin_list.append(np.min(x)-inter)

    for i in range(n-1):
        temp=temp+inter
        bin_list.append(temp)
    
    bin_list.append(np.max(x)+inter)
    
    return bin_list

def allocate_group(x,bin_list):
    for i in range(len(bin_list)):
        if bin_list[i]<=x<bin_list[i+1]:
            return i+1

def accumulated_local_effect(data,feature,n,model):
    interest=data[feature]
    bingroup=bin_generate(interest,n)
    data2=data.copy()
    data3=data2.sort_values(by=[feature]).reset_index().drop('index',axis=1)
    data3['group']=0
    data3['group']=data3[feature].map(lambda x : allocate_group(x,bingroup))
    
    local_effects=[]
    number_of_group_member=[]
    for i in range(1,(n+1)):
        temp1=data3.loc[data3['group']==i,data.columns]
        temp2=temp1.copy()
        temp1[feature]=bingroup[i-1]
        temp2[feature]=bingroup[i]
        number_of_group_member.append(temp1.shape[0])
        if temp1.shape[0]!=0:
            f1=model.predict(temp1)
            f2=model.predict(temp2)
            local_effects.append(np.mean(f2-f1))
        else:
            local_effects.append(np.mean(f2-f1))
    
    ale=np.cumsum(local_effects)
    average_effect=np.sum(ale*number_of_group_member)/np.sum(number_of_group_member)
    
    data4=data3[[feature,'group']]
    data4['ALE']=data4['group'].map(lambda x : ale[x-1])         
    data4['Centered_ALE']=data4['group'].map(lambda x : ale[x-1]-average_effect)
    
    return data4

def ale_plot(data,feature,n,model,centered):
    vision_data=accumulated_local_effect(data,feature,n,model)
    if centered:
        plt.plot(vision_data[feature],vision_data['Centered_ALE'],color='black')
        plt.xlabel(feature)
        plt.ylabel("c-ALE")
        plt.title("Centered Accumulated Local Effects at {}".format(feature))
        plt.show() 
    else:
        plt.plot(vision_data[feature],vision_data['ALE'],color='black')
        plt.xlabel(feature)
        plt.ylabel("ALE")
        plt.title("Accumulated Local Effects at {}".format(feature))
        plt.show()
    

ale_plot(x,"temp",11,reg,centered=True)
ale_plot(x,"temp",11,reg,centered=False)