# Feature Importance 

'''

The importance of a feature is the increase in the prediction error of the model after we 
permuted the feature's values, which breaks the relationship between the feature and the true outcome.


'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import os
from data import data_preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


data=data_preprocess(r"C:\Users\UOS\Desktop\data\Bike-Sharing-Dataset\day.csv")

feat_list=[i for n,i in enumerate(data.columns) if n!=10]
x=data.loc[:,feat_list]
y=data["cnt"]
x=pd.get_dummies(x)

reg=SVR(kernel="rbf",C=1000,gamma=0.001)
reg.fit(x,y)

yhat=reg.predict(x)

plt.plot(y, color='green',linewidth=5, alpha=0.5, label='True')
plt.plot(yhat,color="yellow",alpha=0.5,label="Predict")
plt.xlabel('Instance')
plt.ylabel('Cnt')
plt.title("Model Result")
plt.legend()
plt.show()

e_origin=mean_absolute_error(y,yhat)

def create_perm_error_dict(column_name):
    result={}
    for i in column_name:
        result[i]=[]
    return result

def feature_importance(x,y,model,loss,n=10,rate=True):
    yhat=model.predict(x)
    e_origin=loss(y,yhat)
    
    prefix=[]
    postfix={}
    
    for i in list(x.columns):
        if len(str.split(i,'_'))==2:
            if str.split(i,'_')[0] not in prefix:
                k+=1            
                prefix.append(str.split(i,'_')[0])
                postfix[str.split(i,'_')[0]]=[]
            if str.split(i,'_')[1] not in postfix:
                postfix[str.split(i,'_')[0]].append(str.split(i,'_')[1])
    
    column_name=list(x.columns)
    calculate_name=column_name.copy()
    for i in prefix:
        for j in column_name:
            print(i,j)
            if i in str.split(j,"_"):
                calculate_name.remove(j)
    
    calculate_name=calculate_name+prefix
    e_perm=create_perm_error_dict(calculate_name)
    
        
    for j in calculate_name:
        temp=x
        if j in prefix:
            sample=np.random.multinomial(temp.shape[0],[1/len(postfix[j])]*len(postfix[j]))
            perm_data=np.zeros((0,len(postfix[j])))
            for v in range(len(postfix[j])):
                vec1=np.concatenate([np.zeros((sample[v],v)),np.ones((sample[v],1)),np.zeros((sample[v],len(postfix[j])-v-1))],axis=1)
                perm_data=np.concatenate([perm_data,vec1],axis=0)        
            perm_data=np.random.permutation(perm_data)
            featname=[j+'_'+g for g in postfix[j]]
            temp[featname]=perm_data
            yperm=model.predict(temp)
            permutation_error=loss(y,yperm)
            if rate:
                permutation_error=permutation_error/e_origin
            else:
                permutation_error=permutation_error-e_origin
            
            e_perm[j].append(permutation_error) 
            
        else:
            for i in range(n):
                temp[j]=np.random.permutation(x[j])
                yperm=model.predict(temp)
                permutation_error=loss(y,yperm)
                if rate:
                    permutation_error=permutation_error/e_origin
                else:
                    permutation_error=permutation_error-e_origin
                e_perm[j].append(permutation_error)
               
    return e_perm
    




