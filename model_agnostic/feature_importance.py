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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=data_preprocess(r"C:\Users\UOS\Desktop\data\Bike-Sharing-Dataset\day.csv")

feat_list=[i for n,i in enumerate(data.columns) if n!=10]
x=data.loc[:,feat_list]
y=data["cnt"]
x_num=x.loc[:,'temp':]
x_cat=x.loc[:,:'weathersit']
ss=StandardScaler()
x_num2=ss.fit_transform(x_num)
x2=pd.DataFrame(x_num2,columns=list(x_num.columns))
x=pd.concat([x_cat,x2],axis=1)
x=pd.get_dummies(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)


reg=SVR(kernel="rbf", C=1000, gamma=0.001, epsilon=0.1)
reg.fit(x_train,y_train)

yhat=reg.predict(x_train)
yhat_test=reg.predict(x_test)
plt.plot(y_train.reset_index().drop(['index'],axis=1), color='green',linewidth=5, alpha=0.5, label='True')
plt.plot(yhat,color="yellow",alpha=0.5,label="Predict")
plt.xlabel('Instance')
plt.ylabel('Cnt')
plt.title("Model Result")
plt.legend()
plt.show()


plt.plot(y_test.reset_index().drop(['index'],axis=1), color='green',linewidth=5, alpha=0.5, label='True')
plt.plot(yhat_test,color="yellow",alpha=0.5,label="Predict")
plt.xlabel('Instance')
plt.ylabel('Cnt')
plt.title("Model Result")
plt.legend()
plt.show()


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
    
    k=-1
    
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
            if i in str.split(j,"_"):
                calculate_name.remove(j)
    
    calculate_name=calculate_name+prefix
    e_perm=create_perm_error_dict(calculate_name)
    
        
    for j in calculate_name:
        temp=x.copy()
        if j in prefix:
            for _ in range(n):
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
            for _ in range(n):
                temp[j]=np.random.permutation(x[j])
                yperm=model.predict(temp)
                permutation_error=loss(y,yperm)
                if rate:
                    permutation_error=permutation_error/e_origin
                else:
                    permutation_error=permutation_error-e_origin
                e_perm[j].append(permutation_error)
               
    return e_perm

result=feature_importance(x,y,reg,mean_absolute_error,rate=True)

    




