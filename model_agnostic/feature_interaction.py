# Feature Interaction 

'''

When features interact with each other in a prediction model, the prediction cannot be
expressed as the sum of the feature effects, because the effect of one feature depends on
the value of the other feature.

Theory: Friedman's H-Statistic

If two features do not interact, we can decompose the partial dependence function as follows.
(assuming the partial dependence functions are centered at zero)

PD_jk(x_j,x_k)=PD_j(x_j)+PD_k(x_k)

where PD_jk(x_j,x_k) is the 2-way partial dependence function of both features and 
PD_j(x_j) and PD_k(x_k) the partial dependence functions of the single features.

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

def individual_partial_dependence_function(data,feature,model):
    interest_feature=np.sort(data[feature].unique())    
    fhat=[]
    data2=data.copy()
    
    for i in interest_feature:
        data2[feature]=i
        fhat.append(np.mean(model.predict(data2)))
    fhat2=pd.Series(fhat)
    interest_feature2=pd.Series(interest_feature)
    result=pd.concat([interest_feature2,fhat2],axis=1)
    result.columns=[feature,'{}_fhat'.format(feature)]
    return result


def collective_partial_dependence_function(data,feature_list,model):
    interest_feature=np.array(data[feature_list])
    fhat=[]
    data2=data.copy()
    
    for i in interest_feature:
        data2[feature_list]=i
        fhat.append(np.mean(model.predict(data2)))
    
    interest_feature2=pd.DataFrame(interest_feature,columns=feature_list)
    fhat2=pd.Series(fhat)
    result=pd.concat([interest_feature2,fhat2],axis=1)
    feature_list.append('fhat')
    result.columns=feature_list
    return result


def feature_interaction(data,feature_list,model):
    if len(feature_list)==2:
        pdi1=individual_partial_dependence_function(data,feature_list[0],model)
        pdi2=individual_partial_dependence_function(data,feature_list[1],model)
        pdc=collective_partial_dependence_function(data,feature_list,model)
        result=pd.merge(pdc,pdi1,left_on=feature_list[0],right_on=feature_list[0])
        result=pd.merge(result,pdi2,left_on=feature_list[1],right_on=feature_list[1])
        result['result']=((result['fhat']-result[feature_list[0]+"_"+"fhat"]-result[feature_list[1]+"_"+"fhat"])**2)/np.sum(result['fhat']**2)
        return np.sum(result['result'])
    else:
        pdi1=individual_partial_dependence_function(data,feature_list[0],model)
        allfeature=list(data.columns)
        allfeature.remove(feature_list[0])
        pdc=collective_partial_dependence_function(data,allfeature,model)
        pdc[feature_list[0]]=data[feature_list[0]]
        result=pd.merge(pdc,pdi1,left_on=feature_list[0],right_on=feature_list[0])
        pred=model.predict(data)
        result['result']=((pred-result[feature_list[0]+"_"+"fhat"]-result['fhat'])**2)/np.sum(pred**2)
        return np.sum(result['result'])

