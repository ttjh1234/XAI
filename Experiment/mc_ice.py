import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from data import data_preprocess
import matplotlib.pyplot as plt
from scipy.stats import beta

os.getcwd()

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

def partial_dependence_plot2(data,feature,model):
    interest_feature=np.sort(data[feature].unique())    
    data2=data.copy()

    a=np.repeat(interest_feature,data2.shape[0])
    for _ in range(interest_feature.shape[0]-1):
        data2=pd.concat([data2,data],axis=0)

    data2["temp"]=a

    y=model.predict(data2)
    y2=np.reshape(y,(interest_feature.shape[0],-1))
    fhat=np.mean(y2,axis=1)
    
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
    
    

def categorical_partial_dependence_plot(data,feature_list,model):
    fhat=[]
    data2=data.copy()
    one=np.ones((data2.shape[0],1)).astype(int)
    data2.loc[:,feature_list]=0
    y=model.predict(data)

    for i in range(len(feature_list)):
        data3=data2.copy()        
        data3[feature_list[i]]=one
        fhat.append(np.mean(model.predict(data3)))
    

    q3=np.quantile(y,q=0.75)
    q1=np.quantile(y,q=0.25)
    iqr=q3-q1
    
    feature=str.split(feature_list[0],'_')[0]
    feature_name=[str.split(i,'_')[1] for i in feature_list]
    plt.bar(feature_name,fhat,color="black")
    plt.xlabel(feature)
    plt.ylim(q1-iqr,q3+iqr)
    plt.ylabel("Model Predict")
    plt.title("Partial Dependence Plot about {}".format(feature))
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


individual_conditional_expectation_plot(x,"temp",reg)
centered_individual_conditional_expectation_plot(x,"temp",reg,-5.220871)

ale_plot(x,"temp",11,reg,centered=False)

def partial_dependence_plot(data,feature,model,mode='result',method='crude'):
    interest_feature=np.sort(data[feature].unique())   
    fhat=[]
    std_list=[]
    data2=data.copy()
    data2=data2.sort_values('temp').reset_index().drop('index',axis=1)
    
    if method=='crude':
        np.random.seed(41)
        index=np.random.choice(data2.shape[0],500)
        data2=data2.iloc[index,:].reset_index().drop('index',axis=1)
        for i in interest_feature:
            data2[feature]=i
            temp_result=model.predict(data2)
            mean_temp=np.mean(temp_result)
            std_temp=np.sqrt(np.var(temp_result)/500)
            fhat.append(mean_temp)
            std_list.append(std_temp)
    elif method=='antithetic':
        np.random.seed(41)
        index=np.random.choice(data2.shape[0],250)
        index2=data2.shape[0]-index
        data3=data2.iloc[index,:].reset_index().drop('index',axis=1)
        data4=data2.iloc[index2,:].reset_index().drop('index',axis=1)
        for i in interest_feature:
            data3[feature]=i
            data4[feature]=i
            temp_result1=model.predict(data3)
            temp_result2=model.predict(data4)
            temp_result=(temp_result1+temp_result2)/2
            mean_temp=np.mean(temp_result)
            std_temp=np.sqrt(np.var(temp_result)/500)
            fhat.append(mean_temp)
            std_list.append(std_temp)
    elif method=='control':
        np.random.seed(41)
        index=np.random.choice(data2.shape[0],500)
        data2=data2.iloc[index,:].reset_index().drop('index',axis=1)
        r=data2[feature].mean()
        feat=data2[feature]
        
        for i in interest_feature:
            data2[feature]=i
            temp_result=model.predict(data2)
            alpha=np.cov(temp_result,feat)[0,1]/np.var(feat)
            temp_result2=temp_result-alpha*(feat-r)
            mean_temp=np.mean(temp_result2)
            std_temp=np.sqrt(np.var(temp_result2)/500)
            fhat.append(mean_temp)
            std_list.append(std_temp)
    
    elif method=='stratified':
        n1=data2.loc[data2['season_Spring']==1,:].shape[0]
        n2=data2.loc[data2['season_Summer']==1,:].shape[0]
        n3=data2.loc[data2['season_Fall']==1,:].shape[0]
        n4=data2.loc[data2['season_Winter']==1,:].shape[0]
        p=np.array([n1,n2,n3,n4])/np.sum([n1,n2,n3,n4])
        
        index_list=[]
        
        total_n=0
        for n,feat in zip([n1,n2,n3,n4],['season_Spring','season_Summer','season_Fall','season_Winter']): 
            np.random.seed(41)
            n=int(n/(n1+n2+n3+n4)*500)
            total_n+=n
            index_list.append(np.random.choice(data2.loc[data2[feat]==1].index,n))
        
        for i in interest_feature:
            temp_var=[]
            temp_value=0
            for id,j in enumerate(index_list):
                data3=data2.iloc[j,:].reset_index().drop('index',axis=1)
                data3[feature]=i
                temp_result=model.predict(data3)
                temp_value+=np.sum(temp_result)
                temp_var.append(np.var(temp_result)/total_n*p[id])
            mean_temp=temp_value/total_n
            std_temp=np.sqrt(np.sum(temp_var))
            fhat.append(mean_temp)
            std_list.append(std_temp)
            
    elif method=='importance':
        np.random.seed(41)
        index=np.random.choice(data2.shape[0],500)
        data2=data2.iloc[index,:].reset_index().drop('index',axis=1)
        
        # variance minimization method
        # minimize 1/N sum(H(x_i)^2 * W(Xi;u,v))
        # find parameter v
        index2=index/731
        for i in interest_feature:
            data2[feature]=i
            temp_result=model.predict(data2)

            func_list=[]
            for a in np.arange(0.1,3,step=0.1):
                for b in np.arange(0.1,3,step=0.1):
                    func_list.append(np.mean(temp_result**2/(beta(a,b).pdf(index2))))
            number=np.argmin(func_list)
            np.arange(0.1,5,step=0.1).shape
            a,b=np.arange(0.1,5,step=0.1)[(number//29)],np.arange(0.1,5,step=0.1)[(number%29)]
            mean_temp=np.mean(temp_result/beta(a,b).pdf(index2))
            std_temp=np.sqrt(np.var(temp_result/beta(a,b).pdf(index2))/500)
            fhat.append(mean_temp)
            std_list.append(std_temp)
        
    
    max_value=np.max(fhat)
    min_value=np.min(fhat)
    q3=np.quantile(fhat,q=0.75)
    q1=np.quantile(fhat,q=0.25)
    iqr=q3-q1
    
    plt.plot(list(interest_feature),fhat,color="black",label='mean_value')
    if mode=='confidence1':
        plt.plot(list(interest_feature),np.array(fhat)-1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red",label='Asymtotic belt')
        plt.plot(list(interest_feature),np.array(fhat)+1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red")
    
    if mode=='confidence2':
        plt.plot(list(interest_feature),np.array(fhat)-np.array(std_list),color="red",label='Asymtotic belt')
        plt.plot(list(interest_feature),np.array(fhat)+np.array(std_list),color="red")
    
    plt.xlabel(feature)
    plt.ylabel("Model Predict")
    plt.legend()
    plt.title("Partial Dependence Plot about {}".format(feature))
    plt.show() 
   
    if (mode=='confidence1')|(mode=='confidence2'):
        plt.plot(list(interest_feature),std_list,color="black")    
        plt.xlabel(feature)
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviation Line about {}".format(feature))
        plt.show() 
        
        plt.hist(std_list,bins=int(np.sqrt(len(std_list))))
        plt.title("std dist")
        plt.show()
        
    if mode=='result':
        return interest_feature, fhat, std_list

partial_dependence_plot(x,"temp",reg,method='crude')
partial_dependence_plot(x,"temp",reg,mode='confidence2',method='crude')
partial_dependence_plot(x,"temp",reg,mode='confidence2',method='antithetic')
partial_dependence_plot(x,"temp",reg,mode='confidence2',method='control')
partial_dependence_plot(x,"temp",reg,mode='confidence2',method='stratified')
partial_dependence_plot(x,"temp",reg,mode='confidence2',method='importance')


a,b1,c1=partial_dependence_plot(x,"temp",reg,mode='result',method='crude')
_,b2,c2=partial_dependence_plot(x,"temp",reg,mode='result',method='antithetic')
_,b3,c3=partial_dependence_plot(x,"temp",reg,mode='result',method='control')
_,b4,c4=partial_dependence_plot(x,"temp",reg,mode='result',method='stratified')
_,b5,c5=partial_dependence_plot(x,"temp",reg,mode='result',method='importance')


a.shape

len(b1)
len(c1)



plt.hist(c1,bins=int(np.sqrt(len(c1))),color='red',alpha=0.5,label='crude')
plt.hist(c2,bins=int(np.sqrt(len(c1))),color='orange',alpha=0.5,label='antithetic')
plt.hist(c3,bins=int(np.sqrt(len(c1))),color='yellow',alpha=0.5,label='control')
plt.hist(c4,bins=int(np.sqrt(len(c1))),color='green',alpha=0.5,label='stratified')
plt.hist(c5,bins=int(np.sqrt(len(c1))),color='blue',alpha=0.5,label='importance')
plt.legend()
plt.xlabel('stadard error')
plt.ylabel('frequency')
plt.title("standard error distribution")
plt.show()

mc1=np.mean(c1)
mc2=np.mean(c2)
mc3=np.mean(c3)
mc4=np.mean(c4)
mc5=np.mean(c5)


np.mean(2*1.96*np.array(c1)/np.sqrt(500))

np.mean(2*1.96*np.array(c2)/np.sqrt(500))
np.mean(2*1.96*np.array(c3)/np.sqrt(500))
np.mean(2*1.96*np.array(c4)/np.sqrt(500))
np.mean(2*1.96*np.array(c5)/np.sqrt(500))

np.mean(np.array(c1)/np.array(b1))
np.mean(np.array(c2)/np.array(b2))
np.mean(np.array(c3)/np.array(b3))
np.mean(np.array(c4)/np.array(b4))
np.mean(np.array(c5)/np.array(b5))

a.shape
len(c1)
len(b1)

plt.plot(list(a),b1,color='black',label='Mean Effect')
for i,j,k,l in zip(np.array([b1,b1,b1,b1,b1]),np.array([c1,c2,c3,c4,c5]),['red','orange','yellow','green','blue'],['CMC','Antithetic','Control','Stratified','Importance']):
    plt.plot(list(a),i+2*j,color=k,label=l,alpha=0.4)
    plt.plot(list(a),i-2*j,color=k,alpha=0.4)
plt.xlabel('Temp',size=12)
plt.ylabel('Predict',size=12)
plt.legend()
plt.title('Estimate 95% C.I. of Marginal Effect each method')
plt.show()


