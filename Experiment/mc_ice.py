import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from data import data_preprocess
import matplotlib.pyplot as plt

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

def partial_dependence_plot(data,feature,model):
    interest_feature=np.sort(data[feature].unique())    
    fhat=[]
    std_list=[]
    data2=data.copy()
    
    for i in interest_feature:
        data2[feature]=i
        temp_result=model.predict(data2)
        mean_temp=np.mean(temp_result)
        std_temp=np.std(temp_result)
        fhat.append(mean_temp)
        std_list.append(std_temp)
    
    max_value=np.max(fhat)
    min_value=np.min(fhat)
    q3=np.quantile(fhat,q=0.75)
    q1=np.quantile(fhat,q=0.25)
    iqr=q3-q1
    
    plt.plot(list(interest_feature),fhat,color="black",label='mean_value')
    plt.plot(list(interest_feature),np.array(fhat)-1.96*np.array(std_list)/np.sqrt(std_list),color="red",label='Asymtotic belt')
    plt.plot(list(interest_feature),np.array(fhat)+1.96*np.array(std_list)/np.sqrt(std_list),color="red")
    
    plt.xlabel(feature)
    #plt.ylim(min_value-1.5*iqr,max_value+1.5*iqr)
    plt.ylabel("Model Predict")
    plt.legend()
    plt.title("Partial Dependence Plot about {}".format(feature))
    plt.show() 
   
    plt.plot(list(interest_feature),std_list,color="black")    
    plt.xlabel(feature)
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation Line about {}".format(feature))
    plt.show() 
    
    plt.hist(std_list,bins=int(np.sqrt(len(std_list))))
    plt.title("std dist")
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


partial_dependence_plot(x,"temp",reg)
partial_dependence_plot2(x,"temp",reg)
partial_dependence_plot(x,"hum",reg)
partial_dependence_plot(x,"windspeed",reg)


individual_conditional_expectation_plot(x,"temp",reg)
centered_individual_conditional_expectation_plot(x,"temp",reg,-5.220871)

ale_plot(x,"temp",11,reg,centered=False)

def partial_dependence_plot(data,feature,model):
    interest_feature=np.sort(data[feature].unique())   
    fhat=[]
    std_list=[]
    data2=data.copy()
    #data2['temp'].hist(bins=int(np.sqrt(731)))
    np.random.seed(42)
    index=np.random.choice(data2.shape[0],500)
    data2=data2.iloc[index,:].reset_index().drop('index',axis=1)
    
    for i in interest_feature:
        data2[feature]=i
        temp_result=model.predict(data2)
        mean_temp=np.mean(temp_result)
        std_temp=np.std(temp_result)
        fhat.append(mean_temp)
        std_list.append(std_temp)
    
    max_value=np.max(fhat)
    min_value=np.min(fhat)
    q3=np.quantile(fhat,q=0.75)
    q1=np.quantile(fhat,q=0.25)
    iqr=q3-q1
    
    plt.plot(list(interest_feature),fhat,color="black",label='mean_value')
    plt.plot(list(interest_feature),np.array(fhat)-1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red",label='Asymtotic belt')
    plt.plot(list(interest_feature),np.array(fhat)+1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red")
    
    plt.xlabel(feature)
    #plt.ylim(min_value-1.5*iqr,max_value+1.5*iqr)
    plt.ylabel("Model Predict")
    plt.legend()
    plt.title("Partial Dependence Plot about {}".format(feature))
    plt.show() 
   
    plt.plot(list(interest_feature),std_list,color="black")    
    plt.xlabel(feature)
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation Line about {}".format(feature))
    plt.show() 
    
    plt.hist(std_list,bins=int(np.sqrt(len(std_list))))
    plt.title("std dist")
    plt.show()

def partial_dependence_plot(data,feature,model,mode='result',method='crude'):
    interest_feature=np.sort(data[feature].unique())   
    fhat=[]
    std_list=[]
    data2=data.copy()
    data2=data2.sort_values('temp').reset_index().drop('index',axis=1)
    #data2['temp'].hist(bins=int(np.sqrt(731)))
    
    if method=='crude':
        np.random.seed(41)
        index=np.random.choice(data2.shape[0],500)
        data2=data2.iloc[index,:].reset_index().drop('index',axis=1)
        for i in interest_feature:
            data2[feature]=i
            temp_result=model.predict(data2)
            mean_temp=np.mean(temp_result)
            std_temp=np.std(temp_result)
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
            temp_result=temp_result1+temp_result2
            mean_temp=np.mean(temp_result)/2
            std_temp=np.std(temp_result)+1/500*np.cov(temp_result1,temp_result2)[0,1]
            fhat.append(mean_temp)
            std_list.append(std_temp)
    
    max_value=np.max(fhat)
    min_value=np.min(fhat)
    q3=np.quantile(fhat,q=0.75)
    q1=np.quantile(fhat,q=0.25)
    iqr=q3-q1
    
    plt.plot(list(interest_feature),fhat,color="black",label='mean_value')
    if mode=='confidence':
        plt.plot(list(interest_feature),np.array(fhat)-1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red",label='Asymtotic belt')
        plt.plot(list(interest_feature),np.array(fhat)+1.96*np.array(std_list)/np.sqrt(len(std_list)),color="red")
    
    plt.xlabel(feature)
    #plt.ylim(min_value-1.5*iqr,max_value+1.5*iqr)
    plt.ylabel("Model Predict")
    plt.legend()
    plt.title("Partial Dependence Plot about {}".format(feature))
    plt.show() 
   
    if mode=='confidence':
        plt.plot(list(interest_feature),std_list,color="black")    
        plt.xlabel(feature)
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviation Line about {}".format(feature))
        plt.show() 
        
        plt.hist(std_list,bins=int(np.sqrt(len(std_list))))
        plt.title("std dist")
        plt.show()

partial_dependence_plot(x,"temp",reg,method='crude')
partial_dependence_plot(x,"temp",reg,mode='confidence',method='crude')

partial_dependence_plot(x,"temp",reg,method='antithetic')









