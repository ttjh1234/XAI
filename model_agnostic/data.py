# Example UCI Datasets : Bike Sharing Dataset

import pandas as pd

def data_preprocess(path):
    weekday_list=["SUN","MON","TUE","WED","THU","FRI","SAT"]
    holiday_list=["No Holiday", "Holiday"]
    workingday_list=["No Working Day","Working Day"]
    season_list=["Winter","Spring","Summer","Fall"]
    weathersit_list=["Good","Misty","Rain/Snow/Storm"]
    month_list=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    year_list=[2011,2012]   
    data=pd.read_csv(path)
    data["weekday"]=data["weekday"].map(lambda x : weekday_list[x])
    data["holiday"]=data["holiday"].map(lambda x : holiday_list[x])
    data["workingday"]=data["workingday"].map(lambda x : workingday_list[x])
    data["season"]=(data["season"]-1).map(lambda x : season_list[x])
    data["weathersit"]=(data["weathersit"]-1).map(lambda x : weathersit_list[x])
    data["mnth"]=(data["mnth"]-1).map(lambda x : month_list[x])
    data["yr"]=data["yr"].map(lambda x : year_list[x])
    data["days_since_2011"]=data.index
    # Dataset already normalization (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 
    data["temp"]=data["temp"]*47-8
    # Dataset already normalization (t-t_min)/(t_max-t_min), t_min=16, t_max=+59
    data["atemp"]=data["atemp"]*34+16
    # Dataset already normalization, divided to 67 (max)
    data["windspeed"]=data["windspeed"]*67
    # Dataset already normalization, divided to 100 (max)
    data["hum"]=data["hum"]*100
    
    data2=data.drop(["instant","dteday","atemp","casual","registered"],axis=1)
        
    return data2