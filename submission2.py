import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
from dateutil.parser import parse
import time
from sklearn.decomposition import PCA
from scipy.linalg import *

def data_restructure(data):
    data=data.replace(np.nan, 0)
    data["total_positive_reviews"] +=1 #adding 1 to avoid value 0
    data["total_negative_reviews"] += 1
    pur_date=pd.to_datetime(data["purchase_date"]) #computing the date difference
    rel_date=pd.to_datetime(data["release_date"])
    data["date_diff"]=(pur_date-rel_date).dt.days
    data["heat"]=data["total_positive_reviews"]+data["total_negative_reviews"] #creating a new feature "heat"
    data["good_ratio"]=data["total_positive_reviews"]/data["total_negative_reviews"] #creating a new feature "good_ratio"
    #dummy data
    g_dummy=data["genres"].str.get_dummies(',')
    c_dummy = data['categories'].str.get_dummies(",")
    t_dummy = data['tags'].str.get_dummies(",")
    #rename
    g_dummy.rename(columns=lambda y:"g" + y,inplace=True)
    c_dummy.rename(columns=lambda y: "c" + y,inplace=True)
    t_dummy.rename(columns=lambda y: "t" + y,inplace=True)
    #combine dataframe
    final_data=pd.concat([data,g_dummy,c_dummy,t_dummy],axis=1)
    return final_data

#to save the model
def train_model(x,y):
    model=RandomForestRegressor(n_estimators=800, random_state=3, max_depth=80)
    model.fit(x,y)
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

def test_fct(data):
    with open('model.pickle', 'rb') as f:
        predition=pickle.load(f)
    result=predition.predict(x_test)
    #pd.DataFrame(result,columns=['playtime_forever']).to_csv('result1110.csv')
    resulting_test=pd.DataFrame(result,columns=['playtime_forever'])# the result
    resulting_test.to_csv('C:/Users/LENOVO/Desktop/resulting_test.csv',index_label=["id"])# please change the direction of the file

if __name__ == '__main__':
    traindata=pd.read_csv("C:/Users/LENOVO/Desktop/train.csv") # please change the direction of the file
    traindata.sort_values("total_positive_reviews",inplace=True,ascending=False)
    traindata["playtime_forever"][0:30]=traindata["playtime_forever"][0:30]*3 # increasing the weight of the factor "total_positive_reviews"
    testdata=pd.read_csv("C:/Users/LENOVO/Desktop/test.csv")# please change the direction of the file
    y=traindata["playtime_forever"]
    x1=data_restructure(traindata)
    x=x1.drop(["id","playtime_forever","purchase_date","release_date","is_free","genres","categories","tags","date_diff"],axis=1)
    features=x.keys()
    pca = PCA(n_components=35) #using pca to get the most important 35 factors
    pca.fit(x)
    x_train=pca.transform(x)
    train_model(x_train,y)

    x_test=data_restructure(testdata)
    x_test = x_test.drop(["id", "purchase_date", "release_date", "is_free", "genres", "categories", "tags","date_diff"],
                axis=1)


    for fa in features:
       if x_test.get(fa) is None:
          x_test[fa]=0
    x_test=x_test[x.keys()]
    x_test=pca.transform(x_test)
    #the results
    test_fct(x_test)






