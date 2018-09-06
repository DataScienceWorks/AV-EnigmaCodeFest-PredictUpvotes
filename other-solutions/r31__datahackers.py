#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:46:44 2018

@author: pswaldia1
"""

#importing dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#readind data
dataset=pd.read_csv('train_NIR5Yl1.csv')

X=dataset.iloc[:,1:6].values
y=dataset.iloc[:,6].values

#preprocessing: no splitting is required

#lable encoding required fot tags
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lable_tag=LabelEncoder()
onehotencoding=OneHotEncoder(categorical_features=[0])
X[:,0]=lable_tag.fit_transform(X[:,0])
X=onehotencoding.fit_transform(X).toarray()
#to avoid dummy variable trap, remove one column
X=X[:,1:]   


#standardizing the dataset......to bring the different features in same order

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X[:,9:]=sc.fit_transform(X[:,9:])
rescaledX=X[:,:]



df=pd.read_csv('test_8i3B3FC.csv')
x_test=pd.read_csv('test_8i3B3FC.csv')
x_test=x_test.iloc[:,1:].values
lable_tag_test=LabelEncoder()
onehotencoding_test=OneHotEncoder(categorical_features=[0])
x_test[:,0]=lable_tag_test.fit_transform(x_test[:,0])
x_test=onehotencoding_test.fit_transform(x_test).toarray()
#to avoid dummy variable trap, remove one column
x_test=x_test[:,1:]
x_test[:,9:]=sc.fit_transform(x_test[:,9:])
x_test.shape

import xgboost as xgb
dtrain = xgb.DMatrix(rescaledX, y)

# cross validation steps not included
params2 = {
    #Parameters after tuning
    'max_depth':5,
    'min_child_weight': 4,
    'eta':.05,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'objective':'reg:linear',
}

params2['eval_metric'] = "rmse"
num_boost_round = 600
model = xgb.train(
    params2,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, 'Train')],
    early_stopping_rounds=10
)




''''
Given Below is the function that is required to preprocess the test data before prediction
This is meant for another 70% test dataa.....test data given to us has already been 
preprocessed and result has been stored in a .csv file .
'''
def test_data_preprocess(testdataFrame):
    '''
    passed arguement is the pandas dataframe
    returned value: The testData Array that needs to be passed to xgb.DMatrix()
    
    
    '''
    
    test_data=testdataFrame.iloc[:,1:].values
    lable_tag_test=LabelEncoder()
    onehotencoding_test=OneHotEncoder(categorical_features=[0])
    test_data[:,0]=lable_tag_test.fit_transform(test_data[:,0])
    test_data=onehotencoding_test.fit_transform(test_data).toarray()
    #to avoid dummy variable trap, remove one column
    test_data=test_data[:,1:]
    test_data[:,9:]=sc.fit_transform(test_data[:,9:])
    return test_data

#the value returned from above function is to be passed to the given xgb.dMatrix( ) method
    


dtest = xgb.DMatrix(x_test)
pred=model.predict(dtest)
submission=pd.concat([df['ID'],pd.DataFrame(np.round(pred))],axis=1)
submission.columns=['ID','Upvotes']
submission['Upvotes']=submission.apply(lambda x:1 if (x['Upvotes']<=0) else x['Upvotes'],axis=1)

#submission is the required dataframe that contains the predictions on the test data


    
    







