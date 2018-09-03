import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")

X_train = dataset.iloc[:,:-1]
submission = pd.read_csv("sample_submission.csv")

X_train1 = X_train.iloc[:,1:]
X_test1 = X_test.iloc[:,1:]

X = pd.concat((X_train1,X_test1))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X['Tag'] = labelencoder_1.fit_transform(X['Tag'])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
X = X[:,[0,1,2,3,4,5,6,7,8,9,10,12]]


X_train2 = X[0:330045,:]
X_test2 = X[330045:,:]
y_train = dataset.iloc[:,-1]

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train2, y_train)

y_pred = regressor.predict(X_test2)

submission["Upvotes"] = y_pred

submission.to_csv("sub3.csv",index= False)