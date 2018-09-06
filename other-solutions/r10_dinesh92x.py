import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Trainig data set

dataset=pd.read_csv('train_NIR5Yl1.csv')
dataset.head(5)

X = dataset.drop(['Tag','ID','Username'], axis=1)
y = X.Upvotes                    
X=X.drop(['Upvotes'], axis=1)      

X.head(10)

from sklearn.model_selection import train_test_split
X, x_test, y, y_test = train_test_split(X,y,test_size=.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X=scalar.fit_transform(X)
x_test=scalar.fit_transform(x_test)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X=poly_reg.fit_transform(X)

poly_reg.fit(X,y)

x_test=poly_reg.fit_transform(x_test)
#_______________________________________________________
#                      NOW FOR TESTING DATA SET  
test_dataset=pd.read_csv('test_8i3B3FC.csv')
aa=test_dataset.describe()

X_test= test_dataset.drop(['Tag','ID','Username'], axis=1)

X_test=scalar.fit_transform(X_test)
    

X_testing= poly_reg.fit_transform(X_test)
#_______________________________________________________
#                      APPLING MODELS

from sklearn.linear_model import LinearRegression
classifier = LinearRegression()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
np.mean(accuracies)

classifier.fit(X,y)
y_pred=classifier.predict(x_test)
#______________________________________________________
#                      MEAN SQUARD ERROR
from sklearn.metrics import mean_squared_error
rms = np.sqrt(mean_squared_error(y_test, y_pred))

yy_pred=classifier.predict(X_testing)

y_pred=yy_pred.astype(int)
#                     SAVING AS CSV
np.savetxt('PRED_FILE.csv',y_pred,delimiter='\t')
