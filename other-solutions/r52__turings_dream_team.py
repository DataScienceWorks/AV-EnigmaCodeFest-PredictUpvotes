import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

# Removing outliers
"""upvotes_bound = train_data['Upvotes'].mean() + 3*train_data['Upvotes'].std()
train_data = train_data[train_data['Upvotes']<upvotes_bound]
views_bound = train_data['Views'].mean() + 3*train_data['Views'].std()
train_data = train_data[train_data['Views']<views_bound]
rep_bound = train_data['Reputation'].mean() + 3*train_data['Reputation'].std()
train_data = train_data[train_data['Reputation']<rep_bound]"""


train_data = pd.get_dummies(train_data)

"""cols = ['Reputation','Answers','Views','Tag_a','Tag_c','Tag_h','Tag_i',
               'Tag_j','Tag_o','Tag_p','Tag_r','Tag_s','Tag_x']"""
cols = ['Reputation','Answers','Views']

X = train_data[cols]
y = train_data[['Upvotes']]

X = X.values
y = y.values.ravel()

# Scaling
"""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)"""
y_scaled = np.log1p(y)

# Splitting the dataset

#r = np.random.randint(0,150)

r = 83

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size = 0.3, random_state = r)

# Fitting and Predicting
from xgboost import XGBRegressor
#regressor = XGBRegressor(n_estimators=100)
regressor = XGBRegressor(n_estimators=500,learning_rate=0.1,max_depth=6,
                         gamma=0.5,subsample=0.5,n_jobs=-1,silent=False,eval_metric='rmse',
                         early_stopping_rounds=10)

regressor.fit(X=X_train,y=y_train)

y_pred = regressor.predict(X_test)
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)
y_pred[y_pred<0] = 0
y_pred = np.trunc(y_pred)

full_test = pd.DataFrame(X_test, columns = cols)

full_test['Actual'] = y_test
full_test['Predicted'] = y_pred
full_test['Diff'] = np.abs(y_pred - y_test)

sns.regplot(full_test['Actual'], full_test['Diff'])

# Evaluating
rmse = np.sqrt(mean_squared_error(y_pred,y_test))

importances = pd.DataFrame()
importances['Feature'] = cols
importances['Importance'] = regressor.feature_importances_

sns.barplot(importances['Feature'], importances['Importance'])

# Tuning parameters
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

subsample_list = [0.5,0.75,1]
colsample_bytree_list = [0.5,0.75,1]
min_child_weight_list = [1,2,4]

xgb = XGBRegressor(n_estimators=100,learning_rate=0.01,max_depth=6,
                   gamma=0.1,n_jobs=-1)

grid_search = GridSearchCV(xgb,dict(subsample=subsample_list,colsample_bytree=colsample_bytree_list,
                                    min_child_weight = min_child_weight_list),cv=3,verbose=True)

grid_search.fit(X,y_scaled)

grid_search.best_estimator_
grid_search.best_score_

scores = [x[1] for x in grid_search.grid_scores_]
scores = np.array(scores).reshape(len(estimators), len(max_depth_list))

for ind, i in enumerate(estimators):
    plt.plot(max_depth_list, scores[ind], label='No. of estimators: ' + str(i))
plt.legend()
plt.xlabel('Max_Depth')
plt.ylabel('Mean score')
plt.show()

grid_search.grid_scores_


# Visualisation
sns.regplot(train_data['Reputation'],train_data['Upvotes'])
sns.regplot(train_data['Answers'],train_data['Upvotes'])
sns.regplot(train_data['Views'],train_data['Upvotes'])

sns.distplot(np.log(y+1), hist = False)
sns.distplot(np.cbrt(y_scaled), hist = False)

sns.distplot(train_data['Reputation'])
sns.distplot(test_data['Reputation'])

train_data['Tag'].value_counts()

sns.barplot(train_data['Tag'],train_data['Upvotes'])

sns.heatmap(train_data.corr(), annot=True, fmt="0.2f", cmap='RdGy', center=0)

# Exporting the solution
regressor.fit(X,y_scaled)

test_data = pd.get_dummies(test_data)
X_final = test_data[cols]

X_final = X_final.values

y_pred_final = regressor.predict(X_final)

y_pred_final = np.expm1(y_pred_final)
y_pred_final[y_pred_final<0] = 0
y_pred_final = np.trunc(y_pred_final).astype(int)

submission = pd.DataFrame(test_data['ID'], columns = ['ID'])
submission['Upvotes'] = y_pred_final

submission.to_csv('xgboost_12.csv', index = False)