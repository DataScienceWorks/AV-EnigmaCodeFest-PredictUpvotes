# Best Question Author Prediction - Enigma CodeFest - Analytics Vidya

Source: https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning/

Leaderboard: https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning/lb



## Overview

The Department of Computer Science and Engineering at IIT(BHU) Varanasi is proud to present the fifth instalment of its highly anticipated coding festival, Codefest, that will be held from 31st August - 2nd September 2018.The previous editions have witnessed remarkable success at a global level - from the inaugural edition in 2010 to the rebooted 2016 edition which was especially remarkable in terms of its reach.



## Problem Statement

#### Problem
An online question and answer platform has hired you as a data scientist to identify the best question authors on the platform. This identification will bring more insight into increasing the user engagement. Given the tag of the question, number of views received, number of answers, username and reputation of the question author, the problem requires you to predict the upvote count that the question will receive.

#### DATA DICTIONARY

| **Variable** | **Definition**                                 |
| ------------ | ---------------------------------------------- |
| ID           | Question ID                                    |
| Tag          | Anonymised tags representing question category |
| Reputation   | Reputation score of question author            |
| Answers      | Number of times question has been answered     |
| Username     | Anonymised user id of question author          |
| Views        | Number of times question has been viewed       |
| Upvotes      | (Target) Number of upvotes for the question    |

#### EVALUATION METRIC

The evaluation metric for this competition is **RMSE (root mean squared error)**

#### Leaderboard Rankings and Score

| Public LB Rank | Public Score       | Pvt LB Rank | ML Model                                  |
| -------------- | ------------------ | ----------- | ----------------------------------------- |
| 77             | 3543.8523122425    |             | LinearRegression                          |
| 22             | 1100.3336222340    |             | BaggingRegressor                          |
| 12             | 1016.7805765708    |             | BaggingRegressor                          |
|                | 1452.xxx           |             | BaggingRegressor(PCA=5)                   |
|                | 1268.1448673016037 |             | BaggingRegressor(PCA=5 and Grid)          |
|                | 1601.xx            |             | AdaBoostRegressor                         |
| 45             | 1177.7464239328351 |             | GradientBoostingRegressor(default params) |



#### PUBLIC AND PRIVATE SPLIT (Leader Board thing)

Note that the test data is further randomly divided into Public (30%) and Private (70%) data. Your initial responses will be checked and scored on the Public data.

The final rankings would be based on your private score which will be published once the competition is over.

	

## Solution

### References	

1. [A Complete Tutorial to Learn Data Science with Python from Scratch](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) 
2. [Pipelines, FeatureUnions, GridSearchCV, and Custom Transformers](https://blog.pursuitofzen.com/pipelines-featureunions-gridsearchcv-and-custom-transformers/) -- GoodReads
3. [~~A new categorical encoder for handling categorical features in scikit-learn~~](https://jorisvandenbossche.github.io/blog/2017/11/20/categorical-encoder/)
4. [Feature Union with Heterogeneous Data Sources](http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html) -- GoodReads
5. [Building Scikit-Learn Pipelines With Pandas DataFrames](https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/) -- GoodReads
6. [Using scikit-learn Pipelines and FeatureUnions](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html) -- GoodReads
7. [StackOverflow - Unable to use FeatureUnion to combine processed numeric and categorical features in Python](https://stackoverflow.com/questions/48994618/unable-to-use-featureunion-to-combine-processed-numeric-and-categorical-features)
8. [StackOverflow - Issue with OneHotEncoder for categorical features](https://stackoverflow.com/a/43589167/498604)
9. [StackOverflow - How to make pipeline for multiple dataframe columns?](https://stackoverflow.com/questions/47895434/how-to-make-pipeline-for-multiple-dataframe-columns)
10. [StackOverflow - How many principal components to take?](https://stackoverflow.com/a/12073948) 
   To decide how many eigenvalues/eigenvectors to keep, you should consider your reason for doing PCA in the first place. Are you doing it for reducing storage requirements, to reduce dimensionality for a classification algorithm, or for some other reason? If you don't have any strict constraints, I recommend plotting the cumulative sum of eigenvalues (assuming they are in descending order). If you divide each value by the total sum of eigenvalues prior to plotting, then your plot will show the fraction of total variance retained vs. number of eigenvalues. The plot will then provide a good indication of when you hit the point of diminishing returns (i.e., little variance is gained by retaining additional eigenvalues).
11. [Github -- kennethclitton/Kaggle-College-Students-on-Loans](https://github.com/kennethclitton/Kaggle-College-Students-on-Loans/blob/master/regression_challenge.py)
12. [PCA using Python (scikit-learn)](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
13. [scikit-learn Doc - PCA example with Iris Data-set](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)
14. [API - FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) : 
    `class sklearn.pipeline.FeatureUnion(transformer_list, n_jobs=1, transformer_weights=None)`.
    1. Concatenates results of multiple transformer objects.
    2. This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results. This is useful to combine several feature extraction mechanisms into a single transformer.
    3. Parameters of the transformers may be set using its name and the parameter name separated by a ‘__’. A transformer may be replaced entirely by setting the parameter with its name to another transformer, or removed by setting to `None`.
15. [API - Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) :
    `class sklearn.pipeline.Pipeline(steps, memory=None)`.
    1. Pipeline of transforms with a final estimator.
    2. Sequentially apply a list of transforms and a final estimator.
    3. Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods.
    4. The final estimator only needs to implement fit.
    5. The transformers in the pipeline can be cached using `memory` argument.
    6. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their
       names and the parameter name separated by a ‘__’, as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting to `None`.
16. [API - make_pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) :
    ``sklearn.pipeline.``make_pipeline`(**steps*, **\*kwargs*)`.
    1. Construct a Pipeline from the given estimators.
    2. This is a shorthand for the Pipeline constructor; it does not require, and does not permit, naming the estimators. Instead, their names will be set to the lowercase of their types automatically.
17. [ML-Ensemble: Scikit-learn style ensemble learningflennerhag](https://www.kaggle.com/flennerhag/ml-ensemble-scikit-learn-style-ensemble-learning)
18. [StackOverflow - Ensemble of different kinds of regressors using scikit-learn (or any other python framework)](https://stackoverflow.com/questions/28727709/ensemble-of-different-kinds-of-regressors-using-scikit-learn-or-any-other-pytho)
19. [mlxtend - StackingRegressor](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/)
20. [scikit-learn Doc - Decision Tree Regression with AdaBoost](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py)
21. [Boosting and Bagging: How To Develop A Robust Machine Learning Algorithm](https://hackernoon.com/how-to-develop-a-robust-algorithm-c38e08f32201)
22. [Kaggle -- GridSearchCV + XGBRegressor (0.556+ LB)](https://www.kaggle.com/omarito/gridsearchcv-xgbregressor-0-556-lb)