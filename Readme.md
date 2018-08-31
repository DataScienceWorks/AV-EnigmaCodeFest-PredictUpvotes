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

#### PUBLIC AND PRIVATE SPLIT

Note that the test data is further randomly divided into Public (30%) and Private (70%) data. Your initial responses will be checked and scored on the Public data.

The final rankings would be based on your private score which will be published once the competition is over.

	

## Solution

### References	

1. [A Complete Tutorial to Learn Data Science with Python from Scratch](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) 