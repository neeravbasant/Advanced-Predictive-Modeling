# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:49:24 2014

@author: Neerav Basant
"""

from sklearn.linear_model import LogisticRegression as regr
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random

#Importing data
df = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 4/hw4files/spam.csv")

#Splitting in  train and test
df_train = df[df["test"] == False]
df_test = df[df["test"] == True]

#Standardizing the datasets
X_train, X_test = df_train.ix[:, :57].as_matrix(), df_test.ix[:, :57].as_matrix()
y_train, y_test = df_train.ix[:, 57].as_matrix(), df_test.ix[:, 57].as_matrix()
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4(a)
lr = regr()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
print "Logistic Regression Accuracy: ", accuracy_score(y_test, pred_lr)

# 4(b)
random.seed(1)

def sample(df):
    indx = np.random.randint(0, len(df), 1000)
    return df.iloc[indx]

def scale(dfTr, xTe):
    xTr = dfTr[dfTr.columns[:57]]
    yTr = dfTr[dfTr.columns[57]] 
    scaler = preprocessing.StandardScaler().fit(xTr) 
    xTr = scaler.transform(xTr) 
    xTe = scaler.transform(xTe)
    return (xTr, yTr, xTe)
    
def logit(x,y, xTe, yTe):
    logit = regr()
    modLogit = logit.fit(x, y)
    resTest = modLogit.predict(xTe)
    return (resTest == yTe)

n = [1, 11, 21, 31]
for j in n:
    dfRes = pd.DataFrame()
    for i in range(j):
        dat = sample(df_train)
        xTrn, yTrn, xTst = scale(dat, X_test)
        dfRes[str(i)] = logit(xTrn, yTrn, xTst, y_test)
    dfRes['Overall'] = y_test
    for index, row in dfRes.iterrows():
        dfRes['Overall'].ix[index] = 2*sum(row[:str(j-1)]) > j
    print "Ensemble Logistic Regression Accuracy (n = ", j, "): ", sum(dfRes['Overall'])/float(len(y_test))
