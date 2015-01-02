# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 23:54:07 2014

@author: Neerav Basant
"""

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import math

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

#Running linear and rbf model
C = 10
gamma = math.pow(2, -5)

lin = svm.SVC(kernel = 'linear', C = C)
rbf = svm.SVC(kernel = 'rbf', gamma = gamma, C = C)

lin.fit(X_train_scaled, y_train)
rbf.fit(X_train_scaled, y_train)

linpred = lin.predict(X_test_scaled)
rbfpred = rbf.predict(X_test_scaled)

print accuracy_score(y_test, linpred)
print accuracy_score(y_test, rbfpred)