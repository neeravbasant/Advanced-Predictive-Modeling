# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:06:44 2014

@author: Neerav Basant
"""

from sklearn import svm
from sklearn.linear_model import LinearRegression as regr
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt

train = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 5/HW5_files/data.train")
test = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 5/HW5_files/data.test")

#Standardizing the datasets
X_train, X_test = train.ix[:, 1:].as_matrix(), test.ix[:, 1:].as_matrix()
y_train, y_test = train.ix[:, 0].as_matrix(), test.ix[:, 0].as_matrix()
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3(a)
clf = svm.SVR()
clf.fit(X_train_scaled, y_train)

pred_svm_train = clf.predict(X_train_scaled)
rmse_svm_train = sqrt(mean_squared_error(y_train, pred_svm_train))

pred_svm_test = clf.predict(X_test_scaled)
rmse_svm_test = sqrt(mean_squared_error(y_test, pred_svm_test))

print "RMSE - Train (SVM) = ", rmse_svm_train
print "RMSE - Test (SVM) = ", rmse_svm_test

# 3(b)
lr = regr()
lr.fit(X_train_scaled, y_train)

pred_lr_train = lr.predict(X_train_scaled)
rmse_lr_train = sqrt(mean_squared_error(y_train, pred_lr_train))

pred_lr_test = lr.predict(X_test_scaled)
rmse_lr_test = sqrt(mean_squared_error(y_test, pred_lr_test))

print "RMSE - Train (LR) = ", rmse_lr_train
print "RMSE - Test (LR) = ", rmse_lr_test