# -*- coding: utf-8 -*-
"""
Created on Sun Nov 09 10:31:20 2014

@author: Neerav Basant
"""

from sklearn import svm
import pandas as pd
from scipy import stats

# 5(a)
df = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 4/hw4files/oneclassdata.csv", header=None, names=['F1', 'F2', 'Class'])

outliers_fraction = 0.1

modl = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1)

modl.fit(df.ix[:, :2])

# 5(b)
ypred =  modl.decision_function(df.ix[:, :2]).ravel()

# 5(c)
threshold = stats.scoreatpercentile(ypred, 100 * outliers_fraction)
ypred = ypred > threshold

n_errors = (ypred != df.ix[:, 2]).sum()
print n_errors