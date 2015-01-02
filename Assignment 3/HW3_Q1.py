# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 23:43:41 2014

@author: Neerav Basant
"""

import pandas as pd
from sklearn import linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing

dfAuto = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 3/hw3files/autos.csv")

dfAutoTrain = dfAuto[dfAuto["train"] == True]
dfAutoTest = dfAuto[dfAuto["train"] == False]

"""
    Question 1 - a
"""

dfAutoTrain = dfAutoTrain.reset_index()
dfAutoTest = dfAutoTest.reset_index()

dfFolds = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/\
Assignments/Assignment 3/hw3files/auto.folds", header = None)

dfAutoTrain["fold"] = dfFolds

def MSE(predicted, actual):
    SSE = 0.0
    for i in range(len(actual)):
        SSE += (predicted[i]-actual[i])**2
    return (SSE/len(actual))
    
#Ridge regression

"""
    I created training and test dataset using folds dataset and ran ridge regression on them.
    I calculated mean square error for each value of regularization parameter.
    I got the minimum mean square error for regularization parameter = 5.
"""

ridge_grid = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]

ridge_mse = []
ridge_coef = []
for lamda in ridge_grid:
    clf = lm.Ridge(alpha = lamda)
    ridge_error = 0
    for i in range(0,5):
        train, test = dfAutoTrain[dfAutoTrain["fold"] != i], dfAutoTrain[dfAutoTrain["fold"] == i]
        X_train, X_test = train[[2,3,4,5,6,7]].as_matrix(), test[[2,3,4,5,6,7]].as_matrix()
        y_train, y_test = train[[1]].as_matrix(), test[[1]].as_matrix()
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ridge_mod = clf.fit(X_train_scaled, y_train)
        ridge_pred = clf.predict(X_test_scaled)
        ridge_error += MSE(ridge_pred, y_test)
    ridge_coef.extend(ridge_mod.coef_)
    ridge_mse.append(ridge_error/5)
print ""
print "Mean Square Error for Ridge Regression (Cross Validation) for various regularization parameters: ", ridge_mse


"""
    Plotting Mean Square Error for cross validation using ridge regression
"""

plt.plot(ridge_grid, ridge_mse, linewidth=2) 
plt.title('Ridge Regression - MSE Plot')
plt.xlabel("Regularization Parameter")
plt.ylabel("Mean Square Error")
leg = plt.legend(['Mean Square Error'], loc='best', borderpad=0.3,
                      shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                      markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

print "Best Regularization Parameter for Ridge regression is 5"
print ""
#Lasso regression

"""
    I created training and test dataset using folds dataset and ran lasso regression on them.
    I calculated mean square error for each value of the given regularization parameter.
    I got the minimum mean square error for regularization parameter = 0.1.
    Since 0.1 was the last entry in the list of regularization parameter, I also added 0.5 and 1 to get a better plot.
"""

lasso_grid = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]

lasso_mse = []
lasso_coef = []
for lamda in lasso_grid:
    clf1 = lm.Lasso(alpha = lamda)
    lasso_error = 0
    for i in range(0,5):
        train, test = dfAutoTrain[dfAutoTrain["fold"] != i], dfAutoTrain[dfAutoTrain["fold"] == i]
        X_train, X_test = train[[2,3,4,5,6,7]].as_matrix(), test[[2,3,4,5,6,7]].as_matrix()
        y_train, y_test = train[[1]].as_matrix(), test[[1]].as_matrix()
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lasso_mod = clf1.fit(X_train_scaled, y_train)
        lasso_pred = clf1.predict(X_test_scaled)
        lasso_error += MSE(lasso_pred, y_test)
    lasso_coef.append(lasso_mod.coef_)
    lasso_mse.append(lasso_error/5)
print "Mean Square Error for Lasso Regression (Cross Validation) for various regularization parameters: ", lasso_mse

"""
    Plotting Mean Square Error for cross validation using Lasso Regression
"""
plt.plot(lasso_grid, lasso_mse, linewidth=2) 
plt.title('Lasso Regression - MSE Plot')
plt.xlabel("Regularization Parameter")
plt.ylabel("Mean Square Error")
leg = plt.legend(['Mean Square Error'], loc='best', borderpad=0.3,
                      shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                      markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

print "Best Regularization Parameter for Lasso regression is 0.1"

"""
    Question 1 - b
"""

dfAutoTrain_1 = dfAuto[dfAuto["train"] == True]
dfAutoTest_1 = dfAuto[dfAuto["train"] == False]

X_dfAutoTrain = dfAutoTrain_1[[1,2,3,4,5,6]].as_matrix()
y_dfAutoTrain = dfAutoTrain_1[[0]].as_matrix()
X_dfAutoTest = dfAutoTest_1[[1,2,3,4,5,6]].as_matrix()
y_dfAutoTest = dfAutoTest_1[[0]].as_matrix()
scaler = preprocessing.StandardScaler().fit(X_dfAutoTrain)
X_dfAutoTrain_scaled = scaler.transform(X_dfAutoTrain)
X_dfAutoTest_scaled = scaler.transform(X_dfAutoTest)

#Ridge Regression

"""
    Running the ridge regression on complete training data for all the value of regularization parrameter to get the coefficient plot
"""

ridge_mse = []
ridge_coef = []
for lamda in ridge_grid:
    clf = lm.Ridge(alpha = lamda)
    ridge_mod = clf.fit(X_dfAutoTrain_scaled, y_dfAutoTrain)
    ridge_pred = clf.predict(X_dfAutoTest_scaled)
    ridge_error = MSE(ridge_pred, y_dfAutoTest)
    ridge_coef.extend(ridge_mod.coef_)
    ridge_mse.append(ridge_error)

dfRidgeCoef = pd.DataFrame(ridge_coef, columns = train[[2,3,4,5,6,7]].columns)

"""
    Coefficient plot for ridge regression
"""

plt.plot(ridge_grid, dfRidgeCoef, linewidth=2) 
plt.title('Ridge Regression - Coefficient Plot')
plt.xlabel("Regularization Parameter")
plt.ylabel("Coefficients of Parameters")
leg = plt.legend(['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'origin'], loc='best', borderpad=0.3,
                      shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                      markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

#Lasso Regression

"""
    Running the lasso regression on complete training data for all the value of regularization parrameter to get the coefficient plot
"""

lasso_mse = []
lasso_coef = []
for lamda in lasso_grid:
    clf = lm.Lasso(alpha = lamda)
    lasso_mod = clf.fit(X_dfAutoTrain_scaled, y_dfAutoTrain)
    lasso_pred = clf.predict(X_dfAutoTest_scaled)
    lasso_error = MSE(lasso_pred, y_dfAutoTest)
    lasso_coef.append(lasso_mod.coef_)
    lasso_mse.append(lasso_error)
    
dfLassoCoef = pd.DataFrame(lasso_coef, columns = train[[2,3,4,5,6,7]].columns)

"""
    Coefficient plot for lasso regression
"""
plt.plot(lasso_grid, dfLassoCoef, linewidth=2)
plt.title('Lasso Regression - Coefficient Plot')
plt.xlabel("Regularization Parameter")
plt.ylabel("Coefficients of Parameters")
leg = plt.legend(['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'origin'], loc='best', borderpad=0.3,
                      shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                      markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

"""
    Question 1 - c
"""

# Least Square Regression
lin = lm.LinearRegression()
linear_mod = lin.fit(X_dfAutoTrain_scaled, y_dfAutoTrain)
linear_pred = linear_mod.predict(X_dfAutoTest_scaled)
linear_error = MSE(linear_pred, y_dfAutoTest)
print "Mean Square Error for linear regression using all the parameters is ", linear_error

# Ridge Regression
rd = lm.Ridge(alpha = 5)
rd_mod = rd.fit(X_dfAutoTrain_scaled, y_dfAutoTrain)
rd_pred = rd_mod.predict(X_dfAutoTest_scaled)
rd_error = MSE(rd_pred, y_dfAutoTest)
print "Mean Square Error for ridge regression (regularization parameter = 5) is ", rd_error

# Lasso Regression
ls = lm.Lasso(alpha = 0.1)
ls_mod = ls.fit(X_dfAutoTrain_scaled, y_dfAutoTrain)
ls_pred = ls_mod.predict(X_dfAutoTest_scaled)
ls_pred = np.reshape(ls_pred, (-1, 1))
ls_error = MSE(ls_pred, y_dfAutoTest)
print "Mean Square Error for lasso regression (regularization parameter = 0.1) is ", ls_error

"""
    Question 1 - d
"""

Xnew_dfAutoTrain = dfAutoTrain_1[[1,3,4,5,6]].as_matrix()
Xnew_dfAutoTest = dfAutoTest_1[[1,3,4,5,6]].as_matrix()
scaler = preprocessing.StandardScaler().fit(Xnew_dfAutoTrain)
Xnew_dfAutoTrain_scaled = scaler.transform(Xnew_dfAutoTrain)
Xnew_dfAutoTest_scaled = scaler.transform(Xnew_dfAutoTest)

# Least Square Regression
lin1 = lm.LinearRegression()
linear_mod1 = lin1.fit(Xnew_dfAutoTrain_scaled, y_dfAutoTrain)
linear_pred1 = linear_mod1.predict(Xnew_dfAutoTest_scaled)
linear_error1 = MSE(linear_pred1, y_dfAutoTest)
print "Mean Square Error for linear regression using relevant parameters based on Lasso Regression is ", linear_error1