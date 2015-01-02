# -*- coding: utf-8 -*-
"""
Created on Thu Jan 01 03:09:51 2015

@author: Neerav Basant
"""

"""
    Question 2
"""

from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignment 2/hw2files/congress.csv")

df.drop('name', axis=1, inplace=True)
df.drop('V1002', axis=1, inplace=True)
df.drop('V1003', axis=1, inplace=True)

"""
    2(a)
"""
mean = df.mean(axis=0)
df = df - mean

"""
    2(b)
"""
k = [1,2,5,10,20,50,100,200]
for i in k:     
    pcaobject = TruncatedSVD(n_components=np.int(i))
    Xt = pcaobject.fit_transform(df)
    
    fig = plt.figure()
    explained_variances = np.var(Xt, axis=0) / np.var(df, axis=0).sum()
    sing_vals = np.arange(i) + 1
    cum_sum = np.cumsum(np.var(Xt, axis=0)) / np.var(df, axis=0).sum()
    
    plt.plot(sing_vals, explained_variances, 'ro-', linewidth=2)
    plt.plot(sing_vals, cum_sum, 'bo-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel("Principal Components")
    plt.ylabel("Proportion of Variance")
    leg = plt.legend(["Explained Variance", "Cumulative Sum of Explained Variance"], loc='best', borderpad=0.3, 
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    
"""
    2(c)
"""
df_copy = pd.read_csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignment 2/hw2files/congress.csv")
pcaobject = TruncatedSVD(n_components = np.int(2))
Xt = pcaobject.fit_transform(df)
newDf = pd.DataFrame(Xt)
newDf['party'] = df_copy['V1002']

"""
   Plotting with the two dimensions for different party members in different colors
"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(newDf[1].ix[newDf['party']== 'R'], newDf[0].ix[newDf['party']== 'R'], color='red')
plt.scatter(newDf[1].ix[newDf['party']== 'D'], newDf[0].ix[newDf['party']== 'D'])
ax.set_ylabel('T0')
ax.set_xlabel('T1')
plt.title("Scatter Plot of the top 2 transformed dimensions")
plt.show()


"""
    Question 3
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import math

boston = load_boston()

values = np.concatenate((boston.data, np.reshape(boston.target, (506,1))), axis=1)
df = pd.DataFrame.from_records(values, columns = boston.feature_names)

"""
    Part (a)
"""

lstat_plot = plt.boxplot(df['LSTAT'])
plt.title('Box Plot - LSTAT')
plt.show()
medv_plot = plt.boxplot(df['MEDV'])
plt.title('Box Plot - MEDV')
plt.show()

[item.get_ydata() for item in lstat_plot['whiskers']]
[item.get_ydata() for item in medv_plot['whiskers']]

"""
    Part (b)
"""

df['CATEGORY'] = pd.Series(np.random.randn(len(df['LSTAT'])))

for i in range(0, len(df['LSTAT'])):
    if df['LSTAT'][i] >= 1.73 and df['LSTAT'][i] <= 30.81 and df['MEDV'][i] >= 5.6 and df['MEDV'][i] <= 36.5:
        df['CATEGORY'][i] = 0
    else:
        df['CATEGORY'][i] = 1

lstat_with_outlier = df[['LSTAT']]
medv_with_outlier = df[['MEDV']]

Boston_no_outlier = df[df['CATEGORY']==0]
lstat_no_outlier = Boston_no_outlier[['LSTAT']]
medv_no_outlier = Boston_no_outlier[['MEDV']]

clf = LinearRegression()
lm_fit = clf.fit(lstat_with_outlier, medv_with_outlier)
lm_fit_no_outlier = clf.fit(lstat_no_outlier, medv_no_outlier)

plt.scatter([df['LSTAT'][x] for x in range(0, len(df['LSTAT'])) if df['CATEGORY'][x] == 0], [df['MEDV'][y] for y in range(0, len(df['MEDV'])) if df['CATEGORY'][y] == 0], c= 'blue', alpha = 0.2)
plt.scatter([df['LSTAT'][x] for x in range(0, len(df['LSTAT'])) if df['CATEGORY'][x] == 1], [df['MEDV'][y] for y in range(0, len(df['MEDV'])) if df['CATEGORY'][y] == 1], c= 'red', alpha = 0.5)
plt.plot(lstat_with_outlier, lm_fit.predict(lstat_with_outlier), color='black', linewidth=1.5)
plt.plot(lstat_no_outlier, lm_fit_no_outlier.predict(lstat_no_outlier), color='green', linewidth=1.5)
plt.title('Scatter Plot')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

"""
    Part (c)
"""

df['LMDEV'] = pd.Series(np.random.randn(len(df['MEDV'])))

for i in range(0, len(df['MEDV'])):
    df['LMDEV'][i] = math.log(df['MEDV'][i])

df['CHAS'] = df['CHAS'].astype(bool)
Bostrain = df[:-156]
Bostest = df[-156:]

X = Bostrain[['LSTAT', 'RM', 'CRIM', 'ZN', 'CHAS']]
Y = Bostrain['LMDEV']

X_Test = Bostest[['LSTAT', 'RM', 'CRIM', 'ZN', 'CHAS']]
Y_Test = Bostest['LMDEV']

"""
    c(i)
"""
clf = LinearRegression()
lm_fit_multiple = clf.fit(X, Y)

Bostrain_pred = lm_fit_multiple.predict(X)

square_error = 0.0
for i in range(len(Bostrain_pred)):
    square_error += float(math.pow(math.exp(Bostrain_pred[i:i+1]) - math.exp(Y[i:i+1]), 2))

mean_square_error = square_error/len(Bostrain_pred)
mean_square_error

Bostest_pred = clf.predict(X_Test)

square_error_test = 0
for i in range(156):
    square_error_test += math.pow(math.exp(Bostest_pred[i]) - math.exp(Y_Test[i+350]), 2)

mean_square_error_test = square_error_test/len(Bostest_pred)
mean_square_error_test

"""
    c(ii)
"""
print clf.coef_

"""
    c(iii)
"""
residuals=[0.0]*len(X_Test)

for i in range(len(X_Test)):
    residuals[i]= math.exp(Bostest_pred[i]) - math.exp(Y_Test[i+350])
 
Z=[0.0]*7
for i in xrange(7):
    Z[i]=1.5+(i*0.5)
  
plt.scatter(Bostest_pred,residuals) 
plt.plot(Z,Z)
plt.title('Residual Plot')
plt.xlabel('Predicted vales')
plt.ylabel('Residues')
plt.show()


