# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:32:45 2019

@author: OmPrakash
Multiple linear regression using backward elimination Method 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Companies_Data.csv')
X = dataset.iloc[:,:-1].values
Y =  dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid dummy variable trap(don't take 1st column)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.8,random_state = 50)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)

#optimal solution using back propagation 
import statsmodels.formula.api as sm
# add the b0 term as this library not include the constant term bo
#Convert array of 1 into a integer type to avaoid error axis =1 means add a column
#X = np.append(arr = X , values = np.ones((50, 1)).astype(int), axis = 1)
#To add the column of matrix in beginning chane the position 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# Manually check .Highest p value eliminate 
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
ols = sm.OLS(endog = Y, exog = X_optimal).fit()
ols.summary()

"""
#Automatic do manual back propagation using p- value 
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        ols = sm.OLS(Y, x).fit()
        maxVar = max(ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (ols.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    ols.summary()
    return x
 
SL = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4,5]]
X_Modeled = backwardElimination(X_optimal, SL)
"""


