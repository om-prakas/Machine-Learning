# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:47:56 2019

@author: OmPrakash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Negotiate_position.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values  #2: to make a vector into matrix we use :

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Feature Scaling required in SVR model
from sklearn.preprocessing import StandardScaler
X_Scale = StandardScaler()
Y_Scale = StandardScaler()
X = X_Scale.fit_transform(X)
Y = Y_Scale.fit_transform(Y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X, Y)

# Predicting a new result
Y_pred = svr.predict(6.5)
Y_pred = Y_Scale.inverse_transform(Y_pred)


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'blue')
plt.plot(X_grid, svr.predict(X_grid), color = 'red')
plt.title('Predict Salary using SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
