# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:28:00 2019

@author: OmPrakash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.3, random_state = 30)

# linear regression library take care of feature scaling no need to do manually

# Fitting SlR in testdata
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(X_train,Y_train)

#predicting the value 
Y_pred = LR.predict(X_test)

#visualising the training set result
plt.scatter(X_train,Y_train, color = 'green')
#prediccting the value of training data of X 
plt.plot(X_train,LR.predict(X_train), color = 'red')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience in Training set')
plt.show()

#Visualizing the testing set  result
plt.scatter(X_test,Y_test, color = 'blue')
#prediction line should not b changed.
plt.plot(X_train,LR.predict(X_train), color = 'orange')
plt.xlabel('experience')
plt.ylabel('salary')
plt.title('Salary vs Experience in Test set')
plt.show()



