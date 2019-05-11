# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:17:55 2019

@author: OmPrakash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Negotiate_position.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X,Y)

Y_Pred = RF.predict(6.5)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y, color = 'blue')
plt.plot(X_grid,RF.predict(X_grid),color = "green")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.title("Salary VS position")
plt.show()

