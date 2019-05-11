# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:34:45 2019

@author: OmPrakash
#Non-continous Regreesor model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Negotiate_position.csv")
X= dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
DT= DecisionTreeRegressor()
DT.fit(X,Y)

Y_pred = DT.predict(7.5)

# plot Decision Tree in high resolution other wise error  as it is a non-continous model
#it take vg of all the points to predict y in the region
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,  color = "blue")
plt.plot(X_grid,DT.predict(X_grid), color = "red")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Salary Vs Position using Decision Tree")
plt.show()

