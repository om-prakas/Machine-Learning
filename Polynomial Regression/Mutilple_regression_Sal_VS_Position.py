# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:01:26 2019

@author: OmPrakash
# According to   position(year of experience matters for position) find  salary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Salary_Negotiate_position.csv')
#writing 1 instead of 1:2 create vector.To make matrix we write 1:2
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

"""
#small no of set given. We want predict accurate prediction.So no split 

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

 
plt.scatter(X,Y,color = 'green')
plt.plot(X,LR.predict(X), color = 'red')
plt.xlabel("Position level ")
plt.ylabel("Salary")
plt.title("Position Vs Salary Graph in Linear Regression")
plt.show()

# predict the New value using LR
LR.predict(6.5)
"""

from sklearn.preprocessing import PolynomialFeatures,LinearRegression
#PR object automatically create b0 (constant term) and add 1 column in the metrix 
PR = PolynomialFeatures(degree = 4)
#PR transform X metrics by adding additional polynomial term(depends upon degree) into X
#We are trnasforming X metrics into X_poly. So use fit_transform method
X_poly = PR.fit_transform(X)
#now create a linear model obj and fit the new x_metrics
LR2 = LinearRegression()
LR2.fit(X_poly,Y)

"""
# Basic plot 
plt.scatter(X,Y, color = 'blue')
#Instead of X_poly for generalisation purpose use PR.fit_transform(X) (= X_poly)
plt.plot(X,LR2.predict(PR.fit_transform(X)), color = 'red')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.title("Position Vs Salary Graph in Multiple REgression")
plt.show()
"""

# for high resolution and smother curve 
X_grid = np.arange(min(X), max(X), 0.1)  # create a vector
X_grid = X_grid.reshape((len(X_grid), 1)) # create a matrix using reshape method
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, LR2.predict(PR.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# predict new salary using Multi-linear regression
LR2.predict(PR.fit_transform(3.5))



