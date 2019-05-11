# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dataset Import 
dataset =pd.read_csv('Data.csv')

# create independent variable metrics
X = dataset.iloc[:,:-1].values       # [:,:] - Rows, Columns
#create dependent variable metrics
Y = dataset.iloc[:,3].values

#handling missing Data
from sklearn.preprocessing import Imputer
# NaN,mean,0 (represent column) all are default value
imputer = Imputer(missing_values ='NaN',strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])  
X[:,1:3] = imputer.transform(X[:,1:3])

#handle Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =  LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# categorical value change to numeric but value will 0,1,2 that may lead to confusion for rank
# Use dummy variable  use 001,100,001 (not 1,2,3)
onehotencoder = OneHotEncoder(categorical_features = [0])  # 0 column u encode
X= onehotencoder.fit_transform(X).toarray()     # priviously which column specified.No need here

#In dependent variable 2 category is there .So just do lable encoder not one hot encoder 
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

# split the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 50)

#feature scaling(eucledian dsiance minimize)
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)













