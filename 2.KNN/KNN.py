# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 08:59:58 2019

@author: OmPrakash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Classified Data')
dataset.info()
dataset.describe()

#feature scaling required
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
SC.fit(dataset.drop('TARGET CLASS',axis = 1))
SCALED_VALUES = SC.transform(dataset.drop('TARGET CLASS',axis = 1))

scaled_dataframe = pd.DataFrame(SCALED_VALUES,columns = dataset.columns[:-1])


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(scaled_dataframe,dataset['TARGET CLASS'],
                                                 test_size =0.3,random_state = 30)

# choose k using elbow  method
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1,30):
    KNN = KNeighborsClassifier(n_neighbors = i)
    KNN.fit(X_train,Y_train)
    Y_pred = KNN.predict(X_test)
    error_rate.append(np.mean(Y_pred != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,30),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

KNN =KNeighborsClassifier(n_neighbors = 7)
KNN.fit(X_train,Y_train)
Y_pred = KNN.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,Y_pred))
print('\n')
print (confusion_matrix(Y_test,Y_pred))

