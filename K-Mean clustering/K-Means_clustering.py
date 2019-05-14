# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Shop_Customers.csv")

X= dataset.iloc[:,[3,4]].values

#use elbow method to find optimal number of cluster
from sklearn.cluster import KMeans
list1= []
for i in range(1,11):
    # to avoid random initialisation trap we use k-mean++ (default value)
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 50)
    kmeans.fit(X)
    #to find the distance between points
    list1.append(kmeans.inertia_)
#enter xaxis -and y axis value 
plt.plot(range(1,11), list1)
plt.title("Elbow method")
plt.xlabel("No of cluster")
plt.ylabel("Within cluster sum of square(WCSS)")
plt.show()

#apply k-mean (find k in above steps)
kmeans = KMeans(n_clusters = 5,random_state = 50)
y_kmeans = kmeans.fit_predict(X)

#plot the clster
#in x dataset 2 value present.
#1st parameter for X coordinate - 1st 0 -value which lies in 0 cluster and 2nd 0 is  1st column
# do same for 5 cluster 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'purple', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()