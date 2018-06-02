# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:21:52 2018

@author: trust-tyler
"""

#%reset -f

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
df

x = df.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("Optimal number of clusters with Elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


#Applying k-means with 5 clusters

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(x)


#Visualizing clusters

plt.scatter(x[ykmeans == 0, 0], x[ykmeans == 0 , 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(x[ykmeans == 1, 0], x[ykmeans == 1 , 1], s = 100, c = 'green', label = 'cluster_2')
plt.scatter(x[ykmeans == 2, 0], x[ykmeans == 2 , 1], s = 100, c = 'magenta', label = 'cluster_3')
plt.scatter(x[ykmeans == 3, 0], x[ykmeans == 3 , 1], s = 100, c = 'blue', label = 'cluster_4')
plt.scatter(x[ykmeans == 4, 0], x[ykmeans == 4 , 1], s = 100, c = 'yellow', label = 'cluster_5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'centroids')
plt.title("Cluster of clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()



