import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import sys
sys.path.insert(0,r'C:\Projetos\Machine_Learning\Cluster_Metrics\Clusters-Metrics-Library')

import metric

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(data[0])

kmeans.cluster_centers_

kmeans.labels_


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()

metric.silhouette_plot(data[0], kmeans.labels_)