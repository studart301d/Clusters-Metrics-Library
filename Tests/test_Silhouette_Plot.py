import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
from pathlib import Path
import os


#working with relative path's
#get the currently worked direcotry
directory=Path(os.getcwd()).parents[0]
sys.path.insert(0,str(directory))



import metric
from sklearn.metrics.pairwise import euclidean_distances


data = make_blobs(n_samples=500, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(data[0])

kmeans.cluster_centers_

kmeans.labels_


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()

metric.cluster_avaliation(data[0], kmeans.labels_,euclidean_distances(data[0]),20,4)
#metric.silhouette_plot(data[0], kmeans.labels)