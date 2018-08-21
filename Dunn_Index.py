
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

get_ipython().magic('matplotlib inline')


# In[13]:


from scipy.spatial.distance import euclidean


# In[14]:


def cluster_diameter(data, cluster_number):
    x = list()
    cluster = data.loc[data['labels'] == cluster_number]
    for i in range(len(cluster)):
        x.append(math.sqrt(sum((cluster.drop('labels', axis=1).iloc[i,:])**2)))
    return max(x)


# In[15]:


def max_cluster_diameter(data, n_clusters):
    diameters = list()
    for i in range(n_clusters):
        diameters.append(cluster_diameter(data, i))
    return max(diameters)


# In[16]:


#calculate distances between cluster midpoints
def min_dist_interpoints(points, n_points):
    distances = list()
    for i in range(n_points):
        if i < n_points-1:
            distances.append(euclidean(points[i], points[i+1]))
        else:
            distances.append(euclidean(points[0], points[i]))
    return min(distances)


# In[21]:


#ideia: calcular os pontos médios de cada cluster e, então, a distância entre eles
"""def min_intercluster_distance(data, n_clusters):
    midpoints = list()
    clusters = list(np.arange(1, n_clusters+1))
    for i in range(n_clusters):
        cluster = data.loc[data['labels'] == clusters[i]]
        for k in range(len(cluster)):
            cluster_features = list()
            cluster_features.append(cluster.drop('labels', axis=1).iloc[k,:])
        midpoint_features = list()
        for j in range(len(cluster.drop('labels', axis=1).columns)):
            midpoint_features.append(cluster.drop('labels', axis=1).iloc[:,j].mean())
        #midpoint_features = pd.Series(midpoint_features)
        midpoints.append(midpoint_features)
    return min_dist_interpoints(midpoints, n_clusters)"""


# In[22]:


def min_intercluster_distance(n_clusters, centers):
    return min_dist_interpoints(centers, n_clusters)


# ### Calculate general Dunn index

# In[1]:


def dunn_index(data, n_clusters, centers):
    dunn_index = min_intercluster_distance(n_clusters, centers)/max_cluster_diameter(data, n_clusters)
    return dunn_index

