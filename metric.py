# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:19:57 2018
@author: gabri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

#working with relative path's
#get the currently worked direcotry
directory=Path(os.getcwd()).parents[0]
sys.path.insert(0,str(directory))

#import dunn_sklearn



#Implementation of silhouette


from sklearn.metrics.cluster import unsupervised

def silhouette_samples(X,labels,metric='euclidean',**kwds):
	return unsupervised.silhouette_samples(X,labels,metric,**kwds)

def silhouette_score(X,labels,metric='euclidean',sample_size=None,random_state=None,**kwds):
	return unsupervised.silhouette_score(X,labels,metric,sample_size,random_state,**kwds)

def silhouette_plot(X,labels,metric='euclidean',fig_size = None,type = None,cluster = None,y=None,index = False,**kwds):
    
	silhouette_samples = unsupervised.silhouette_samples(X,labels,metric,**kwds)

	df = pd.DataFrame(silhouette_samples)
	df['cluster'] = labels

	if type == None:
		cluster_means = df.groupby('cluster').mean()
		dit = dict(zip(cluster_means.index,cluster_means[0]))
		df2 = pd.DataFrame(list(dit.items()))
		df2.columns = ['Cluster','silhouette_mean']
		
		
		if fig_size == None:
			if len(df2) > 64:
				fig = plt.figure(figsize=(len(df2)/8,len(df2)/4))
			else:
				fig = plt.figure(figsize=(8,6))
		else:
			fig = plt.figure(figsize = fig_size)


		df2 = df2.sort_values(['silhouette_mean'],ascending=False).reset_index(drop=True)
		ax = sns.barplot(df2['silhouette_mean'],y = df2.index,orient='h')
		ax.set_yticklabels(df2['Cluster'])
		plt.ylabel('Cluster')
		plt.show()	

	elif type == 'cluster':

		if index == True:
			df['y'] = y

		cluster = df[df['cluster'] == cluster]
		cluster.columns = ['silhouette','Cluster','y']
		cluster = cluster.sort_values(['silhouette'],ascending=False).reset_index(drop=True)
		
		if fig_size == None:
			if len(cluster) > 64:
				fig = plt.figure(figsize=(len(cluster)/8,len(cluster)/4))
			else:
				fig = plt.figure(figsize=(8,6))
		else:
			fig = plt.figure(figsize = fig_size)

		ax = sns.barplot(cluster['silhouette'],y = cluster.index,orient='h')

		if index == True:
			ax.set_yticklabels(cluster['y'])

		plt.ylabel('Index')
		plt.show()

def elbow(data, max_number_of_clusters,step = 1):

    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    distortions = []
    K = np.arange(1, max_number_of_clusters+1,step)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(data)
        kmeans.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(data))

    # Plot the elbow
    plt.plot(K, distortions, 'x-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

#Dunn Index:



def dunn(labels, distances):
    """
    Dunn index for cluster validation (the bigger, the better)
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, given by the distances between its
    two closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.
    
    The bigger the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart.

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    
	O Parametro distances pode ser utiliado sklearn.metrics.pairwise
		por exemplo from sklearn.metrics.pairwise import euclidean_distances
    """


    labels = normalize_to_smallest_integers(labels)

    unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
    max_diameter = max(diameter(labels, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def normalize_to_smallest_integers(labels):
    """Normalizes a list of integers so that each number is reduced to the minimum possible integer, maintaining the order of elements.

    :param labels: the list to be normalized
    :returns: a numpy.array with the values normalized as the minimum integers between 0 and the maximum possible value.
    """

    max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
    sorted_labels = np.sort(np.unique(labels))
    unique_labels = range(max_v)
    new_c = np.zeros(len(labels), dtype=np.int32)

    for i, clust in enumerate(sorted_labels):
        new_c[labels == clust] = unique_labels[i]

    return new_c


def min_cluster_distances(labels, distances):
    """Calculates the distances between the two nearest points of each cluster.

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    """
    labels = normalize_to_smallest_integers(labels)
    n_unique_labels = len(np.unique(labels))

    min_distances = np.zeros((n_unique_labels, n_unique_labels))
    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
    return min_distances

    
def diameter(labels, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :returns:
    """
    labels = normalize_to_smallest_integers(labels)
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = distances[i, ii]
    return diameters
        

def cluster_avaliation(X,labels,distances,max_number_of_clusters = None,step = 1,fig_size = None,type = None,cluster = None,y=None,index = False):
	print("Result of silhouette: "+ str(silhouette_score(distances,labels, metric= 'precomputed')))
	print("Result of dunn index: ",str(dunn(labels,distances)),"\n")
	if max_number_of_clusters != None:
		elbow(X,max_number_of_clusters,step)
	silhouette_plot(distances,labels,metric= 'precomputed',fig_size = None,type = None,cluster = None,y=None,index = False)