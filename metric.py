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

import dunn_sklearn

from sklearn.metrics.cluster import unsupervised

def silhouette_samples(X,labels,metric='euclidean',**kwds):
    return unsupervised.silhouette_samples(X,labels,metric,**kwds)

def silhouette_score(X,labels,metric='euclidean',sample_size=None,random_state=None,**kwds):
    return unsupervised.silhouette_score(X,labels,metric,sample_size,random_state,**kwds)

def silhouette_plot(X,labels,metric='euclidean',fig_size = None,type = None,cluster = None,**kwds):
    
    silhouette_samples = unsupervised.silhouette_samples(X,labels,metric,**kwds)
    
    df = pd.DataFrame(silhouette_samples)
    df['cluster'] = labels

    if type == 'None':

	    cluster_means = df.groupby('cluster').mean()
	    dit = dict(zip(cluster_means.index,cluster_means[0]))
	    df2 = pd.DataFrame(list(dit.items()))
	    df2.columns = ['Cluster','silhouette_mean']
	    df2 = df2.sort_values(['silhouette_mean'],ascending=False).reset_index(drop=True)
	    
	    if fig_size == None:
	        if len(df2) > 64:
	            fig = plt.figure(figsize=(len(df2)/8,len(df2)/4))
	        else:
	            fig = plt.figure(figsize=(8,6))
	    else:
	        fig = plt.figure(figsize = fig_size)
	        
	    ax = sns.barplot(df2['silhouette_mean'],y = cluster_means.index,orient='h',)
	    ax.set_yticklabels(df2['Cluster'])
	    plt.show()

	elif type == 'cluster':

		df['id_cliente'] = labels['id_cliente']
		cluster = df[df['cluster'] == cluster]
		cluster.columns = ['silhouette','Cluster','id_cliente']
		cluster = cluster.sort_values(['silhouette'],ascending=False).reset_index(drop=True)

		if fig_size == None:
	        if len(cluster) > 64:
	            fig = plt.figure(figsize=(len(cluster)/8,len(cluster)/4))
	        else:
	            fig = plt.figure(figsize=(8,6))
	    else:
	        fig = plt.figure(figsize = fig_size)
	        
	    ax = sns.barplot(cluster['silhouette'],y = cluster.index.values,orient='h',)
	    ax.set_yticklabels(cluster['Clientes'])
	    plt.show()


#Implementation of Dunn index

from sklearn.metrics.pairwise import euclidean_distances

def normalize(X):
	return dunn_sklearn.normalize_to_smallest_integers(X)

def cluster_diameter(X, distances):
	return dunn_sklearn.diameter(X, euclidean_distances(X.drop('labels')))

def min_cluster_distances(X, distances):
	return dunn_sklearn.min_cluster_distances(X, euclidean_distances(X.drop('labels')))

def dunn_index(X, distances):
	return dunn_sklearn.dunn(labels, euclidean_distances(X.drop('labels')))
>>>>>>> 1141e0b41e2fc263a13902ad207eeb4ed8a7adc6
