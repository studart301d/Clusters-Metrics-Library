# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:19:57 2018

@author: gabri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.cluster import unsupervised

def silhouette_samples(X,labels,metric='euclidean',**kwds):
    return unsupervised.silhouette_samples(X,labels,metric,**kwds)

def silhouette_score(X,labels,metric='euclidean',sample_size=None,random_state=None,**kwds):
    return unsupervised.silhouette_score(X,labels,metric,sample_size,random_state,**kwds)

def silhouette_plot(X,labels,metric='euclidean',**kwds):
    silhouette_samples = unsupervised.silhouette_samples(X,labels,metric,**kwds)
    df = pd.DataFrame(silhouette_samples)
    df['cluster'] = labels
    cluster_means = df.groupby('cluster').mean()
    dit = dict(zip(cluster_means.index,cluster_means[0]))
    df2 = pd.DataFrame(list(dit.items()))
    df2.columns = ['Cluster','silhouette_mean']
    df2 = df2.sort_values(['silhouette_mean'],ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(12,24))
    ax = sns.barplot(df2['silhouette_mean'],y = cluster_means.index,orient='h',)
    ax.set_yticklabels(df2['Cluster'])
    plt.show()
