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

def silhouette_plot():
    silhouette_samples = unsupervised.silhouette_samples(X,labels,metric,**kwds)
    sns.countplot()