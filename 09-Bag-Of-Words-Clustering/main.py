#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:59:02 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

df = pd.read_csv("nips-87-92.csv")
del df["Unnamed: 0"]
df.index = df.doc_id
del df["doc_id"]

kmeans_model_dict = {}
sse_dict = {}
silhouette_score_dict = {}
cluster_size_dict = [10,50,100,300,500,600,650]
cluster_size_dict = [100]

for k in sorted(cluster_size_dict): 
    print "Cluster :",k
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(df)
    kmeans_model_dict[k] = kmeans_model
    sse_dict[k] = sum(np.min(cdist(df, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
    silhouette_score_dict[k] = silhouette_score(df, kmeans_model.labels_)

plt.figure()
plt.plot(*zip(*sorted(sse_dict.items())))
plt.xlabel("Cluster size")
plt.ylabel("sum of squared error (SSE)")
plt.title("Elbow method to determine no of cluster(k)")
plt.savefig('sse.png')

plt.figure()
plt.plot(*zip(*sorted(silhouette_score_dict.items())))
plt.xlabel("Cluster size")
plt.ylabel("Silhouette Score")
plt.title("Elbow method to determine no of cluster(k)")
plt.savefig('silh.png')


best_cluster_size = 100
df['Cluster-Labels'] = kmeans_model_dict[best_cluster_size].labels_

for i in range(0,best_cluster_size):
    print "Document Id's belonging to Cluster: ", i
    print df.loc[df['Cluster-Labels'] == i].index
    print "-----------------------------------------"
  