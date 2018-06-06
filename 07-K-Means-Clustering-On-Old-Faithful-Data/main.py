#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:49:05 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = np.loadtxt("old-faithful.txt")

plt.figure()
plt.scatter(X[:,1], X[:,2])
plt.title("Scatter plot of old-faithful data")
plt.xlabel("Duration of old faithful geyser eruptions")
plt.ylabel("waiting time between eruptions")

kmeans_model = KMeans(n_clusters=2)
kmeans_model.fit(X[:,1:])

plt.figure()
plt.scatter(X[:,1], X[:,2], c=kmeans_model.labels_, cmap="jet")
plt.title("Scatter plot of old-faithful data - Clustered")
plt.xlabel("Duration of old faithful geyser eruptions")
plt.ylabel("waiting time between eruptions")



