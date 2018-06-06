#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:28:47 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

n_list = [100, 1000, 5000, 10000, 100000]
dim = 2
k = 50

sample_variance_list = []

for n in n_list:
    print "-----------",n
    pi_estimate_list = []
    
    for iteration in range(0,k):
        uniform_trials= np.random.uniform(size = [n,dim])
        area = np.sum(uniform_trials**2, axis=1)
        no_points_inside = sum(area<=1)
        pi_estimate = 4 * (float(no_points_inside)/float(n))
        pi_estimate_list.append(pi_estimate)
        
    plt.figure()
    plt.hist(pi_estimate_list, bins=50)
    plt.title('Histogram of pi estimates using '+str(n)+' samples')
    plt.xlabel('Pi estimated values')
    plt.ylabel('Frequency')
    
    sample_variance_list.append(np.var(pi_estimate_list))
    print "Mean pi Estimate: ", np.mean(pi_estimate_list)
    print "Sample Variance: ", np.var(pi_estimate_list)
    
plt.figure()
plt.plot(n_list, sample_variance_list)
plt.title('Sample Variance for different Values of n')
plt.xlabel('Varying n')
plt.ylabel('Variance Value')
