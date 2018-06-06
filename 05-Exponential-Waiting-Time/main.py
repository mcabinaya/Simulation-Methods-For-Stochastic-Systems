#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:13:05 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby
from scipy.stats import *

k = 1000
theta = 0.2
no_of_trials = 1

### Generate Uniform samples
uniform_trials= np.random.uniform(size = k)

### Generate Exponential samples from Uniform samples -- Observed
exponential_trials = - (theta) * np.log(1 - uniform_trials)
plt.figure()
observed_bin_hist = plt.hist(exponential_trials, bins=80)[0]
plt.title('Histogram of Observed Samples from Inverse CDF method')

### Generate Exponential samples using inbuilt functions -- Expected
expected_exponential_trials = np.random.exponential(scale = theta, size=k)
plt.figure()
expected_bin_hist = plt.hist(expected_exponential_trials, bins=60)[0]
plt.title('Histogram of Expected Samples using Python In-Built Function')

### Chisquare Goodness of Fit Test
chisq, p = chisquare(observed_bin_hist[:30], expected_bin_hist[:30])
print chisq, p

ks, p = kstest(exponential_trials, cdf='expon')
print ks, p

### Histogram of count
count = []
temp_sum = 0
temp_count = 0
for i in exponential_trials:
    temp_sum += i
    temp_count += 1
    if temp_sum > 1:
        count.append(temp_count)
        temp_sum = 0
        temp_count = 0
        
plt.figure()
plt.hist(count, bins=50)
plt.title('Histogram of Time Intervals that Occur in 1 Time Count')

        
    