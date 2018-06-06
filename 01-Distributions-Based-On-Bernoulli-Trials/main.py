#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:33:00 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby

### Question 1 - a : Generate 100 Bernoulli trials
### Generate uniform samples and assign 1 if the samples are > 0.5 and 0 if samples are < 0.5

uniform_trials_100 = np.random.uniform(size = 100)
bernoulli_trials_100 = np.copy(uniform_trials_100)
bernoulli_trials_100[uniform_trials_100 <= 0.5] = 1
bernoulli_trials_100[uniform_trials_100 > 0.5] = 0

bernoulli_trials_100 = pd.DataFrame(bernoulli_trials_100)
count_to_plot_a = bernoulli_trials_100[0].value_counts().sort_index()
count_to_plot_a.plot(kind='bar')
plt.title('100 Bernoulli trials - Histogram')

### ---------------

### Question 1 - b : Generate 100 trials: 7 Bernoulli trials
### Its a Binomial trial with n=7 and size = 100, p = 0.5 (fair bernoulli trials)
### Returns the number of successes in 7 bernouli trials
binomial_trial_sucess = []

for i in range(0,100):
    uniform_trials_7 = np.random.uniform(size = 7)
    bernoulli_trials_7 = np.copy(uniform_trials_7)
    bernoulli_trials_7[uniform_trials_7 <= 0.5] = 1
    bernoulli_trials_7[uniform_trials_7 > 0.5] = 0
    binomial_trial_sucess.append(sum(bernoulli_trials_7))

binomial_trial_sucess = pd.DataFrame(binomial_trial_sucess)
count_to_plot_b = binomial_trial_sucess[0].value_counts().sort_index()
plt.figure()
count_to_plot_b.plot(kind='bar')
plt.title('Success Count of 7 fair bernoulli trials - Simulated 100 times')

### ---------------

### Question 1 - c : Generate 100 Bernoulli samples  : Count the longest run of heads, R epest 100 times
longest_run_of_heads = []
for i in range(0,500):
    uniform_trials_100 = np.random.uniform(size = 100)
    bernoulli_trials_100 = np.copy(uniform_trials_100)
    bernoulli_trials_100[uniform_trials_100 <= 0.5] = 1
    bernoulli_trials_100[uniform_trials_100 > 0.5] = 0    
    longest_heads = max(len(list(v)) for k,v in groupby(bernoulli_trials_100) if k==1.0)
    longest_run_of_heads.append(longest_heads)
    
longest_run_of_heads = pd.DataFrame(longest_run_of_heads)
count_to_plot_c = longest_run_of_heads[0].value_counts().sort_index()
plt.figure()
count_to_plot_c.plot(kind='bar')
plt.title('Count of longest run of head in 100 bernoulli trials - Simulated 500 times')


