#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:22:43 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby

### Using the same Counting success code from Question 1

def countingSuccesses(k, p, no_of_trials):
    binomial_trial_sucess = []
    
    for i in range(0,no_of_trials):
        uniform_trials= np.random.uniform(size = k)
        bernoulli_trials = np.copy(uniform_trials)
        bernoulli_trials[uniform_trials <= p] = 1
        bernoulli_trials[uniform_trials > p] = 0
        binomial_trial_sucess.append(sum(bernoulli_trials))
    
    binomial_trial_sucess = pd.DataFrame(binomial_trial_sucess)
    count_to_plot_b = binomial_trial_sucess[0].value_counts().sort_index()
    plt.figure()
    count_to_plot_b.plot(kind='bar')
    plt.title('Success Count of k = '+str(k)+' fair bernoulli trials - Simulated '+str(no_of_trials)+' times')

countingSuccesses(5, 0.5, 300)
countingSuccesses(10, 0.5, 300)
countingSuccesses(30, 0.5, 300)
countingSuccesses(50, 0.5, 300)

