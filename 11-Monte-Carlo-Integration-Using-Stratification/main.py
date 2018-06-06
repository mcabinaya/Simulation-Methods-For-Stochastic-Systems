#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:49:30 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, uniform

def function_f(x):
    z = (3 - 0.8) * (1/ (1 + (np.sinh(2*x) * np.log(x))))
    return z
 
def function_p(x):
    z = uniform.pdf(x, loc=0.8, scale=3)
    return z

def function_g(x, mu, std):
    z = norm.pdf(x, loc=mu, scale=std)
    return z

print "MONTE CARLO ------------------------"

max_iterations = 50
monte_carlo_estimates = []
for i in range(0,max_iterations):    
    test_x1 = np.random.uniform(0.8,3,size=1000)
    #test_z1 = map(lambda t: function_f(t),test_x1)
    test_z1 = (3 - 0.8) * (1/ (1 + (np.sinh(2*test_x1) * np.log(test_x1))))
    monte_carlo_estimates.append(np.mean(test_z1))
    
print "Mean Integration Value- Simple Monte Carlo: ", np.mean(monte_carlo_estimates)
print "Variance Integration Value- Simple Monte Carlo: ", np.var(monte_carlo_estimates)

'''
test_x1 = np.linspace(0.8,3,1000)
test_z1 = (3 - 0.8) * (1/ (1 + (np.sinh(2*test_x1) * np.log(test_x1))))
plt.figure()
plt.plot(test_x1,test_z1)
plt.xlabel('Samples')
plt.ylabel('Function Value')
plt.title('Integration Function Plot - Stratification')
'''

print "\nMONTE CARLO USING STRATIFICATION ------------------------"

n=1000
n1=700
n2=300
value1 = []
value2 = []

for i in range (0,50):
    X1_1 = np.random.uniform(0.8, 1.24, n1) 
    Fnc1_1 = (1.24-0.8) * pow((1 + (np.sinh(2*X1_1)*np.log(X1_1))),-1) 
    value1_1 = (np.sum(Fnc1_1))/ n1
    
    X1_2 = np.random.uniform(1.24, 3, n2) 
    Fnc1_2 = (3-1.24)*pow((1 + (np.sinh(2*X1_2)*np.log(X1_2))),-1) 
    value1_2 = (np.sum(Fnc1_2))/ n2
    
    value2.append(value1_1 + value1_2)
variance_stratified_Fnc1 = np.var(value2)
Mean_stratified_Fnc1 = np.mean(value2)
print  "Mean Integration Value- Stratification: ", Mean_stratified_Fnc1
print  "Variance Integration Value- Stratification: ", variance_stratified_Fnc1

'''
plt.scatter(X1_1,Fnc1_1, color='r')
plt.scatter(X1_2,Fnc1_2, color='g')
'''

print "\nMONTE CARLO USING IMPORTANCE SAMPLING ------------------------"

importance_sampling_estimates = []
mu = 0.5
std = 0.7

for i in range(0,max_iterations):
    test_x3 = np.random.normal(loc=mu, scale=std, size=1000)
    
    fx3 =  (3 - 0.8) * (1/ (1 + (np.sinh(2*test_x3) * np.log(test_x3))))
    px3 = uniform.pdf(test_x3, loc=0.8, scale=3)
    gx3 = norm.pdf(test_x3, loc=mu, scale=std)
    
    z3 = (np.array(fx3)) * (np.array(px3) / np.array(gx3))
    z3 = z3[~np.isnan(z3)]
    importance_sampling_estimates.append(np.mean(z3))
    
print  "Mean Integration Value- Importance sampling: ", np.mean(importance_sampling_estimates)
print  "Variance Integration Value- Importance sampling: ", np.var(importance_sampling_estimates)



