#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:27:26 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, uniform
from scipy.stats import multivariate_normal

def function_f(x,y):
    z = math.pow((math.pi*2),2) * (np.exp((-1 * math.pow(x,4)) + (-1 * math.pow(y,4))))
    return z

def function_p(x):
    z1 = uniform.pdf(x[0], loc=-math.pi, scale=math.pi)
    z2 = uniform.pdf(x[1], loc=-math.pi, scale=math.pi)
    return z1*z2

def function_g(x, mu, std):
    z = multivariate_normal.pdf(x, mean=mu, cov=std) 
    return z
   

test_x2 = np.linspace(-math.pi, math.pi,100)
test_y2 = np.linspace(-math.pi, math.pi,100)
xx, yy = np.meshgrid(test_x2, test_y2)
test_xy = np.hstack( (xx.reshape(xx.shape[0]*xx.shape[1], 1, order='F'), yy.reshape(yy.shape[0]*yy.shape[1], 1, order='F')))
test_z2 = map(lambda t:function_f(t[0],t[1]), test_xy)

fig = plt.figure()
ax = Axes3D(fig)
num_pts = np.shape(test_xy)[0]
xs_plot = np.reshape(test_xy[:,0], [num_pts])
ys_plot = np.reshape(test_xy[:,1], [num_pts])
zs_plot = np.reshape(test_z2, [num_pts])
ax.plot(xs=xs_plot, ys=ys_plot, zs=xs_plot) 
plt.xlabel('Samples 1')
plt.ylabel('Samples 2')
plt.zlabel('Function Value')
plt.title('Integration Function Plot')

zz =  zs_plot.reshape(100,100)
plt.figure()
plt.contour(xx, yy, zz)
plt.xlabel('Samples 1')
plt.ylabel('Samples 2')
plt.title('Integration Function Plot - contour')


print "MONTE CARLO ------------------------"

max_iterations = 50

monte_carlo_estimates = []
for i in range(0,max_iterations):   
    test_x1 = np.random.uniform(-math.pi, math.pi,1000)
    test_y1 = np.random.uniform(-math.pi, math.pi,1000)
    xx, yy = np.meshgrid(test_x1, test_y1)
    test_xy = np.hstack((xx.reshape(xx.shape[0]*xx.shape[1], 1, order='F'), yy.reshape(yy.shape[0]*yy.shape[1], 1, order='F')))
    test_z1 = math.pow((math.pi*2),2) * (np.exp((-1 * pow(test_xy[:,0],4)) + (-1 * pow(test_xy[:,1],4))))
    monte_carlo_estimates.append(np.mean(test_z1) )
    
print "Mean Integration Value- Simple Monte Carlo: ", np.mean(monte_carlo_estimates)
print "Variance Integration Value- Simple Monte Carlo: ", np.var(monte_carlo_estimates)

print "\nMONTE CARLO USING STRATIFICATION ------------------------"
value3 = []
value4 = []
n = 1000

for i in range (0,50): 
    X2_1 = np.random.uniform(-1.73, 1.73, n) 
    Y2_1 = np.random.uniform(-1.73, 1.73, n) 
    Fnc2_1 = pow(math.e, (-pow(X2_1,4)-pow(Y2_1,4))) 
    value3_1 = ((1.73+1.73)**2) * np.sum(Fnc2_1)/ n
    value4.append(value3_1)
variance_stratified_Fnc2 = np.var(value4)
Mean_stratified_Fnc2 = np.mean(value4)
print  "Mean Integration Value- Stratification: ", Mean_stratified_Fnc2
print  "Variance Integration Value- Stratification: ", variance_stratified_Fnc2

print "\nMONTE CARLO USING IMPORTANCE SAMPLING ------------------------"

importance_sampling_estimates = []
mu = 0
std = 0.5

mu_mat = np.array([mu, mu])

for i in range(0,max_iterations):
    #print "--- iteration ", i
    test_x3 = np.random.normal(loc=mu,size=1000)
    test_y3 = np.random.normal(loc=mu,size=1000)

    xx, yy = np.meshgrid(test_x3, test_y3)
    test_xy = np.hstack((xx.reshape(xx.shape[0]*xx.shape[1], 1, order='F'), yy.reshape(yy.shape[0]*yy.shape[1], 1, order='F')))
    
    fx3 = math.pow((math.pi*2),2) * (np.exp((-1 * pow(test_xy[:,0],4)) + (-1 * pow(test_xy[:,1],4))))
    z1 = uniform.pdf(test_xy[:,0], loc=-math.pi, scale=math.pi+math.pi)
    z2 = uniform.pdf(test_xy[:,1], loc=-math.pi, scale=math.pi+math.pi)
    px3 = z1*z2
    gx3 = multivariate_normal.pdf(test_xy, mean=mu_mat)
    
    
    z3 = (np.array(fx3)) * (np.array(px3) / np.array(gx3))
    z3 = z3[~np.isnan(z3)]
    importance_sampling_estimates.append(np.mean(z3))
    
print  "Mean Integration Value- Importance sampling: ", np.mean(importance_sampling_estimates)
print  "Variance Integration Value- Importance sampling: ", np.var(importance_sampling_estimates)


