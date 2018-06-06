#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:05:26 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Simulated_Annealing_Optimization():
    
    def __init__(self, proposal_pdf, sigma, T, max_iter, cooling_function):
        self.proposal_pdf = proposal_pdf
        self.sigma = sigma       
        self.T = T
        self.max_iter = max_iter
        self.cooling_function = cooling_function
    
    def visualize_cost_function(self):
        n = 2
        x1 = np.arange(-500, 500, 0.1)
        x2 = np.arange(-500, 500, 0.1)
        x1mesh, x2mesh = np.meshgrid(x1, x2, sparse=True)
        z = (418.9829 * n) - ((x1mesh * np.sin(np.sqrt(abs(x1mesh)))) + (x2mesh * np.sin(np.sqrt(abs(x2mesh)))))
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        plt.title('3D plot of the cost function')
        #plt.colorbar()
        
        plt.figure()
        plt.contourf(x1,x2,z)
        plt.title('Contour plot of the cost function')
        plt.colorbar()        
    
    def return_cost_value(self,x1,x2):
        n = 2
        if ((x1 >= -500) & (x1 <= 500) & (x2 >= -500) & (x2 <= 500)):
            return (418.9829 * n) - ((x1 * np.sin(np.sqrt(abs(x1)))) + (x2 * np.sin(np.sqrt(abs(x2)))))
        
    def proposal_step(self,previous_sample):
        if self.proposal_pdf == "Normal":
            sample = np.random.normal(loc=previous_sample, scale=self.sigma)
            if sample[0] < -500:
                sample[0] = 500
            if sample[0] > 500:
                sample[0] = 500
            if sample[1] < -500:
                sample[1] = -500
            if sample[1] > 500:
                sample[1] = 500
            return sample
        if self.proposal_pdf == "Cauchy":
            return np.random.standard_cauchy(size=2)
        
    
    def generate_initial_sample(self):
        initial_sample = np.random.uniform(-500,500, size=2)
        print "Initial Chosen Sample: ",initial_sample
        return initial_sample

    def return_gibbs_acceptance_probability(self, new_sample, previous_sample, iterT):
        cost_previous_sample = self.return_cost_value(previous_sample[0], previous_sample[1]) 
        cost_new_sample = self.return_cost_value(new_sample[0], new_sample[1]) 
        #print "---"
        #print new_sample, cost_new_sample
        #print previous_sample, cost_previous_sample
        #print iterT
        acceptance_probability = min(1, np.exp(-1 * (cost_new_sample - cost_previous_sample)/ iterT))
        return cost_previous_sample, cost_new_sample, acceptance_probability
    
    def return_iterTemp(self,n):
        if self.cooling_function == "Polynomial":
            return self.T * pow(n+1, -0.751)
        elif self.cooling_function == "Logarithmic":
            return self.T / np.log(n+1)
        elif self.cooling_function == "Exponential":
            return self.T / np.exp(n+1)
         
         
    def optimize(self):
        print "Plotting given cost function --"
        #self.visualize_cost_function()
        samples_generated = []
        initial_sample = self.generate_initial_sample()
        samples_generated.append(initial_sample)
        i=1
        n=0
        while (n < self.max_iter):
            #print "--------------"
            previous_sample = samples_generated[i-1]
            #print previous_sample
            new_sample = self.proposal_step(previous_sample)
            #print new_sample
            iterT = self.return_iterTemp(i)
            #print iterT
            cost_previous_sample, cost_new_sample, acceptance_probability = self.return_gibbs_acceptance_probability(new_sample, previous_sample, iterT)
            #print acceptance_probability
            if((cost_new_sample <= cost_previous_sample) or (np.random.uniform() <= acceptance_probability)):
            #if(acceptance_probability > 0.5):
                #print True
                samples_generated.append(new_sample)
                i+= 1
                #n-= 1
            n+= 1
        return samples_generated
 
  

      
optimize_1 = Simulated_Annealing_Optimization("Normal", sigma=200, T=20, max_iter=1000, cooling_function="Exponential")
samples_generated = optimize_1.optimize()
optimize_1.visualize_cost_function()

print "Converged to: ", samples_generated[-1]
samples_generated = np.array(samples_generated)
samples_generated = np.reshape(samples_generated, [len(samples_generated),2])
plt.plot(samples_generated[:,0], samples_generated[:,1], marker='*', color='orange')

'''
optimize_1 = Simulated_Annealing_Optimization("Normal", sigma=1, T=50, max_iter=100, cooling_function="Exponential")
samples_generated = optimize_1.optimize()
print "Converged to: ", samples_generated[-1]
        
optimize_1 = Simulated_Annealing_Optimization("Normal", sigma=1, T=100, max_iter=100, cooling_function="Exponential")
samples_generated = optimize_1.optimize()
print "Converged to: ", samples_generated[-1]
        
optimize_1 = Simulated_Annealing_Optimization("Normal", sigma=1, T=1000, max_iter=100, cooling_function="Exponential")
samples_generated = optimize_1.optimize()
print "Converged to: ", samples_generated[-1]
'''