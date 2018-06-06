#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:24:07 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm
import scipy
#plt.close('all')

class Metropolis_Hastings():
    
    def __init__(self, proposal_pdf, sigma):
        self.proposal_pdf = proposal_pdf
        self.sigma = sigma              
    
    def return_stationary_pdf(self,x):
        return ((0.6*beta.pdf(x,1,8)) + (0.4*beta.pdf(x,9,1)))
    
    def return_proposal_pdf(self,x,mu):
        if self.proposal_pdf == "Normal":
            return norm.pdf(x, loc=mu, scale=self.sigma)
        if self.proposal_pdf == "Cauchy":
            return scipy.stats.cauchy.pdf(x, loc=mu, scale=self.sigma)
        

    def generate_initial_sample(self):
       while True:
           sample = np.random.uniform(-1,1)
           sample_pdf = self.return_stationary_pdf(sample)
           if sample_pdf!= 0:
               print "Initial Chosen Sample: ",sample
               return sample
               break

    def proposal_step(self,previous_sample):
        if self.proposal_pdf == "Normal":
            return np.random.normal(loc=previous_sample, scale=self.sigma)
        elif self.proposal_pdf == "Cauchy":
            return scipy.stats.cauchy.rvs(loc=previous_sample)

    
    def acceptance_step(self, new_sample, previous_sample):
        num1 = self.return_stationary_pdf(new_sample)
        num2 = self.return_proposal_pdf(previous_sample,new_sample)
        den1 = self.return_stationary_pdf(previous_sample)
        den2 = self.return_proposal_pdf(new_sample,previous_sample)   
        acceptance_probability = min(1, ((num1*num2)/(den1*den2)))
        return acceptance_probability
    
    def generate_samples(self, no_samples):
        samples_generated = []
        samples_generated_pdf = []
        initial_sample = self.generate_initial_sample()
        samples_generated.append(initial_sample)
        samples_generated_pdf.append(self.return_stationary_pdf(initial_sample))
        i=1
        while (i < no_samples):
            previous_sample = samples_generated[i-1]
            new_sample = self.proposal_step(previous_sample)
            acceptance_probability = self.acceptance_step(new_sample, previous_sample)
            if(np.random.uniform() <= acceptance_probability):
            #if(acceptance_probability > 0.5):
                samples_generated.append(new_sample)
                samples_generated_pdf.append(self.return_stationary_pdf(new_sample))
                i+= 1
        return samples_generated, samples_generated_pdf


trial = Metropolis_Hastings(proposal_pdf="Normal",sigma=10) 
orig_pdf = []  
x_samples = []
for x in np.arange(0,1, 0.01):
    x_samples.append(x)
    orig_pdf.append(trial.return_stationary_pdf(x))
plt.figure()
plt.plot(x_samples, orig_pdf)
plt.title('Given Stationary Mixed Distribution')
plt.xlabel('x')
plt.ylabel('pdf')

mh_samples_0_001 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=0.001)      
samples_generated, samples_generated_pdf = mh_samples_0_001.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 0.001')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)
plt.title('Histogram of samples - Proposal pdf = Cauchy; Sigma = 0.001')

###
mh_samples_0_01 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=0.01)      
samples_generated, samples_generated_pdf = mh_samples_0_01.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 0.01')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)
plt.title('Histogram of samples - Proposal pdf = Cauchy; Sigma = 0.01')

###
mh_samples_0_1 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=0.1)      
samples_generated, samples_generated_pdf = mh_samples_0_1.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 0.1')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)
plt.title('Histogram of samples - Proposal pdf = Cauchy; Sigma = 0.1')

###
mh_samples_1 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=1)      
samples_generated, samples_generated_pdf = mh_samples_1.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 1')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)
plt.title('Histogram of samples - Proposal pdf = Cauchy; Sigma = 1')

###
mh_samples_5 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=5)      
samples_generated, samples_generated_pdf = mh_samples_5.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 5')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)
plt.title('Histogram of samples - Proposal pdf = Cauchy; Sigma = 5')

###
mh_samples_10 = Metropolis_Hastings(proposal_pdf="Cauchy",sigma=10)      
samples_generated, samples_generated_pdf = mh_samples_10.generate_samples(5000)
plt.figure()
plt.plot(samples_generated,marker='*')
plt.title('Samples Generated - Proposal pdf = Cauchy; Sigma = 10')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.figure()
plt.hist(samples_generated, bins=100)


