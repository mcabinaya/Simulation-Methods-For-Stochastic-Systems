#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 19:07:34 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class Traveling_Salesmen_problem():
    
    def __init__(self, no_cities, T, cooling_function, max_iter):
        self.no_cities = no_cities
        self.T = T
        self.cooling_function = cooling_function
        self.max_iter = max_iter
    
    def generate_cities(self):
        return [random.sample(range(0,1000), 2) for x in range(self.no_cities)]
    
    def generate_initial_random_tour(self):
        return random.sample(range(self.no_cities),self.no_cities)
    
    def travel_cost(self, tour):
        cost = 0
        for i in range(0, self.no_cities-1):
            from_city_index = tour[i]
            to_city_index = tour[i+1]
            from_city = self.cities[from_city_index]
            to_city = self.cities[to_city_index]
            cost += np.linalg.norm(np.array(from_city) - np.array(to_city))
        return cost
    
    def return_iterTemp(self,n):
        if self.cooling_function == "Polynomial":
            return self.T * pow(n+1, -0.751)
        elif self.cooling_function == "Logarithmic":
            return self.T / np.log(n+1)
        elif self.cooling_function == "Exponential":
            return self.T / np.exp(n+1)
        elif self.cooling_function == "Other":
            return self.T * 0.99
      
    def generate_acceptance_probability(self, tour_cost, swapped_tour_cost, iterT):
        return np.exp((tour_cost - swapped_tour_cost) / iterT)
        
    
    def solve_problem(self):
        self.cities = self.generate_cities()
        tour = self.generate_initial_random_tour()
        initial_tour = tour[:]
        
        cost_to_plot = []

        for i in range(0,self.max_iter):
            swap_positions =  random.sample(range(self.no_cities),2)
            swapped_tour = tour[:]
            swapped_tour[swap_positions[0]] = tour[swap_positions[1]]
            swapped_tour[swap_positions[1]] = tour[swap_positions[0]]
            
            tour_cost = self.travel_cost(tour)
            swapped_tour_cost = self.travel_cost(swapped_tour)
            
            iterT = self.return_iterTemp(i)
            
            acceptance_probability = self.generate_acceptance_probability(tour_cost, swapped_tour_cost, iterT)
            if(np.random.uniform() <= acceptance_probability):
                tour = swapped_tour[:]
                cost_to_plot.append(swapped_tour_cost)
                
        return initial_tour,tour, cost_to_plot
            
    

no_cities = 400
tr_sa_pr = Traveling_Salesmen_problem(no_cities, 1, "Other", 5000)
initial_tour,tour, cost_to_plot = tr_sa_pr.solve_problem()
oldx = zip(*[tr_sa_pr.cities[initial_tour[i % no_cities]] for i in range(no_cities+1)])[0]
oldy = zip(*[tr_sa_pr.cities[initial_tour[i % no_cities]] for i in range(no_cities+1)])[1]
newx = zip(*[tr_sa_pr.cities[tour[i % no_cities]] for i in range(no_cities+1)])[0]
newy = zip(*[tr_sa_pr.cities[tour[i % no_cities]] for i in range(no_cities+1)])[1]

plt.figure()
plt.plot(oldx, oldy, marker='o', color='r')
plt.grid('on')
plt.title('Initial Random Tour: Number of Cities = '+str(no_cities))

plt.figure()
plt.plot(newx, newy, color='g', marker='*')
plt.grid('on')
plt.title('Optmized Tour: Number of Cities = '+str(no_cities))

plt.figure()
plt.plot(cost_to_plot)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Number of Cities = '+str(no_cities))
