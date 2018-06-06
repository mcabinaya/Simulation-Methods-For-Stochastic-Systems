#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:22:35 2018

@author: abinaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby, combinations
import networkx as nx

def returnBernoulliTrials(k, p):
    uniform_trials= np.random.uniform(size = k)
    bernoulli_trials = np.copy(uniform_trials)
    bernoulli_trials[uniform_trials <= p] = 1
    bernoulli_trials[uniform_trials > p] = 0
    return bernoulli_trials

def returnGraph(nodes_list,edges_list):
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)
    return G


def networking(n,p):  
    ### Number of edges that can be selected
    N = (n * (n - 1)) / 2
    edge_df = pd.DataFrame(list(combinations(range(1,n+1) , 2)), columns=['Node1','Node2'])
    edge_df['Edges'] = edge_df[['Node1', 'Node2']].apply(tuple, axis=1)
    
    bernoulli_trials = returnBernoulliTrials(N, p)
    edge_df['Selection'] = bernoulli_trials
    
    ### Form Nodes and Edges
    nodes_list = range(1,n+1)
    edge_df_edgepresent = edge_df.loc[edge_df['Selection'] == 1]
    
    ### Form Graphs
    G = returnGraph(nodes_list,list(edge_df_edgepresent['Edges']))
    
    ### Graph Visualization
    plt.figure()
    nx.draw(G,with_labels = True)
   
    ### Get Degree of Vertex
    vertex_df = pd.DataFrame(nodes_list, columns=['Vertex'], index=nodes_list)
    vertex_df['DOV'] = edge_df_edgepresent['Node1'].value_counts().sort_index()
    
    ### Get Hisogram of DOV
    hist_dov = vertex_df['DOV'].value_counts().sort_index()
    plt.figure()
    plt.bar(hist_dov.index, hist_dov.values)
    plt.title('Histogram of Degree of Vertex n = '+ str(n)+' p = '+str(p))
    plt.xlabel('Degree Of Vertex')
    plt.ylabel('Frequency')
    

### Given parameters
networking(50,0.02)
networking(50,0.09)
networking(50,0.12)
networking(100,0.06)



    
    


