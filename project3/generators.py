

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from itertools import product

def generate(Shape):
    keys =  [] 
    for i in range(Shape[0] ):
        for j in range( Shape[1]):
            keys.append( [i,j]) 
    return keys 

def Chi_list_generator(keys):
        
    Chi_list = {}
    for k_ in keys :
        Chi_list[str(k_)] = [] 
    return Chi_list

def Indices_generate_samples( Indices  ,  Sampling_rates , keys):
    d = len(Indices) +1
    Lists = [] 

    Indices_sample = {}
    
    n_Edges_per_pixel =1
    for i in range(d-1):
        Indices_sample[str(i)] =     Indices[str(i)][  : int(Indices[str(i)].shape[0] * Sampling_rates[i]   ) ] 
        n_Edges_per_pixel  = n_Edges_per_pixel * len(Indices_sample[str(i)] ) 
        #Lists.append(   np.arange( len(Indices_sample[str(i)])   ) )  
        Lists.append(   Indices_sample[str(i)]  ) 
    
    Indices_sample[str(d-1)] =  keys
    
    Lists.append(   Indices_sample[str(d-1)]  ) 

    return Lists , Indices_sample  , n_Edges_per_pixel
