import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

def to_empirical_cdf(X):
    """
    Transform an array X into empirical CDF values in [0, 1].
    """
    X = np.asarray(X)
    ranks = rankdata(X, method='average')  # Rank the data
    return ranks / (len(X) + 1)  # Convert ranks to CDF values


def Multi_dim_CDF_transform( X):
    Z_array =  []
    Shape = X.shape
    Z = np.zeros( Shape) 
    for i in range(Shape[1]):
        for j in range(Shape[2]):
            #Z[: , i ,j ] = to_empirical_cdf(X[: , i , j ])
            Z_array.append(  to_empirical_cdf(X[: , i , j ]) ) 
    Z_array  = np.array(Z_array) 
    return Z_array 
