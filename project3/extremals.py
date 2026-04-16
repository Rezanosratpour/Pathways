

import numpy as np
from scipy.stats import rankdata

    
def Theta2(Fx, Fy):
    nu = 0.5* np.mean( np.absolute( Fx -Fy) , 1)
    return (1+2*nu) /(1-2*nu)
  
def extremal_coeff_madogram(x, y):
    n = len(x)
    # Empirical CDFs
    Fx = pd.Series(x).rank(method="average") / (n + 1)
    Fy = pd.Series(y).rank(method="average") / (n + 1)

    # F-madogram
    nu_F = np.mean(np.abs(Fx - Fy)) / 2
    theta_hat = (1 + 2 * nu_F) / (1 - 2 * nu_F)
    return theta_hat

def Multivar_Chi(df , P):
    """
    Multivariate empirical chi coefficient.

    Parameters:
        df (pd.DataFrame): Input data as Data Frame.
        p (float or np.ndarray): Threshold for data2mpareto (optional).

    Returns:
        float: Empirical chi coefficient.
    """
    K = list(df.keys()) 
    qu = df
    for k in K :
        pr = np.percentile(df[k].values ,  P *100)
        qu = qu[qu[k]>=pr]

    return len(qu) /len(df) / (1-P)

import numpy as np

def Multivar_Chi2(X, P , thrsh =1.0):
    """
    Multivariate empirical chi coefficient (NumPy version).

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_variables)
    P : float
        Quantile threshold in [0, 1], e.g. 0.95

    Returns
    -------
    float
        Empirical chi coefficient
    """
    # Ensure array and remove NaNs if any
    X = np.asarray(X)
    X = X[~np.isnan(X).any(axis=1)]
    n, d = X.shape

    X = X[X >= thrsh ]

    # Compute column-wise percentile thresholds
    thresholds = np.percentile(X, P * 100, axis=0)

    # Boolean mask for rows where all columns exceed their thresholds
    mask = np.all(X >= thresholds, axis=1)

    # Empirical chi = proportion of exceedances normalized by (1-P)
    chi = np.sum(mask) / n / (1 - P)
    return chi

def Multivar_Chi3(U, P):
    """
    Multivariate empirical chi coefficient (NumPy version).

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_variables)
    P : float
        Quantile threshold in [0, 1], e.g. 0.95

    Returns
    -------
    float
        Empirical chi coefficient
    """
    n, d = U.shape
    
    # Compute column-wise percentile thresholds
    
    # Boolean mask for rows where all columns exceed their thresholds
    mask = np.all(U >= P, axis=1)

    # Empirical chi = proportion of exceedances normalized by (1-P)
    chi = np.sum(mask) / n / (1 - P)
    return chi