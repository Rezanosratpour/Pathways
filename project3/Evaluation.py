import numpy as np 
import numpy as np
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr, spearmanr

import tifffile as tff 

import sys 

from pathlib import Path
# .parent gets the directory containing the file
script_dir = Path(__file__).resolve().parent
sys.path.insert(0,script_dir)


import cateval

def corr(x, y) :

    return np.corrcoef(x, y)[0,1] 

def RMSE(x, y):
    return np.sqrt(np.mean((x-y)**2))

def corr_spearman(x , y ):
    return spearmanr(x, y)[0]

def CSI(obs, model):
    ev = cateval.Categorical(model , obs )
    ev.evaluate(20)
    return ev.CSI


def evaluate_2D(y_true_3D , y_pred_3d , function):
    n , row , col  ,  = y_true_3D.shape
    eval_image = np.zeros((row, col) )
    
    for i in range(row):
        for j in range(col):
            eval_image[i,j] = function( y_true_3D[: , i , j]  , y_pred_3d[: , i ,j ]
                                )
    return eval_image
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _mask_valid(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def _rankdata(a):
    """Simple average-rank for ties (fallback if scipy is unavailable)."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def _pearson_corr(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    if _HAVE_SCIPY:
        return _safe_float(pearsonr(y_true, y_pred)[0])
    # fallback
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt**2).sum() * (yp**2).sum())
    return _safe_float((yt * yp).sum() / denom) if denom > 0 else np.nan


def _spearman_corr(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    if _HAVE_SCIPY:
        return _safe_float(spearmanr(y_true, y_pred).correlation)
    # fallback: Pearson on ranks
    return _pearson_corr(_rankdata(y_true), _rankdata(y_pred))


def _kendall_tau(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    if _HAVE_SCIPY:
        return _safe_float(kendalltau(y_true, y_pred).correlation)
    return np.nan  # (fallback not implemented)


def _rmse(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    return _safe_float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else np.nan


def _mae(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    return _safe_float(np.mean(np.abs(y_pred - y_true))) if y_true.size else np.nan


def _mse(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    return _safe_float(np.mean((y_pred - y_true) ** 2)) if y_true.size else np.nan


def _bias(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    return _safe_float(np.mean(y_pred - y_true)) if y_true.size else np.nan


def _r2(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return _safe_float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _explained_variance(y_true, y_pred):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    var_y = np.var(y_true)
    var_err = np.var(y_true - y_pred)
    return _safe_float(1.0 - var_err / var_y) if var_y > 0 else np.nan


def _mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return _safe_float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0) if y_true.size else np.nan


def _smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = _mask_valid(y_true, y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return _safe_float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0) if y_true.size else np.nan


def _nse(y_true, y_pred):
    # Nash–Sutcliffe Efficiency
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return _safe_float(1.0 - num / denom) if denom > 0 else np.nan


def _kge(y_true, y_pred):
    # Kling–Gupta Efficiency (2009 form)
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    r = _pearson_corr(y_true, y_pred)
    mu_y, mu_x = np.mean(y_true), np.mean(y_pred)
    sig_y, sig_x = np.std(y_true), np.std(y_pred)
    alpha = sig_x / sig_y if sig_y > 0 else np.nan
    beta = mu_x / mu_y if np.abs(mu_y) > 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return _safe_float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def _quantile_mae(y_true, y_pred, q=0.95):
    """MAE restricted to samples above the q-quantile of y_true (extremes focus)."""
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size < 2:
        return np.nan
    thr = np.quantile(y_true, q)
    m = y_true >= thr
    return _safe_float(np.mean(np.abs(y_pred[m] - y_true[m]))) if np.any(m) else np.nan


def _threshold_event_scores(y_true, y_pred, threshold):
    """
    Binary event verification: event = value >= threshold
    Returns POD, FAR, CSI, BiasScore, TP/FP/FN/TN.
    """
    y_true, y_pred = _mask_valid(y_true, y_pred)
    if y_true.size == 0:
        return {
            "POD": np.nan, "FAR": np.nan, "CSI": np.nan, "BiasScore": np.nan,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0
        }
    obs = y_true >= threshold
    fc  = y_pred >= threshold
    TP = int(np.sum(obs & fc))
    FP = int(np.sum(~obs & fc))
    FN = int(np.sum(obs & ~fc))
    TN = int(np.sum(~obs & ~fc))

    POD = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FAR = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan
    BiasScore = (TP + FP) / (TP + FN) if (TP + FN) > 0 else np.nan

    return {"POD": POD, "FAR": FAR, "CSI": CSI, "BiasScore": BiasScore,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def _spatial_metric_reduce(y_true_2d, y_pred_2d, func, reduce="mean"):
    """
    Apply a metric func to each (i,j) pixel's time-series (or last axis),
    then reduce to scalar.
    """
    y_true_2d = np.asarray(y_true_2d)
    y_pred_2d = np.asarray(y_pred_2d)
    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_2d.shape} vs y_pred {y_pred_2d.shape}")

    # expects (T, H, W) OR (H, W, T). Try to standardize to (T, H, W)
    if y_true_2d.ndim != 3:
        raise ValueError("Expected 3D arrays (T,H,W) or (H,W,T).")

    # if last axis looks like spatial (H,W), keep; else transpose
    # Heuristic: if first axis is much larger, assume it's time.
    T_first = y_true_2d.shape[0]
    T_last  = y_true_2d.shape[-1]
    if T_last > 3 and T_first <= 3:  # likely (H,W,T)
        y_true = np.transpose(y_true_2d, (2, 0, 1))
        y_pred = np.transpose(y_pred_2d, (2, 0, 1))
    else:
        y_true, y_pred = y_true_2d, y_pred_2d

    T, H, W = y_true.shape
    out = np.full((H, W), np.nan, dtype=float)
    for i in range(H):
        for j in range(W):
            out[i, j] = func(y_true[:, i, j], y_pred[:, i, j])

    if reduce == "mean":
        return _safe_float(np.nanmean(out))
    if reduce == "median":
        return _safe_float(np.nanmedian(out))
    if reduce == "map":
        return out
    raise ValueError("reduce must be one of: 'mean', 'median', 'map'")


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

import numpy as np

def bivariate_Chi( x, y , P = 0.95):
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
    X = np.column_stack((x, y))  # shape (n_samples, 2)
    X = np.asarray(X)
    X = X[~np.isnan(X).any(axis=1)]
    n, d = X.shape

    # Compute column-wise percentile thresholds
    thresholds = np.percentile(X, P * 100, axis=0)

    # Boolean mask for rows where all columns exceed their thresholds
    mask = np.all(X >= thresholds, axis=1)

    # Empirical chi = proportion of exceedances normalized by (1-P)
    chi = np.sum(mask) / n / (1 - P)
    return chi



def evaluate_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, MBE, Pearson r, Spearman ρ, NSE, KGE.

    y_true, y_pred: arrays of same shape (can be 2D/3D/4D). 
                    They will be flattened.
    """

    # Flatten and mask NaNs
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Basic differences
    diff = y_pred - y_true

    # --- Error metrics ---
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    mbe = np.mean(diff)   # bias (pred - obs)

    # --- Correlations ---
    if len(y_true) > 1:
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_rho, _ = spearmanr(y_true, y_pred)
    else:
        pearson_r, spearman_rho = np.nan, np.nan

    # --- Nash–Sutcliffe Efficiency (NSE) ---
    denom_nse = np.sum((y_true - np.mean(y_true))**2)
    if denom_nse == 0:
        nse = np.nan
    else:
        nse = 1 - np.sum((y_pred - y_true)**2) / denom_nse

    # --- Kling–Gupta Efficiency (KGE) ---
    mu_o = np.mean(y_true)
    mu_s = np.mean(y_pred)
    sigma_o = np.std(y_true, ddof=1)
    sigma_s = np.std(y_pred, ddof=1)

    if sigma_o == 0 or sigma_s == 0 or mu_o == 0:
        kge = np.nan
    else:
        r_kge = pearson_r
        alpha = sigma_s / sigma_o
        beta = mu_s / mu_o
        kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    chi90 = bivariate_Chi(y_true, y_pred, P=0.90)
    chi95 = bivariate_Chi(y_true, y_pred, P=0.95)
    chi99 = bivariate_Chi(y_true, y_pred, P=0.99)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MBE": mbe,
        "Pearson_r": pearson_r,
        "Spearman_rho": spearman_rho,
        "NSE": nse,
        "KGE": kge,
        "Chi90": chi90,
        "Chi95": chi95,
        "Chi99": chi99,
    }
