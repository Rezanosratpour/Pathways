from itertools import product
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point

import sys

from pathlib import Path
# .parent gets the directory containing the file
script_dir = Path(__file__).resolve().parent
sys.path.insert(0,script_dir)

from extremals import Multivar_Chi3


def _flat_combo_to_coords(combo, ncols):
    if combo is None:
        return None
    combo = np.asarray(combo, dtype=np.int64)
    return np.column_stack((combo // ncols, combo % ncols))




def _eval_station_batch(batch, flat_cdf, starts, ends, chi_q):
    """
    Evaluate one batch of combinations.
    Returns:
        targets   : flat indices of the terminal node of each combo
        chi_vals  : chi value for each combo
        best_combo: combo with highest chi in this batch
        best_chi  : highest chi in this batch
    """
    d = len(starts)
    t_eff = int(ends[0] - starts[0])

    X = np.empty((d, t_eff), dtype=flat_cdf.dtype)
    targets = np.empty(len(batch), dtype=np.int64)
    chi_vals = np.empty(len(batch), dtype=np.float64)

    best_combo = None
    best_chi = -np.inf

    for b, combo in enumerate(batch):
        for i_d, pix in enumerate(combo):
            X[i_d, :] = flat_cdf[pix, starts[i_d]:ends[i_d]]

        chi = Multivar_Chi3(X.T, chi_q)

        targets[b] = combo[-1]
        chi_vals[b] = chi

        if chi > best_chi:
            best_chi = chi
            best_combo = combo

    return targets, chi_vals, best_combo, best_chi


def _coords_to_flat(coords, ncols):
    coords = np.asarray(coords)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return coords[:, 0].astype(np.int64) * ncols + coords[:, 1].astype(np.int64)


def _unique_combo_batches2(flat_lists, batch_size):
    """
    Lazily generate unique combinations in batches.
    """
    seen = set()
    batch = []

    for combo in product(*flat_lists):
        if combo in seen:
            continue
        seen.add(combo)
        batch.append(combo)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

def _unique_combo_batches(flat_lists, seen, batch_size):
    """
    Generate unique combinations lazily in batches.
    """
    batch = []
    for combo in product(*flat_lists):
        if combo in seen:
            continue
        seen.add(combo)
        batch.append(combo)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _evaluate_combo_batch(batch, flat_cdf, starts, ends, chi_q):
    """
    Worker: evaluate one batch of unique combinations.
    Returns target pixels and corresponding chi values.
    """
    d = len(starts)
    T_eff = int(ends[0] - starts[0])

    targets = np.empty(len(batch), dtype=np.int64)
    chi_vals = np.empty(len(batch), dtype=np.float64)

    X = np.empty((d, T_eff), dtype=flat_cdf.dtype)

    for b, combo in enumerate(batch):
        for i_d, pix in enumerate(combo):
            X[i_d, :] = flat_cdf[pix, starts[i_d]:ends[i_d]]

        chi_vals[b] = Multivar_Chi3(X.T, chi_q)
        targets[b] = combo[-1]

    return targets, chi_vals


