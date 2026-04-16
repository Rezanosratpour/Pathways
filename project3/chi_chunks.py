from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from itertools import product, islice

import tifffile as tff
import os
import numpy as np
from itertools import product, islice
from joblib import Parallel, delayed
from extremals import Multivar_Chi3

def _chunked_product(lists, start=0, chunk_size=2000):
    """
    Yield chunks from product(*lists), starting from index `start`.
    """
    iterator = islice(product(*lists), start, None)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def _chi_stats_chunk(
            combo_chunk,
            d,
            cdf_data,
            I_start_time,
            I_end_time,
            start_offsets,
            end_offsets,
            Col,
            n_keys,
            percentile,
                        ):
    sums = np.zeros(n_keys, dtype=np.float64)
    sums2 = np.zeros(n_keys, dtype=np.float64)
    counts = np.zeros(n_keys, dtype=np.int64)

    for comb in combo_chunk:
        X_parts = []

        for i_d in range(d):
            ii, jj = int(comb[i_d][0]), int(comb[i_d][1])
            index = ii * Col + jj

            start = I_start_time + start_offsets[i_d]
            stop = I_end_time - end_offsets[i_d]

            X_parts.append(cdf_data[index, start:stop])

        X = np.asarray(X_parts).T
        chi_val = Multivar_Chi3(X, percentile)

        # same logic as your original code: assign to last coordinate
        last_ii, last_jj = int(comb[d - 1][0]), int(comb[d - 1][1])
        pos = last_ii * Col + last_jj

        if np.isfinite(chi_val):
            sums[pos] += chi_val
            sums2[pos] += chi_val * chi_val
            counts[pos] += 1

    return sums, sums2, counts


def _compute_chi_block(i_block, V, data_left, data_right, percentile):
    """
    Worker function executed in parallel.
    Computes chi values for a block of source indices i_block against all V.
    """
    block = np.empty((len(i_block), len(V)), dtype=np.float32)

    for r, indx_i in enumerate(i_block):
        x1 = data_left[indx_i]   # shape: (T_lagged,)

        for c, indx_j in enumerate(V):
            x2 = data_right[indx_j]  # shape: (T_lagged,)
            X = np.column_stack((x1, x2))
            block[r, c] = Multivar_Chi3(X, percentile)

    return np.asarray(i_block), block


def _chunk_array(arr, chunk_size):
    """Split a 1D array into chunks."""
    for k in range(0, len(arr), chunk_size):
        yield arr[k:k + chunk_size]


# --- multiprocessing helpers (top-level so they are picklable) ---
_CHI_CTX = {}

def _init_chi_ctx(cdf_data, Col, I_start_time, I_end_time, start_off, end_off, percentile):
    """Initializer: runs once in each worker process."""
    global _CHI_CTX
    _CHI_CTX = {
        "cdf_data": cdf_data,
        "Col": Col,
        "I_start_time": I_start_time,
        "I_end_time": I_end_time,
        "start_off": start_off,
        "end_off": end_off,
        "percentile": percentile,
    }

def _chi_worker(combo):
    """Compute chi for a single combination of d points."""
    ctx = _CHI_CTX
    cdf_data = ctx["cdf_data"]
    Col = ctx["Col"]
    I_start_time = ctx["I_start_time"]
    I_end_time = ctx["I_end_time"]
    start_off = ctx["start_off"]
    end_off = ctx["end_off"]
    percentile = ctx["percentile"]

    d = len(combo)
    X_parts = []
    for k in range(d):
        ii, jj = combo[k]          # combo[k] is (ii, jj) or array([ii, jj])
        idx = ii * Col + jj

        s = I_start_time + start_off[k]
        e = I_end_time   - end_off[k]
        X_parts.append(cdf_data[idx, s:e])

    # shape (T, d) -> matches your original Multivar_Chi3(X.T, ...)
    X = np.stack(X_parts, axis=1)
    chi = Multivar_Chi3(X, percentile)

    # keep the same key behaviour as your original code: last (ii, jj)
    ii_last, jj_last = combo[-1]
    key = str(np.array([ii_last, jj_last]))
    return key, float(chi), combo

def _coord_key(ii, jj):
    return (int(ii), int(jj))

def _chunked_product(lists, start=0, chunk_size=2000):
    """
    Lazily generate chunks from product(*lists), skipping the first `start` items.
    """
    it = islice(product(*lists), start, None)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk




def _reduce_stats(stats_list, n_keys):
    total_sum = np.zeros(n_keys, dtype=np.float64)
    total_sum2 = np.zeros(n_keys, dtype=np.float64)
    total_count = np.zeros(n_keys, dtype=np.int64)

    for s, s2, c in stats_list:
        total_sum += s
        total_sum2 += s2
        total_count += c

    return total_sum, total_sum2, total_count



def _coords_to_flat(coords, ncols):
    coords = np.asarray(coords)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return coords[:, 0].astype(np.int64) * ncols + coords[:, 1].astype(np.int64)


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




