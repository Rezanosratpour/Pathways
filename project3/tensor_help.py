

import numpy as np 


import sys
sys.path = [str(p) for p in sys.path]

import tensorflow as tf
from tensorflow.data import AUTOTUNE


def to_ds(X, y, training=True, BATCH = 32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

# Rebuild pairs if needed
def make_pairs(frames, seq):
    X, y = [], []
    for i in range(len(frames) - seq):
        X.append(frames[i:i+seq])
        y.append(frames[i+seq])
    return np.asarray(X , dtype = np.float32), np.asarray(y, dtype = np.float32)
