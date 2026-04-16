
import numpy as np

def max_pooling_2d(X, pool_size, stride):
    # Get the dimensions of the input array
    n_h, n_w = X.shape

    # Calculate the dimensions of the output array
    out_h = (n_h - pool_size) // stride + 1
    out_w = (n_w - pool_size) // stride + 1

    # Initialize the output array
    pooled_array = np.zeros((out_h, out_w))

    # Perform max pooling
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            pooled_array[i, j] = np.max(X[h_start:h_end, w_start:w_end])
    
    return pooled_array

def pooling_2d(X, pool_size, stride , func):
    # Get the dimensions of the input array
    n_h, n_w = X.shape

    # Calculate the dimensions of the output array
    out_h = (n_h - pool_size) // stride + 1
    out_w = (n_w - pool_size) // stride + 1

    # Initialize the output array
    pooled_array = np.zeros((out_h, out_w))

    # Perform max pooling
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            pooled_array[i, j] = func(X[h_start:h_end, w_start:w_end])
    
    return pooled_array


def pooling_1d(X, pool_size, stride , func):
    # Get the length of the input array
    n = len(X)

    # Calculate the length of the output array
    out_length = (n - pool_size) // stride + 1

    # Initialize the output array
    pooled_array = np.zeros(out_length)

    # Perform max pooling
    for i in range(out_length):
        start = i * stride
        end = start + pool_size
        pooled_array[i] = func(X[start:end])
    
    return pooled_array
