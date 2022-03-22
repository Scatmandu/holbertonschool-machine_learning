#!/usr/bin/env python3
"""optimization"""
import numpy as np


def shuffle_data(X, Y):
    """
       method that shuffles the data points in two matrices the same way
       X = ndarray of shape (m, nx) to shuffle
       Y = second ndarray of shape (m, ny) to shuffle
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
