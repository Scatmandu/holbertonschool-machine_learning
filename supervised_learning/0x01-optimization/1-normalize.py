#!/usr/bin/env python3
"""optimization"""
import numpy as np


def normalize(X, m, s):
    """
        method to normalize a matrix
        X = ndarray, shape (d, nx) to normalize
        m = ndarray, shape (nx,) that contains the mean of all features of X
        s = ndarray, shape (nx,) that contains the std dev of all features of X
    """
    return (X - m) / (s)
