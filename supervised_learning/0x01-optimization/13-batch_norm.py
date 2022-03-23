#!/usr/bin/env python3
"""Optimization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        method  that normalizes an unactivated output of a neural
            network using batch normalization
        Z = array of shape (m, n) to be normalized
            m = number of data points
            n = number of features in Z
        gamma = numpy.ndarray of shape (1, n) containing the
            scales used for batch normalization
        beta = numpy.ndarray of shape (1, n) containing the
            offsets used for batch normalization
        epsilon = small number used to avoid division by zero
    """
    z_norm = ((Z - np.average(Z, axis=0)) /
              (((np.var(Z, axis=0) + epsilon) ** 0.5)))
    return (gamma * z_norm) + beta
