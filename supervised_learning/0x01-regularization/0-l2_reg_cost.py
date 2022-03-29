#!/usr/bin/python3
import numpy as np
"""Regularization"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        method calculates the cost of a neural network with L2 regularization
        cost = cost of the network without L2 regularization
        lambtha = regularization parameter
        weights = dictionary of the weights and biases
        L = number of layers in the neural network
        m = number of data points used

        Returns: cost of the network accounting for L2 regularization
    """
    norm = 0
    for layer in range(1, L + 1):
        norm += np.linalg.norm(weights.get('W' + str(layer)))
    return (cost + (norm * (lambtha / (2 * m))))