#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        method  that updates the weights and biases of a neural network using
            gradient descent with L2 regularization
        Y = one-hot numpy.ndarray of shape (classes, m) that contains
                the correct labels for the data
                    classes = the number of classes
                    m = the number of data points
        weights = a dictionary of the weights and biases of the neural network
        cache = a dictionary of the outputs of each layer of the neural network
        alpha = the learning rate
        lambtha = the L2 regularization parameter
        L = the number of layers of the network
    """
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        a = cache["A{}".format(layer)]
        if layer == L:
            dz = (cache["A{}".format(layer)] - Y)
        else:
            dz = da * (1 - np.square(a))

        l2 = ((lambtha/m) * weights["W{}".format(layer)])
        dw = (np.matmul(dz, cache["A{}".format(layer-1)].T) / m) + l2
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(weights["W{}".format(layer)].T, dz)
        weight = weights["W{}".format(layer)] - (alpha * dw)
        bias = weights["b{}".format(layer)] - (alpha * db)
        weights["W{}".format(layer)] = weight
        weights["b{}".format(layer)] = bias
