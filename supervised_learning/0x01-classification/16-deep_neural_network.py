#!/usr/bin/env python3
"""creating the class DeepNeuralNetwork"""


import numpy as np


class DeepNeuralNetwork:
    """initializes DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for x in range(self.L):
            if type(layers[x]) is not int or layers[x] < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(x)
            b = "b{}".format(x)
            layers = l
            self.weights[b] = np.zeros((layers[x], 1))
            self.weights[w] = np.random.randn(l[x],
                                              l[x - 1]) * np.sqrt(2 / l[x - 1])
