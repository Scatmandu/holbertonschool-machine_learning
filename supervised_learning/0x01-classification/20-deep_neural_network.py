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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for x in range(self.L):
            if type(layers[x]) is not int or layers[x] < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(x + 1)
            b = "b{}".format(x + 1)
            self.weights[b] = np.zeros((layers[x], 1))
            self.weights[w] = np.random.randn(layers[x],
                                              prev) * np.sqrt(2 / prev)
            prev = layers[x]

    def forward_prop(self, X):
        """forward propagation using linear regression"""
        self.__cache["A0"] = X
        for x in range(self.L):
            W = self.__weights["W{}".format(x + 1)]
            b = self.__weights["b{}".format(x + 1)]
            y = np.matmul(W, self.cache["A{}".format(x)]) + b
            A = 1 / (1 + np.exp(-y))
            self.__cache["A{}".format(x + 1)] = A
        return (A, self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = -(1 / m) * (loss)
        return (cost)

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""

        A1, cache = self.forward_prop(X)
        cost = self.cost(Y, A1)
        A1 = np.where(A1 >= 0.5, 1, 0)
        return (A1, cost)

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights
