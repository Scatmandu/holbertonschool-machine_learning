#!/usr/bin/env python3
"""creating the class Neural Network"""


import numpy as np


class NeuralNetwork:
    """initializing Neural Network"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """calculates forward propagation"""
        x = np.dot(self.W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-x))
        x = np.dot(self.W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-x))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates cost using logistic regression"""
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = -(1 / m) * (loss)
        return (cost)

    def evaluate(self, X, Y):
        """evaluate neural network predictions"""
        A1, pred = self.forward_prop(X)
        cost = self.cost(Y, pred)
        limit = np.where(pred >= 0.5, 1, 0)
        return (limit, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculate gradient descent for neural network"""
        m = X.shape[1]
        dz = (A2 - Y)
        dw = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        dz1 = np.matmul(self.__W2.T, dz) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - (alpha * dw)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b2 = self.__b2 - (alpha * db)
        self.__b1 = self.__b1 - (alpha * db1)

    @property
    def W1(self):
        """W1 setter"""
        return self.__W1

    @property
    def b1(self):
        """b1 setter"""
        return self.__b1

    @property
    def A1(self):
        """A1 setter"""
        return self.__A1

    @property
    def W2(self):
        """W2 setter"""
        return self.__W2

    @property
    def b2(self):
        """b2 setter"""
        return self.__b2

    @property
    def A2(self):
        """A2 setter"""
        return self.__A2
