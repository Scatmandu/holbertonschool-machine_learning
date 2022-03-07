#!/usr/bin/env python3
"""creating the class Neuron"""


import numpy as np


class Neuron:
    """instantiating Neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, self.nx))
        self.__b = 0

    def forward_prop(self, X):
        """calculate forward propagagtion"""
        activation = np.matmul(self.__W, X) + self.__b
        self.__A = (1 / (1 + np.exp(-activation)))
        return self.__A

    def cost(self, Y, A):
        """calculates cost"""
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        return -(1 / m) * loss

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A
