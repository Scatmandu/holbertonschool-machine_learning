#!/usr/bin/env python3
"""creating the class Neuron"""


import numpy as np


class Neuron:
    """instantiating Neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise ValueError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, self.nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W setter"""
        return self.__W

    @property
    def b(self):
        """b setter"""
        return self.__b

    @property
    def A(self):
        """A setter"""
        return self.__A
