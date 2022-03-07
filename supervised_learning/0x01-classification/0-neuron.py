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
        self.W = np.random.normal(size=(1, self.nx))
        self.b = 0
        self.A = 0
