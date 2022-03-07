#!/usr/bin/env python3
"""creating the class Neuron"""


import matplotlib.pyplot as plt
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
        """calculates forward propagation"""
        activation = np.matmul(self.__W, X) + self.__b
        self.__A = (1 / (1 + np.exp(-activation)))
        return self.__A

    def cost(self, Y, A):
        """evaluates cost"""
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        return -(1 / m) * loss

    def evaluate(self, X, Y):
        """evaluates a neuron's predictions"""
        pred = self.forward_prop(X)
        cost = self.cost(Y, pred)
        limit = np.where(pred >= 0.5, 1, 0)
        return limit, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent"""
        m = Y.shape[1]
        dz = np.subtract(A, Y)
        dW = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dW.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        new_list = []
        for x in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if x % step == 0 or x == 0:
                cost = self.cost(Y, A)
                new_list.append(cost)
                if verbose is True:
                    if type(step) is not int:
                        raise TypeError("step must be an integer")
                    if step <= 0 or step > iterations:
                        raise
                        TypeError("step must be positive and <= iterations")
                    print("Cost after {} iterations: {}".format(x, cost))
        if graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise TypeError("step must be positive and <= iterations")
            plt.plot(x, y, "b-")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
            plt.savefig('0.png')
        return self.evaluate(X, Y)

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

    @A.setter
    def A(self, A):
        self.__A = A
