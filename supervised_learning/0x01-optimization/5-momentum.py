#!/usr/bin/env python3
"""optimization"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, V):
    """
        method  that updates a variable using the gradient descent with
            momentum optimization algorithm
        alpha = learning rate
        beta1 = momentum weight
        var = ndarray containing the variable to be updated
        grad = ndarray containing the gradient of var
        V = the previous first moment of var
    """
    moment = (beta1 * V) + ((1 - beta1) * grad)
    update = var - (alpha * moment)
    return update, moment
