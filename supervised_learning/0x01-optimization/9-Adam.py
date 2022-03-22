#!/usr/bin/env python3
"""Optimization"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        method that updates a variable in place using
            the Adam optimization algorithm
        alpha = learning rate
        beta1 = weight used for first moment
        beta2 = weight used for second moment
        epsilon = small number to avoid division by 0
        var = numpy.ndarray containing the variable to be updated
        grad = numpy.ndarray containing the gradient of var
        v = the previous first moment of var
        s = the previous second moment of var
        t = the time step used for bias correction
    """
    moment1 = (beta1 * v) + ((1 - beta1) * grad)
    new_first = moment1 / (1 - (beta1 ** t))
    moment2 = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    new_second = moment2 / (1 - beta2 ** t)
    update = var - (alpha * (c_moment1 / ((c_moment2 + epsilon) ** (0.5))))
    return update, new_first, new_second
