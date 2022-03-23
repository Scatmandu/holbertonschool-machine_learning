#!/usr/bin/env python3
"""optimization"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        method that updates a variable using the RMSProp optimization algorithm
        alpha = learning rate
        beta2 = RMSProp weight
        epsilon = small number to avoid division by zero
        var = numpy.ndarray containing the variable to be updated
        grad = numpy.ndarray containing the gradient of var
        s = the previous second moment of var
    """
    moment = beta2 * s + (1 - beta2) * (grad ** 2)
    update = var - (alpha * (grad / ((moment ** (0.5) + epsilon))))
    return update, moment
