#!/usr/bin/env python3
"""optimization"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        method that updates a variable using the RMSProp optimization algorithm
    """
    moment = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    update = var - (alpha * (grad / ((moment ** (0.5) + epsilon))))
    return updated, moment
