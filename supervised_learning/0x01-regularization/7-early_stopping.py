#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        method  that determines if you should stop gradient descent early
        cost = current validation cost of the neural network
        opt_cost = the lowest recorded validation cost of the neural network
        threshold = the threshold used for early stopping
        patience = the patience count used for early stopping
        count = the count of how long the threshold has not been met

        Return: a boolean of whether the network should be stopped early
            followed by the updated count
    """
    boole = False
    if cost >= opt_cost - threshold:
        count += 1
        if count == patience:
            return True, count
    else:
        boole = False
        count = 0
    return boole, count
