#!/usr/bin/env python3
"""optimization"""
import numpy as np


def create_momentum_op(loss, alpha, beta1):
    """
        creates the training operation for a neural network in tensorflow
            using the gradient descent with momentum optimization algorithm
        loss = the loss of the network
        alpha = the learning rate
        beta1 = the momentum weight
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
