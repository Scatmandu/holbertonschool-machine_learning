#!/usr/bin/env python3
"""Optimization"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
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
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)
