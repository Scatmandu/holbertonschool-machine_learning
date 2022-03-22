#!/usr/bin/env python3
"""optimization"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        method that creates the training operation for a neural network in
            tensorflow using the RMSProp optimization algorithm
        loss = loss of the network
        alpha = the learning rate
        beta2 = the RMSProp weight
        epsilon = small number to avoid division by 0
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)
