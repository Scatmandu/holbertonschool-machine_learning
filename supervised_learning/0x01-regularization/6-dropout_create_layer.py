#!/usr/bin/env python3
"""Regularization"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
        method that creates a layer of a neural network using dropout.
        prev = a tensor containing the output of the previous layer.
        n = the number of nodes the new layer should contain.
        activation = the activation function that
            should be used on the layer.
        keep_prob = the probability that a node will be kept.

        Returns: the output of the new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation, name="layer",
                            kernel_initializer=init, kernel_regularizer=reg)
    return layer(prev)
