#!/usr/bin/env python3
"""Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
        method that creates a tensorflow layer that includes L2
            regularization.

        prev =  a tensor containing the output of the previous layer.
        n =  the number of nodes the new layer should contain.
        activation = the activation function that should be used on the
            layer.
        lambtha = the L2 regularization parameter.

        Returns: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(units=n, activation=activation, name="layer",
                            kernel_initializer=init, kernel_regularizer=reg)
    return layer(prev)
