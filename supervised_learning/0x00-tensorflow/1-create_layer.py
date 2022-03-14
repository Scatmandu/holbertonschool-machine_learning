#!/usr/bin/env python3
"""returns tensor output of a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """returns tensor output of a layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,  name="layer")
    return layer(prev)
