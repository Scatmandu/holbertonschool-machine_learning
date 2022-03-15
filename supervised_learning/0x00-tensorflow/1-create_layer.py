#!/usr/bin/env python3
"""returns tensor output of a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """returns tensor output of a layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(n, activation, kernel_initializer=w, name="layer")
    return lay(prev)
