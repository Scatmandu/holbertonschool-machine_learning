#!/usr/bin/env python3
"""calculates loss using tensorflow"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates loss of neural network"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
