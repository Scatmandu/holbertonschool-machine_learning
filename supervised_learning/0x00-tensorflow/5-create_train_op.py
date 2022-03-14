#!/usr/bin/env python3
"""trains neural network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """trains neural network"""
    train = tf.train.GradientDescentOptimizer(alpha)
    return train.minimize(loss)
