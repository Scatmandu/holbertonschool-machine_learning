#!/usr/bin/env python3
"""calculates neural network accuracy with tensorflow"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates neural network accuracy"""
    yMax = tf.argmax(y, axis=1)
    predMax = tf.argmax(y_pred, axis=1)
    equal = tf.equal(predMax, yMax)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
