#!/usr/bin/env python3
"""optimization"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        method that creates a learning rate decay operation in tensorflow
            using inverse time decay
        alpha = the original learning rate
        decay_rate = weight used to determine the rate at which alpha decays
        global_step = number of passes of gradient descent that have elapsed
        decay_step = number of passes of gradient descent that should
            occur before alpha is decayed further
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
