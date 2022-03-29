#!/usr/bin/env python3
"""Regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
        cost = tensor containing the cost of the network
            without L2 regularization

        Return: tensor containing the cost of the network accounting
            for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
