#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
        method that converts a label vector into a one-hot matrix

        labels = vector to convert
        classes = number of classes (default None)

        Return: the one-hot matrix
    """
    return (K.utils.to_categorical(labels, num_classes=classes))
