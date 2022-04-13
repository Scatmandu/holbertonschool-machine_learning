#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """
        method that saves model

        network = model to save
        filename = filename to output

        Return: nothing
    """
    network.save(filename)


def load_model(filename):
    """
        method that loads model

        filename = filename to load

        Return: nothing
    """
    return K.models.load_model(filepath=filename)
