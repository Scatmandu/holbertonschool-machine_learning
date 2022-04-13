#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
        method that saves weights

        network =  model whose weights should be saved
        filename = path of the file that the weights should be saved to
        save_format = the format in which the weights should be saved

        Return: nothing
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """
        method that loads weights

        network =  model whose weights should be saved
        filename = path of the file that the weights should be saved to

        Return: nothing
    """
    return network.load_weights(filepath=filename)
