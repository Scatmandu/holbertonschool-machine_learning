#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """
        method that saves a config

        network = model whose configuration should be saved
        filename = file to save config as

        Return: nothing
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
        method that loads a config

        filename = file to load

        Return: nothing
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
