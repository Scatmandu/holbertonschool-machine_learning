#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        method that makes prediction for a neural network

        network = network model to test
        data = input data to test the model with
        verbose = boolean that determines if output should be
            printed during the testing process

        Return: the loss and accuracy of the model
            with the testing data, respectively
    """
    return network.predict(data, verbose=verbose)
