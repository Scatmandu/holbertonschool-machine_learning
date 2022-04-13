#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        method that tests a neural network

        netowrk = network model to test
        data = input data to test the model with
        labels = the correct one-hot labels of data
        verbose = boolean that determines if output should be
            printed during the testing process

        Return: the loss and accuracy of the model with
            the testing data, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
