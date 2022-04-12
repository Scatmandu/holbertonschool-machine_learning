#!/usr/bin/env python3
"""0x02-keras"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
     method for Adam optimization for a keras model with categorical
        crossentropy loss and accuracy metrics

    network = the model to optimize
    alpha = the learning rate
    beta1 = the first Adam optimization parameter
    beta2 = the second Adam optimization parameter

    Return: None
    """
    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    return None
