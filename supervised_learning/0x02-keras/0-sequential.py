#!/usr/bin/env python3
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    for x in range(len(layers)):
        if x == 0:
            model.add(K.layers.Dense(units=layers[x],
                                     activation=activations[x],
                                     kernel_regularizer=reg,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[x],
                                     activation=activations[x],
                                     kernel_regularizer=reg))
        if x + 1 != len(layers):
            model.add(K.layers.Dropout(keep_prob, noise_shape=None, seed=None))
    return model
