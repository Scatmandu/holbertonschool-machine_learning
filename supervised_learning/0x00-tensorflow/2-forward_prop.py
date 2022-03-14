#!/usr/bin/env python3
"""forward prop using tensorflow"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward prop using tensorflow"""
    prev = x
    for i in range(len(layer_sizes)):
        lay = create_layer(prev, layer_sizes[i], activations[i])
        prev = lay(prev)
    return prev
