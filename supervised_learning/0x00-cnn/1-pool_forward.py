#!/usr/bin/env python3
"""0x00-convlutional_neural_networks"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    method that that performs forward propagation over a
        pooling layer of a neural network

    A_prev = numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer

            m = the number of examples
            h_prev = the height of the previous layer
            w_prev = the width of the previous layer
            c_prev = the number of channels in the previous layer

    kernel_shape =  a tuple of (kh, kw) containing the size of the
        kernel for the pooling

        kh = the kernel height
        kw = the kernel width

    stride =  a tuple of (sh, sw) containing the strides
            for the convlution

            sh = the stride for the height
            sw = the stride for the width

    mode = a string containing either max or avg, indicating whether to perform
        maximum or average pooling, respectively

    Return: the output of the pooling layer
    """
    m, h, w, npc = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = (h - kh) // sh + 1
    pw = (w - kw) // sw + 1
    conv = np.zeros((m, ph, pw, npc))
    if mode == 'max':
        func = np.max
    if mode == 'avg':
        func = np.average
    for x in range(ph):
        for y in range(pw):
            conv[:, x, y, :] = func(
                A_prev[:, x*sh: x*sh + kh, y*sw: y*sw + kw, :],
                axis=(1, 2))
    return conv
