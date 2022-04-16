#!/usr/bin/env python3
"""0x00-convolutional_neural_networks"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        method that performs forward propagation over a
            convolutional layer of a neural network

        A_prev = numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer

            m = the number of examples
            h_prev = the height of the previous layer
            w_prev = the width of the previous layer
            c_prev = the number of channels in the previous layer

        W = a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution

            kh = the filter height
            kw = the filter width
            c_prev = the number of channels in the previous layer
            c_new = the number of channels in the output

        b = a numpy.ndarray of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution

        activation = an activation function applied to the convolution
        padding = a string that is either same or valid, indicating the
            type of padding used

        stride =  a tuple of (sh, sw) containing the strides
            for the convolution

            sh = the stride for the height
            sw = the stride for the width

        Return: the output of the convolutional layer
    """
    m, h, w, c = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) / 2) + 1
        pw = (((w - 1) * sw + kw - w) / 2) + 1
    pad_m = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant', constant_values=0)
    ch = (h + (2 * ph) - kh) // sh + 1
    cw = (w + (2 * pw) - kw) // sw + 1
    conv = np.zeros((m, ch, cw, nc))
    for x in range(nc):
        for y in range(ch):
            for z in range(cw):
                conv[:, y, z, x] = np.sum(np.multiply(
                    W[:, :, :, x],
                    pad_m[:, sh*y: sh*y + kh, sw*z: sw*z + kw]),
                    axis=(1, 2, 3)) + b[:, :, :, x]
    return activation(conv)
