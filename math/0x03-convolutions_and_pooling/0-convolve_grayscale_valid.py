#!/usr/bin/env python3
"""0x03-convolutions_and_pooling"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        method that performs a valid convolution on grayscale images

        images = numpy.ndarray with shape (m, h, w) containing
            multiple grayscale images

            m = number of images
            h = height in pixels of images
            w = width in pixels of images

        kernel = numpy.ndarray with shape (kh, kw) containing the
            kernel for the convolution

            kh = kernel height
            kw = kernel width

        Returns: numpy.ndarray containing the convolved images
    """

    h = images.shape[1]
    w = images.shape[2]
    m = images.shape[0]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    conv = np.zeros((m, h - kh + 1, w - kw + 1))
    for x in range(conv.shape[1]):
        for y in range(conv.shape[2]):
            output = np.sum(images[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            conv[:, x, y] = output
    return conv
