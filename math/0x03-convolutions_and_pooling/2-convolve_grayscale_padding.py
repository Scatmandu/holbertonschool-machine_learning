#!/usr/bin/env python3
"""0x03-convlutions_and_pooling"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        method that performs a valid convlution on grayscale images

        images = numpy.ndarray with shape (m, h, w) containing
            multiple grayscale images

            m = number of images
            h = height in pixels of images
            w = width in pixels of images

        kernel = numpy.ndarray with shape (kh, kw) containing the
            kernel for the convlution

            kh = kernel height
            kw = kernel width

        padding = a tuple of (ph, pw)

            ph = the padding for the height of the image
            pw = the padding for the width of the image

        Returns: numpy.ndarray containing the convlved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    pad_total = ((0, 0), (ph, ph), (pw, pw))
    pad_m = np.pad(images, pad_width=pad_total, mode='constant')
    conv = np.zeros((m, h + (2 * ph) - kh + 1, w + (2 * pw) - kw + 1))
    for x in range(conv.shape[1]):
        for y in range(conv.shape[2]):
            output = np.sum(pad_m[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            conv[:, x, y] = output
    return conv
