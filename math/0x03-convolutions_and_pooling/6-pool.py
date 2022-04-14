#!/usr/bin/env python3
"""0x03-convolutions_and_pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        method that performs a convlution on grayscale images

        images = numpy.ndarray with shape (m, h, w, c)
            containing multiple images

            m = the number of images
            h = the height in pixels of the images
            w = the width in pixels of the images
            c = number of channels in the image

        kernel_shape = tuple of (kh, kw) containing the
            kernel shape for the pooling

            kh = the height of the kernel
            kw = the width of the kernel

        stride = a tuple of (sh, sw)

            sh = the stride for the height of the image
            sw = the stride for the width of the image

        mode = indicates the type of pooling

            max = max pooling
            avg = average pooling

        Return: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    ch = int(((h - kh) / sh) + 1)
    cw = int(((w - kw) / sw) + 1)
    conv = np.zeros((m, ch, cw, c))
    for x in range(ch):
        j = x * sh
        for y in range(cw):
            k = y * sw
            output = images[:, j:j + kh, k:k + kw, :]
            if mode == 'max':
                conv[:, x, y, :] = np.max(output, axis=(1, 2))
            if mode == 'avg':
                conv[:, x, y, :] = np.average(output, axis=(1, 2))
    return conv
