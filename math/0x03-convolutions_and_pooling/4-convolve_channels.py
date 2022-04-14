#!/usr/bin/env python3
"""0x03-convolutions_and_pooling"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        method that performs a convlution on grayscale images

        images = numpy.ndarray with shape (m, h, w) containing multiple
            grayscale images

            m = the number of images
            h = the height in pixels of the images
            w = the width in pixels of the images

        kernel = numpy.ndarray with shape (kh, kw)
            containing the kernel for the convlution

            kh = the height of the kernel
            kw = the width of the kernel

        padding = either a tuple of (ph, pw), same, or valid

            if tuple =
                ph = the padding for the height of the image
                pw = the padding for the width of the image
            if same = performs the same convlution
            if valid = performs a valid convlution

        stride = a tuple of (sh, sw)

            sh = the stride for the height of the image
            sw = the stride for the width of the image

        Return: a numpy.ndarray containing the convlved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if type(padding) is tuple:
        ph, pw = padding
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    pad_total = ((0, 0), (pw, pw), (ph, ph), (0, 0))
    pad_m = np.pad(images, pad_width=pad_total, mode='constant',
                   constant_values=0)
    ch = int((pad_m.shape[1] - kh) / sh + 1)
    cw = int((pad_m.shape[2] - kw) / sw + 1)
    conv = np.zeros((m, ch, cw))
    for x in range(ch):
        j = 0
        for y in range(cw):
            conv[:, x, y] = (kernel * pad_m[:,
                             y * sh: y * sh + kh,
                             x * sw: x * sw + kw,
                             :]).sum(axis=(1, 2, 3))
    return conv
