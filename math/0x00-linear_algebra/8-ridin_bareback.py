#!/usr/bin/env python3

import numpy as np


def mat_mul(mat1, mat2):

    product = np.dot(mat1, mat2)
    x = product.tolist()
    return x
