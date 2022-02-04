#!/usr/bin/env python3
"""multiplies two matrices using numpy"""
import numpy as np


def mat_mul(mat1, mat2):
    """multiplies two matrices using numpy"""
    product = np.dot(mat1, mat2)
    x = product.tolist()
    return x
