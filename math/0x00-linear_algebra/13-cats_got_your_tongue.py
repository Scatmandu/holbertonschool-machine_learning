#!/usr/bin/env python3
"""concatenates two matrices using numpy"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices using numpy"""
    return np.concatenate((mat1, mat2), axis)
