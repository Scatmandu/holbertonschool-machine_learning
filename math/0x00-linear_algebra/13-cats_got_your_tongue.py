#!/usr/bin/env python3
"""concatenates two matrices using numpy"""
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices using numpy"""
    new_list = []
    a = np.array(mat1)
    b = np.array(mat2)
    new_list = np.concatenate((a, b), axis)
    x = new_list.tolist()
    return x