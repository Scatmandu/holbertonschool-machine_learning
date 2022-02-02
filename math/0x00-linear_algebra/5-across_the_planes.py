#!/usr/bin/env python3

import numpy as np


def add_matrices2D(mat1, mat2):

    if np.shape(mat1) != np.shape(mat2):
        return None
    else:
        vector1 = np.array(mat1)
        vector2 = np.array(mat2)
        sum_vector = vector1 + vector2
        x = sum_vector.tolist()
        return x
