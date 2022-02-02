#!/usr/bin/env python3

import numpy as np


def matrix_transpose(matrix):
    trans_mat = np.transpose(matrix)
    x = trans_mat.tolist()
    return x
