#!/usr/bin/env python3
"""multiplies two matrices"""


def mat_mul(mat1, mat2):
    """multiplies two matrices"""

    if len(mat1) != len(mat2):
        return None
    elif len(mat1[0]) != len(mat2[0]):
        return None
    else:
        result = []
        for a, b in zip(mat1, mat2):
            current_list = []
            for x, y in zip(a, b):
                current_list.append(x * y)
            result.append(current_list)
        return result
