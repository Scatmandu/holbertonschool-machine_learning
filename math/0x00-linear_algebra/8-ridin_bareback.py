#!/usr/bin/env python3
"""multiplies two matrices"""


def mat_mul(mat1, mat2):
    """multiplies two matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    else:
        new_list = []
        i = 0
        j = 0
        row = len(mat1)
        col = len(mat2[0])
        while len(new_list) < row:
            new_list.append([])
            while len(new_list[-1]) < col:
                new_list[-1].append(0)
        while i < len(new_list):
            while j < len(new_list[0]):
                new_list[i][j] = mat1[i][0] * mat2[0][j] + \
                 mat1[i][1] * mat2[1][j]
                j += 1
            i += 1
            j = 0
        return new_list
