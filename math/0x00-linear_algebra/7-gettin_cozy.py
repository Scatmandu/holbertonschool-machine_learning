#!/usr/bin/env python3
"""concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            new_list = []
            for row in mat1:
                new_list.append(row.copy())
            for i in range(len(mat2)):
                new_list.append(mat2[i].copy())
            return new_list
    else:
        if len(mat1) != len(mat2):
            return None
        else:
            new_list = []
            for i in range(len(mat1)):
                new_list.append(mat1[i].copy() + mat2[i].copy())
            return new_list