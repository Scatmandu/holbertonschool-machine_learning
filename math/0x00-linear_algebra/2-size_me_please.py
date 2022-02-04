#!/usr/bin/env python3
"""calculates the shape of two matrices"""


def matrix_shape(matrix):
    """calculates the shape of two matrices"""
    new_list = []
    row = len(matrix)
    column = len(matrix[0])
    if type(matrix[0][0]) is int:
        new_list.append(row)
        new_list.append(column)
        return new_list
    else:
        length = len(matrix[0][0])
        new_list.append(row)
        new_list.append(column)
        new_list.append(length)
    return new_list
