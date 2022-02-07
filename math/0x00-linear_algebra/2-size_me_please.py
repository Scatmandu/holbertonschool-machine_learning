#!/usr/bin/env python3
"""calculates the shape of two matrices"""


def matrix_shape(matrix):
    """calculates the shape of two matrices"""
    new_list = []

    if type(matrix[0]) is int:
        row = len(matrix)
        new_list.append(row)
        return(new_list)

    elif type(matrix[0][0]) is int:
        row = len(matrix)
        column = len(matrix[0])
        new_list.append(row)
        new_list.append(column)
        return new_list
    else:
        row = len(matrix)
        column = len(matrix[0])
        length = len(matrix[0][0])
        new_list.append(row)
        new_list.append(column)
        new_list.append(length)
        return new_list
