#!/usr/bin/env python3


def matrix_shape(matrix):
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
