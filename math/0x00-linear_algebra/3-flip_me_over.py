#!/usr/bin/env python3
"""transposes a matrix"""


def matrix_transpose(matrix):
    """transposes a matrix"""
    new_matrix = []
    rows = len(matrix)
    columns = len(matrix[0])
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        new_matrix.append(row)
    return new_matrix
