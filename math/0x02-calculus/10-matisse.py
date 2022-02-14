#!/usr/bin/env python3
"""calculated the derivative of a polynomial"""


def poly_derivative(poly):
    """calculated the derivative of a polynomial"""
    new_list = []
    if len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        for i in range(1, len(poly)):
            new_list.append(poly[i] * i)
        return(new_list)
