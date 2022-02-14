#!/usr/bin/env python3
"""calculates the sigma of i**2 with stopping condition n"""


def summation_i_squared(n):
    """calculates the sigma of i**2 with stopping condition n"""
    if n == 1:
        return 1
    elif n <= 0 or type(n) is not int:
        return None
    else:
        return (n**2 + summation_i_squared(n - 1))
