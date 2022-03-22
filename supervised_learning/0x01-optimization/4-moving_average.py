#!/usr/bin/env python3
"""optimization"""
import numpy as np


def moving_average(data, beta):
    """
       method that calculates the weighted moving average of a data set
       data = list of data to calculate the moving average of
       beta = weight used for the moving average
    """
    x = 0
    new_list = []
    for i in range(len(data)):
        x = (beta * x) + ((1 - beta) * data[i])
        new_list.append(x / (1 - (beta ** (i + 1))))
    return new_list
