#!/usr/bin/env python3
"""error_analysis"""
import numpy as np


def precision(confusion):
    """
        method that calculates the precision for each class
            in a confusion matrix
        confusion = a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
                represent the predicted labels
        classes = the number of classes

        Returns: a numpy.ndarray of shape (classes,) containing
            the precision of each class
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    TP = np.diag(confusion)
    return TP / (TP + FP)
