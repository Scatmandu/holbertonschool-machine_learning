#!/usr/bin/env python3
"""error_analysis"""
import numpy as np


def sensitivity(confusion):
    """
        method that calculates the sensitivity for each class
            in a confusion matrix
        confusion = a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
                represent the predicted labels
        classes = the number of classes

        Returns: a numpy.ndarray of shape (classes,) containing
            the sensitivity of each class
    """
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    return TP / (TP + FN)
