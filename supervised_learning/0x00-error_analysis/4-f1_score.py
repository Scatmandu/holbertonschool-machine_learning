#!/usr/bin/env python3
"""error_analysis"""
import numpy as np


def f1_score(confusion):
    """
        method that calculates the F1 score of a confusion matrix
        confusion = a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
                represent the predicted labels
        classes = the number of classes

        Returns: a numpy.ndarray of shape (classes,) containing
            the F1 score of each class
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    return (((precision * sensitivity) / (precision + sensitivity)) * 2)
