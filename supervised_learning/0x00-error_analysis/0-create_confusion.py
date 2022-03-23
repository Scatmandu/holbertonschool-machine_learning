#!/usr/bin/env python3
"""error_analysis"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
        method that creates a confusion matrix
        labels = one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels for each data point
        logits = one-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels
        m = the number of data points
        classes = the number of classes

        Returns:  a confusion numpy.ndarray of shape (classes, classes) with
            row indices representing the correct labels and column indices
                representing the predicted labels
    """
    return np.matmul(labels.T, logits)
