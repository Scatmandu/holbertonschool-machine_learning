#!/usr/bin/env python3
"""creating class Exponential"""


class Exponential:
    """initializing class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                length = float(len(data))
                total = float(sum(data))
                self.lambtha = 1 / (total / length)

    def pdf(self, x):
        """calculates pdf of exponential distribution"""
        if x < 0:
            return 0
        e = 2.7182818285
        lmb = self.lambtha
        neg = lmb * -1
        pdf = lmb * (e ** (neg * x))
        return pdf

    def cdf(self, x):
        """calculates cdf of exponential distribution"""
        if x < 0:
            return 0
        e = 2.7182818285
        lmb = self.lambtha
        neg = lmb * -1
        cdf = 1 - e ** (neg * x)
        return cdf
