#!/usr/bin/env python3
"""creating class Exponential"""


class Exponential:
    """initializing class Exponential"""
    def __init__(self, data=None, lambtha=1.):
        if not data:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            length = len(data)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif length < 2:
                raise ValueError("data must contain multiple values")
            else:
                total = 0
                for x in data:
                    total = x + total
                self.lambtha = float(1 / (total / length))

    def pdf(self, x):
        """calculates pdf of exponential distribution"""
        neg = self.lambtha * -1
        pdf = self.lambtha * (e ** (neg * x))
        return pdf

    def cdf(self, x):
        """calculates cdf of exponential distribution"""
        e = 2.7182818285
        neg = self.lambtha * -1
        cdf = 1 - e ** (neg * x)
        return cdf
