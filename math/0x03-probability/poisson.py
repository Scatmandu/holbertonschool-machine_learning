#!/usr/bin/env python3
"""creating class Poisson"""


class Poisson:
    """initializing class Poisson"""
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
                self.lambtha = total / length

    def pmf(self, k):
        """returns pmf of Poisson distribution"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        e = 2.7182818285
        lam = self.lambtha
        denominator = self.fact(k)
        return (lam ** k * e ** -lam / denominator)

    def cdf(self, k):
        """returns cdf of Poisson distribution"""
        e = 2.7182818285
        cdf_store = []
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(key)
        lam = self.lambtha
        for i in range(k + 1):
            cdf_numerator = (e ** (-lam) * (lam ** i))
            cdf_denominator = 1
            for x in range(1, i + 1):
                cdf_denominator = self.fact(x)
            cdf_store.append(cdf_numerator / cdf_denominator)
        return sum(cdf_store)

    def fact(self, n):
        """returns factorial"""
        fact = 1
        for num in range(1, n + 1):
            fact *= num
        return fact
