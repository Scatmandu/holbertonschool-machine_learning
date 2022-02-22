#!/usr/bin/env python3
"""creating class Binomial"""


class Binomial:
    """initializing class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total = sum(data)
            length = len(data)
            mean = total / length
            var = sum([((x - mean) ** 2) for x in data]) / len(data)
            q = var / mean
            p = 1 - q
            n = round(mean / p)
            true_p = mean / n
            self.n = int(n)
            self.p = float(true_p)

    def pmf(self, k):
        """calculates pmf for binomial distribution"""
        if k < 0 or k > self.n:
            return 0
        if type(k) is not int:
            k = int(k)
        n = self.n
        p = self.p
        q = 1 - p
        n_fact = self.fact(n)
        k_fact = self.fact(k)
        n_sub_k_fact = self.fact(n - k)
        left = (n_fact / (k_fact * n_sub_k_fact))
        right = ((p ** k) * (q ** (n - k)))
        return left * right

    def cdf(self, k):
        """calculates cdf for binomial distribution"""
        sum_list = []
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        for x in range(k + 1):
            sum_list.append(self.pmf(x))
        return sum(sum_list)

    def fact(self, n):
        """returns factorial"""
        fact = 1
        for num in range(1, n + 1):
            fact *= num
        return fact
