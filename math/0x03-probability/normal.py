#!/usr/bin/env python3
"""creates class Normal"""


class Normal:
    """initializes class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                var = sum([((x - self.mean) ** 2) for x in data]) / len(data)
                result = var ** 0.5
                self.stddev = result

    def z_score(self, x):
        """returns z score for normal distribution"""
        zed = (x - self.mean) / self.stddev
        return zed

    def x_value(self, z):
        """returns x score for normal distribution"""
        xed = (z * self.stddev) + self.mean
        return xed

    def pdf(self, x):
        """returns pdf for normal distribution"""
        e = 2.7182818285
        pi = 3.1415926536
        std = self.stddev
        mean = self.mean
        left = (1 / (std * ((2 * pi) ** .5)))
        expo = (-.5 * ((x - mean) / std) ** 2)
        total = (left * e ** expo)
        return total

    def cdf(self, x):
        """returns cdf for normal distribution"""
        e = 2.7182818285
        pi = 3.1415926536
        std = self.stddev
        mean = self.mean
        bit0 = (2 / pi ** .5)
        bit1 = ((x ** 3) / 3)
        bit2 = ((x ** 5) / 10)
        bit3 = ((x ** 7) / 42)
        bit4 = ((x ** 9) / 216)
        inside = (x - mean) / (2 ** .5 * std)
        total = (0.5 * (1 + self.erf(inside)))
        return total

    def erf(self, x):
        """error func used in cdf"""
        pi = 3.1415926536
        bit0 = (2 / pi ** .5)
        bit1 = ((x ** 3) / 3)
        bit2 = ((x ** 5) / 10)
        bit3 = ((x ** 7) / 42)
        bit4 = ((x ** 9) / 216)
        return (bit0 * (x - bit1 + bit2 - bit3 + bit4))
