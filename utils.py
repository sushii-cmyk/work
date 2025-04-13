import functools
from functools import partial
from logging import Logger

import numpy as np


def arrayed(*inds, kwds=None):
    if kwds is None:
        kwds = []

    def wrapper(f):
        def wrapped(*args, **kwargs):
            brgs = array(array(a) if i in inds else a
                         for i, a in enumerate(args))
            bwargs = {k: array(v) if k in kwds else v
                      for (k, v), i in enumerate(kwargs.items())}

            return f(*brgs, **bwargs)

        return wrapped

    return wrapper


i = complex(0, 1)
e = np.e
pi = np.pi
tau = 2 * np.pi


def uint(n, a = 0, b = 1, *, exc=True, inc=True):
    """
    gives n points in [a, b];
    default n pts in the Unit INTerval
    """

    # teeny tiny epsilon diff for drawing
    eps = 0.0001
    r = np.unique(np.linspace(a + eps, b - eps, round(n), exc)[not inc:])
    return r.reshape((1, round(n)))


def unit(n, a = 0, b = 1, *, exc=True, inc=True):
    """
    n points in the UNIT circle (C)
    """

    return exp2ni(uint(n, a, b, exc=exc, inc=inc))


def exp2ni(t):
    """n kinda looks liek pi.... good enough.

    e^0pi

    :param t:
    :type t:
    :return:
    :rtype:
    """
    return np.exp(tau * i * t)


class logger:
    active = False

    def __init__(self, p=">"):
        self.i = 0

        self.b = True
        self.p = p  # prefix

    def set(self, b):
        self.b = b  # (in)active bool

        return self

    def __getitem__(self, *m):
        if self.b and logger.active:
            self.i += 1
            # print(self.p, self.i, *m)


class vunc:
    def __init__(self, f):
        self.f = f

    def __getitem__(self, x):
        if isinstance(x, complex):
            # print(x)
            return complex(self.f(x.real), self.f(x.imag))



        else:
            # print(x)
            return np.vectorize(self.f)(x)

    def __mul__(self, other):
        return self.f(other)

    def __get__(self, other):
       if isinstance(other, vunc):
            print("hi")
            return self[self.f[other]]


class f:
    def __init__(self, fn):
        self.fn = fn

    def __mul__(self, other):
        return self.access(other)

    def __getitem__(self, other):
        return self.access(other)

    def __call__(self, other, *args, **kwargs):
        if args or kwargs:
            return self.access(other)(*args, **kwargs)
        return self.access(other)

    def access(self, x):
        if isinstance(x, f):
            return f(lambda s: self.fn(s))
        else:
            return self.fn(x)


array = np.asarray
rect = lambda z: array([rovnd * z.real, rovnd * z.imag])
comp = lambda *p: complex(*p)
def _rovnd(x):
    return vunc(round) * x
rovnd = vunc(np.vectorize(round))


if __name__ == '__main__':
    a = f(lambda x: x ** 2)
    b = f(lambda x: x + 1)

    print(a(a[a * 1]))