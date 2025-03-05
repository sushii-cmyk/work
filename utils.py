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


def uint(n, *, exc=True, inc=True):
    """
    n points in the Unit INTerval (R)

    :param n:
    :type n:
    :param inc:
    :type inc:
    :param exc:
    :type exc:
    :return:
    :rtype:
    """
    # teeny tiny epsilon diff for drawing
    eps = 0.0001
    r = np.linspace(0 + eps, 1 - eps, round(n), exc)[not inc:]
    return r


def unit(n, *, exc=True, inc=True):
    """
    n
    print(vunc(np.linalg.norm)[t]i points in the UNIT circle (C)
    exclude wi

    :param n:
    :type n:
    :param inc:
    :type inc:
    :param exc:
    :type exc:
    :return:
    :rtype:
    """

    return exp2ni(uint(n, exc=exc, inc=inc))


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


class f:
    def __init__(self, fn):
        self.fn = fn

    def __mul__(self, other):
        return self.fn(other)



array = np.asarray
rect = lambda z: array([int(z.real), int(z.imag)])
comp = lambda *p: complex(*p)
def _rovnd(x):
    return x
rovnd = vunc(_rovnd)