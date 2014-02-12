import numpy as _np

from convexopt.operators.util import Operator

__all__ = ["L1Norm"]


class L1NormGradient(Operator):
    def backward(self, x, tau):
        return _np.maximum(abs(x) - tau, 0) * _np.sign(x)


class L1Norm(Operator):
    """|| x ||_1"""

    shape = (1, None)
    gradient = L1NormGradient()

    def __call__(self, x):
        return _np.linalg.norm(x.ravel(), 1)
