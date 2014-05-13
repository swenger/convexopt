""":math:`\\ell_2`-norm :math:`\\Vert x \\Vert_2`
"""

import numpy as _np

from convexopt.operators.util import Operator

__all__ = ["L2Norm"]


class L2NormGradient(Operator):
    """Gradient of l2-norm
    """

    def backward(self, x, tau):
        with _np.errstate(divide='ignore'):
            # TODO return x * _np.maximum(1.0 - tau / _np.sqrt(_np.square(x).sum(axis=0)), 0.0)
            return x * _np.maximum(1.0 - tau / _np.linalg.norm(x.ravel()), 0.0)


class L2Norm(Operator):
    """l2-norm ||x||_2
    """

    shape = (0, None)
    gradient = L2NormGradient()

    def __call__(self, x):
        return _np.linalg.norm(x.ravel(), 2)
