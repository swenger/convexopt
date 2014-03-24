""":math:`\\ell_1`-norm :math:`\\Vert x \\Vert_1`
"""

import numpy as _np

from convexopt.operators.util import Operator

__all__ = ["L1Norm", "NonnegativeL1Norm"]


class L1NormGradient(Operator):
    """Gradient of l1-norm

    The actual subgradient would be multi-valued at zero, which is not
    currently supported.  However, the backward operator, the proximal mapping
    of the l1-norm, is implemented (as soft-thresholding).
    """

    def backward(self, x, tau):
        return _np.maximum(abs(x) - tau, 0) * _np.sign(x)


class L1Norm(Operator):
    """l1-norm ||x||_1
    """

    shape = (0, None)
    gradient = L1NormGradient()

    def __call__(self, x):
        return _np.linalg.norm(x.ravel(), 1)


class NonnegativeL1NormGradient(Operator):
    """Gradient of l1-norm

    The actual subgradient would be multi-valued at zero, which is not
    currently supported.  However, the backward operator, the proximal mapping
    of the l1-norm, is implemented (as soft-thresholding).
    """

    def backward(self, x, tau):
        return _np.maximum(x - tau, 0)


class NonnegativeL1Norm(Operator):
    """l1-norm ||x||_1
    """

    shape = (0, None)
    gradient = NonnegativeL1NormGradient()

    def __call__(self, x):
        return _np.linalg.norm(x.ravel(), 1) if all(x.ravel() >= 0) else float("inf")
