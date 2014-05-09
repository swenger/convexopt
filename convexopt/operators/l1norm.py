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


''' # weighted L1 norms - can often be replaced by ScaledOperator, even when weights is an array!
class L1NormGradient(Operator):
    """Gradient of l1-norm

    The actual subgradient would be multi-valued at zero, which is not
    currently supported.  However, the backward operator, the proximal mapping
    of the l1-norm, is implemented (as soft-thresholding).
    """

    def __init__(self, weights=None):
        super(L1NormGradient, self).__init__()
        self.weights = weights

    def backward(self, x, tau):
        if self.weights:
            return _np.maximum(abs(x) - tau * self.weights, 0) * _np.sign(x)
        else:
            return _np.maximum(abs(x) - tau, 0) * _np.sign(x)


class L1Norm(Operator):
    """l1-norm ||x||_1
    """

    shape = (0, None)

    def __init__(self, weights=None):
        super(L1Norm, self).__init__()
        self.weights = weights
        self.gradient = L1NormGradient(weights)

    def __call__(self, x):
        if self.weights:
            return _np.linalg.norm(x.ravel() * self.weights.ravel(), 1)
        else:
            return _np.linalg.norm(x.ravel(), 1)


class NonnegativeL1NormGradient(Operator):
    """Gradient of l1-norm

    The actual subgradient would be multi-valued at zero, which is not
    currently supported.  However, the backward operator, the proximal mapping
    of the l1-norm, is implemented (as soft-thresholding).
    """

    def __init__(self, weights=None):
        super(NonnegativeL1NormGradient, self).__init__()
        self.weights = weights

    def backward(self, x, tau):
        if self.weights:
            return _np.maximum(x - tau * self.weights, 0)
        else:
            return _np.maximum(x - tau, 0)


class NonnegativeL1Norm(Operator):
    """l1-norm ||x||_1
    """

    shape = (0, None)

    def __init__(self, weights=None):
        super(NonnegativeL1Norm, self).__init__()
        self.weights = weights
        self.gradient = NonnegativeL1NormGradient(weights)

    def __call__(self, x):
        if self.weights:
            return _np.linalg.norm(x.ravel() * self.weights.ravel(), 1) \
                if all(x.ravel() >= 0) else float("inf")
        else:
            return _np.linalg.norm(x.ravel(), 1) \
                if all(x.ravel() >= 0) else float("inf")
'''
