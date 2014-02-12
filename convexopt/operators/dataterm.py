import numpy as _np

from convexopt.operators.util import Operator, squared_operator_norm

__all__ = ["DataTerm"]


class DataTermGradient(Operator):
    def __init__(self, A, b):
        super(DataTermGradient, self).__init__()
        if A.shape[:1] != b.shape:
            raise TypeError("incompatible shapes")
        self.A = A  # LinearOperator
        self.b = b  # vector
        self._lipschitz = None

    def __call__(self, x):
        return self.A.rmatvec(self.A.matvec(x) - self.b)

    @property
    def shape(self):
        return (self.A.shape[1], self.A.shape[1])

    @property
    def lipschitz(self):
        if self._lipschitz is None:
            self._lipschitz = squared_operator_norm(self.A)
        return self._lipschitz


class DataTerm(Operator):
    """0.5 || A x - b ||_2^2"""

    def __init__(self, A, b):
        super(DataTerm, self).__init__()
        if A.shape[:1] != b.shape:
            raise TypeError("incompatible shapes")
        self.A = A  # LinearOperator
        self.b = b  # vector
        self.gradient = DataTermGradient(A, b)

    def __call__(self, x):
        return 0.5 * _np.linalg.norm(self.A.matvec(x) - self.b, 2) ** 2

    @property
    def shape(self):
        return (1, self.A.shape[1])
