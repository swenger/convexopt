"""
Alternating Proximal Gradient Method for Convex Minimization (APGM)
"""

import numpy as _np
import scipy.sparse as _sp

from convexopt.algorithms.util import Algorithm
from convexopt.operators.util import squared_operator_norm


class APGM(Algorithm):
    """Minimize f(x) + g(y) s.t. A x + B y = c using APGM

    Parameters
    ----------
    f : `Operator`
        An operator whose gradient implements `backward()`.
    g : `Operator`
        An operator whose gradient implements `backward()`.
    A : `scipy.sparse.linalg.LinearOperator`
        A matrix, identity by default.
    B : `scipy.sparse.linalg.LinearOperator`
        A matrix, negative identity by default.
    c : `np.ndarray`
        A vector, zero by default.
    rho : `float`
        Step size parameter.
    gamma : `float`
        Step size parameter. Must be between 0 and 1.
    callbacks : list of callable, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.
    """

    def __init__(self, f, g, A=None, B=None, c=None, rho=1.0, gamma=0.99, x0=None, y0=None, *args, **kwargs):
        super(APGM, self).__init__(*args, **kwargs)

        if A is None:
            if f.gradient.shape[1] is not None:
                n = f.gradient.shape[1]
            elif g.gradient.shape[1] is not None:
                n = g.gradient.shape[1]
            elif B is not None and B.shape[1] is not None:
                n = B.shape[1]
            elif c is not None:
                n = len(c)
            else:
                raise ValueError("shape of A is undefined")
            A = _sp.linalg.aslinearoperator(_sp.eye(n, n))
        elif A.shape[1] is None:
            assert f.gradient.shape[1] is not None
        else:
            assert f.gradient.shape[1] is None or f.gradient.shape[1] == A.shape[1]

        if B is None:
            if f.gradient.shape[1] is not None:
                n = f.gradient.shape[1]
            elif g.gradient.shape[1] is not None:
                n = g.gradient.shape[1]
            elif A is not None and A.shape[1] is not None:
                n = A.shape[1]
            elif c is not None:
                n = len(c)
            else:
                raise ValueError("shape of B is undefined")
            B = _sp.linalg.aslinearoperator(-_sp.eye(n, n))
        elif B.shape[1] is None:
            assert g.gradient.shape[1] is not None
        else:
            assert g.gradient.shape[1] is None or g.gradient.shape[1] == B.shape[1]

        if c is None:
            c = _np.zeros(A.shape[0])
        assert c.shape == (A.shape[0],) == (B.shape[0],)

        self.x = _np.zeros(A.shape[1]) if x0 is None else x0
        self.y = _np.zeros(B.shape[1]) if y0 is None else y0
        self.u = _np.zeros(A.shape[0])

        self.f = f
        self.g = g
        self.A = A
        self.B = B
        self.c = c
        self.rho = rho

        assert 0 < gamma < 1
        self.tauA = gamma / squared_operator_norm(A)
        self.tauB = gamma / squared_operator_norm(B)

    def step(self):
        A, B, c = self.A, self.B, self.c
        rho, tauA, tauB = self.rho, self.tauA, self.tauB
        pf, pg = self.f.gradient.backward, self.g.gradient.backward

        # v <- x - tau_A A.T(A x + B y - c - rho u)
        v = self.x - tauA * A.rmatvec(A.matvec(self.x) + B.matvec(self.y) - c - rho * self.u)
        # x <- argmin_x 0.5 ||x - v||_2^2 + rho tau_A f(x)
        self.x = pf(v, rho * tauA)

        # w <- y - tau_B B.T(A x + B y - c - rho u)
        w = self.y - tauB * B.rmatvec(A.matvec(self.x) + B.matvec(self.y) - c - rho * self.u)
        # y <- argmin_y 0.5 ||y - w||_2^2 + rho tau_B g(y)
        self.y = pg(w, rho * tauB)

        # u <- u - (A x + B y - c) / rho
        self.u -= (A.matvec(self.x) + B.matvec(self.y) - c) / rho

apgm = APGM.run
