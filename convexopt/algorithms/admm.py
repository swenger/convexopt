"""
Alternating Direction Method of Multipliers (ADMM)
"""

import numpy as _np

from convexopt.algorithms.util import Algorithm


class ADMM(Algorithm):
    """Minimize f(x) + g(y) s.t. x - y = 0 using ADMM

    Parameters
    ----------
    f : `Operator`
        An operator whose gradient implements `backward()`.
    g : `Operator`
        An operator whose gradient implements `backward()`.
    callbacks : list of callable, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.
    """

    def __init__(self, f=None, g=None, rho=1.0, alpha=1.0, *args, **kwargs):
        super(ADMM, self).__init__(*args, **kwargs)

        if f.gradient.shape[1] is None:
            assert g.gradient.shape[1] is not None
            self.x = _np.zeros(g.gradient.shape[1])
        elif g.gradient.shape[1] is None:
            assert f.gradient.shape[1] is not None
            self.x = _np.zeros(f.gradient.shape[1])
        else:
            assert f.gradient.shape[1] == g.gradient.shape[1]
            self.x = _np.zeros(f.gradient.shape[1])

        self._f = f
        self._g = g
        self._z = self.x
        self._u = _np.zeros_like(self.x)
        self._rho = rho
        self._alpha = alpha

        self._primal_residuals = []
        self._dual_residuals = []

    def step(self):
        # TODO: passing 1 / rho correct?
        # x-update
        self.x  = self._f.gradient.backward(self._z - self._u, 1 / self._rho)
        # overrelaxation step
        x_hat = self._alpha * self.x + (1 - self._alpha) * self._z
        # z-update
        z_old = self._z
        self._z = self._g.gradient.backward(x_hat + self._u, 1 / self._rho)
        # update residuals
        self._u += x_hat - self._z

        # diagnostics
        primal_residual = _np.linalg.norm(self.x - self._z)
        dual_residual = _np.linalg.norm(self._rho * (self._z - z_old))
        self._primal_residuals.append(primal_residual)
        self._dual_residuals.append(dual_residual)
        # TODO convergence checks here

admm = ADMM.run

