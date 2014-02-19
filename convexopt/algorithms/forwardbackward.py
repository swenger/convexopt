"""Forward-backward algorithm
"""

from warnings import warn as _warn

import numpy as _np

from convexopt.algorithms.util import Algorithm


class ForwardBackward(Algorithm):
    """Minimize f(x) + g(x) using the forward-backward algorithm

    More generally, find a zero of A(x) + B(x), where A and B are, by default,
    the gradients of f and g, respectively.

    Parameters
    ----------
    f : `Operator`
        An operator whose gradient implements `backward()`.  Alternatively, `A`
        can be given.
    g : `Operator`
        An operator whose gradient implements `forward()`.  Alternatively, `B`
        can be given.
    A : `Operator`, optional
        An operator that implements `backward()`.
    B : `Operator`, optional
        An operator that implements `forward()`.
    callbacks : list of callable, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.
    gamma : float, optional
        Relative step size, 0 < `gamma` < 2.
    alpha : float, optional
        Extrapolation factor, 0 <= `alpha` < 1.
    epsilon : float, optional
        If `alpha` is not given, computes one for which convergence is
        guaranteed.
    """

    def __init__(self, f=None, g=None, A=None, B=None, gamma=1.0, alpha=None, epsilon=None, *args, **kwargs):
        super(ForwardBackward, self).__init__(*args, **kwargs)

        if (A is None) == (f is None):
            raise TypeError("must specify either A or f, but not both")
        if A is None:
            A = f.gradient
        if (B is None) == (g is None):
            raise TypeError("must specify either B or g, but not both")
        if B is None:
            B = g.gradient

        if A.shape[0] is None:
            assert B.shape[0] is not None
            self.x = _np.zeros(B.shape[0])
        elif B.shape[0] is None:
            assert A.shape[0] is not None
            self.x = _np.zeros(A.shape[0])
        else:
            assert A.shape[0] == B.shape[0]
            self.x = _np.zeros(A.shape[0])

        if not 0 < gamma < 2:
            _warn("convergence is only guaranteed for 0 < gamma < 2")

        if alpha is None:
            if epsilon is not None:
                if not 0 < epsilon < (9.0 - 4 * gamma) / (2.0 * gamma):
                    _warn("convergence is only guaranteed for 0 < epsilon < (9.0 - 4 * gamma) / (2.0 * gamma)")
                alpha = 1 + (_np.sqrt(9.0 - 4 * gamma - 2 * epsilon * gamma) - 3) / gamma
                print alpha # XXX
            else:
                alpha = 0
        else:
            if not 0 <= alpha < 1:
                _warn("convergence is only guaranteed for 0 <= alpha < 1")
            if epsilon is not None:
                _warn("ignoring epsilon since alpha is given")

        self._A = A
        self._B = B
        self._tau = gamma / B.lipschitz
        self._alpha = alpha

        if self._alpha:
            self._last_x = self.x

    def step(self):
        if self._alpha:
            y = (1 + self._alpha) * self.x - self._alpha * self._last_x
            self._last_x = self.x
        else:
            y = self.x
        self.x = self._A.backward(self._B.forward(y, self._tau), self._tau)
