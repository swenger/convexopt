"""Fast iterative shrinkage-thresholding algorithm
"""

from itertools import repeat as _repeat

import numpy as _np

from convexopt.algorithms.util import Algorithm


def _accumulate(iterable, func=lambda total, element: total + element):
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def _diff(iterable, func=lambda old, new: new - old):
    it = iter(iterable)
    old = next(it)
    for new in it:
        yield func(old, new)
        old = new


class FISTA(Algorithm):
    """Minimize f(x) + g(x) using FISTA

    Parameters
    ----------
    f : `Operator`
        An operator with Lipschitz-continuous `gradient`.
    g : `Operator`
        An operator whose gradient implements `backward()`.
    callbacks : list of callable, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.
    """

    def __init__(self, f=None, g=None, *args, **kwargs):
        super(FISTA, self).__init__(*args, **kwargs)

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
        self._y = self.x

        ts = _accumulate(_repeat(1.0), lambda t, el:
                         0.5 * (1 + _np.sqrt(1 + 4 * t ** 2)))
        self._alphas = iter(_diff(ts, lambda t1, t2: (t1 - 1) / t2))

    def step(self):
        alpha = next(self._alphas)
        last_x = self.x
        self.x = self._g.gradient.backward(
            self._y - self._f.gradient(self._y) / self._f.gradient.lipschitz,
            1.0 / self._f.gradient.lipschitz)
        self._y = (1 + alpha) * self.x - alpha * last_x


fista = FISTA.run

