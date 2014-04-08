"""
Douglas-Rachford algorithm
https://www.ceremade.dauphine.fr/~peyre/numerical-tour/tours/optim_4b_dr/
Patrick L. Combettes and Jean-Christophe Pesquet, Proximal Splitting Methods in Signal Processing, in: Fixed-Point Algorithms for Inverse Problems in Science and Engineering, New York: Springer, 2010.
"""

import numpy as _np

from convexopt.algorithms.util import Algorithm


class DouglasRachford(Algorithm):
    """Minimize f(x) + g(y)

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

    def __init__(self, f, g, gamma, mu=1.99, *args, **kwargs):
        super(DouglasRachford, self).__init__(*args, **kwargs)

        if f.gradient.shape[1] is None:
            assert g.gradient.shape[1] is not None
            self.y = _np.zeros(g.gradient.shape[1])
        elif g.gradient.shape[1] is None:
            assert f.gradient.shape[1] is not None
            self.y = _np.zeros(f.gradient.shape[1])
        else:
            assert f.gradient.shape[1] == g.gradient.shape[1]
            self.y = _np.zeros(f.gradient.shape[1])

        self._f = f
        self._g = g

        assert 0 < gamma # TODO assert for any assignment (via property)
        self.gamma = gamma # TODO provide an automatic guess
        assert 0 < mu < 2 # TODO assert for any assignment (via property)
        self.mu = mu

    def step(self):
        rpf = lambda x: 2 * self._f.gradient.backward(x, self.gamma) - x
        rpg = lambda x: 2 * self._g.gradient.backward(x, self.gamma) - x
        self.y = (1 - 0.5 * self.mu) * self.y + 0.5 * self.mu * rpg(rpf(self.y))

    @property
    def x(self):
        return self._f.gradient.backward(self.y, self.gamma) # TODO caching

douglas_rachford = DouglasRachford.run

