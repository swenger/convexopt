"""Forward-backward algorithm
"""

import numpy as _np


def find_root(A, B, niter=100, gamma=1.0, callback=None):
    """Find a zero of A(x) + B(x) using the forward-backward algorithm

    Parameters
    ----------
    A : `Operator`
        An operator that implements `backward()`.
    B : `Operator`
        An operator that implements `forward()`.
    niter : int
        Number of iterations to perform.
    gamma : float
        Relative step size, 0 < `gamma` < 2.
    callback : callable
        Called once in each iteration with the current iterate as an argument.
        May raise `StopIteration` to terminate the algorithm.
    """

    if A.shape[0] is None:
        assert B.shape[0] is not None
        x = _np.zeros(B.shape[0])
    elif B.shape[0] is None:
        assert A.shape[0] is not None
        x = _np.zeros(A.shape[0])
    else:
        assert A.shape[0] == B.shape[0]
        x = _np.zeros(A.shape[0])

    tau = gamma / B.lipschitz
    for i in range(niter):
        x = A.backward(B.forward(x, tau), tau)
        if callback:
            try:
                callback(x)
            except StopIteration:
                break

    return x


def minimize(f, g, *args, **kwargs):
    """Minimize f(x) + g(x) using the forward-backward algorithm

    `args` and `kwargs` are passed to `find_root`.

    Parameters
    ----------
    f : `Operator`
        An operator whose gradient implements `backward()`.
    g : `Operator`
        An operator whose gradient implements `forward()`.

    See also
    --------
    find_root : Root finding algorithm used internally.
    """

    return find_root(f.gradient, g.gradient, *args, **kwargs)
