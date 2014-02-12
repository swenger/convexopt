"""Forward-backward algorithm for solving A(x) + B(x) = 0
"""

import numpy as _np


def forward_backward(A, B, niter=100, gamma=1.0, callback=None):
    """Forward-backward algorithm

    Find a zero of :math:`A(x) + B(x)`.

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
