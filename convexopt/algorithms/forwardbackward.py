import numpy as _np


def forward_backward(A, B, niter=100, gamma=1.0, callback=None):
    """Forward-backward algorithm

    Find a zero of :math:`A(x) + B(x)`.
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
            callback(x)

    return x
