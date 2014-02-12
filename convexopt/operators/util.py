"""Base classes and utilities for operators
"""

from inspect import getargspec as _getargspec

import numpy as _np


def squared_operator_norm(A, niter=100):
    """Return largest eigenvalue of :math:`A^T A`

    The eigenvalue with largest magnitude, the Lipschitz constant of :math:`A^T
    A`, is computed by applying the power iteration method to :math:`A^T A`.

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        A linear operator that implements shape, matvec(), and rmatvec().
    niter : int
        Number of power iterations to perform.

    Returns
    -------
    float
        The largest eigenvalue of :math:`A^T A`.

    Examples
    --------
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> squared_operator_norm(aslinearoperator(np.eye(3)))
    1.0
    >>> squared_operator_norm(aslinearoperator(3 * np.ones(3)))
    27.0
    """

    x = _np.random.randn(A.shape[1])
    for i in range(niter):
        x /= _np.linalg.norm(x)
        x = A.rmatvec(A.matvec(x))
    return _np.linalg.norm(x)


class Operator(object):
    """Abstract operator

    Operators are functions that map from one vector space to another (as a
    special case, the vector space may be zero-dimensional, in which case it is
    treated as scalar).  Objective functions, terms of objective functions, and
    gradients of those terms are all operators.  They may implement any subset
    of a number of operations that are often useful in convex optimization
    algorithms, such as computing the gradient (or (sub-)differential), the
    resolvent operator (or backward operator), or the forward operator.  The
    proximal mapping can also be computed; it is simply the backward operator
    of the gradient.
    """

    shape = (None, None)
    """Lengths of input and output vectors

    May be `None` to indicate that any length is accepted.
    """

    lipschitz = None
    """Lipschitz constant

    May be `None` to indicate that the operator is not Lipschitz continuous, or
    that the Lipschitz constant cannot be computed efficiently.
    """

    gradient = None
    """Gradient operator

    May be `None` to indicate that the operator is not differentiable, or that
    no efficient implementation of the gradient operator is available.
    """

    conjugate = None
    """Convex conjugate

    May be `None` to indicate that the convex conjugate is undefined (for
    example, because the operator does not map to a scalar field), or that no
    efficient implementation of the convex conjugate is available.
    """

    def __call__(self, x):
        """Apply operator :math:`A(x)`

        Parameters
        ----------
        x : `np.ndarray`
            An array of length self.shape[1].

        Returns
        -------
        np.ndarray
            An array of length self.shape[0].

        Raises
        ------
        NotImplementedError
            If the operator does not support evaluation.
        """

        raise NotImplementedError

    def forward(self, x, tau):
        """Apply forward operator :math:`(1 - \\tau A)(x)`

        Parameters
        ----------
        x : `np.ndarray`
            An array of length self.shape[1].
        tau : float
            Step size parameter.

        Returns
        -------
        np.ndarray
            An array of length self.shape[0].

        Raises
        ------
        NotImplementedError
            If the operator does not implement a forward operator.
        """

        return x - tau * self(x)

    def backward(self, x, tau):
        """Apply backward operator :math:`(1 + \\tau A)^{-1}(x)`

        Parameters
        ----------
        x : `np.ndarray`
            An array of length self.shape[0].
        tau : float
            Step size parameter.

        Returns
        -------
        np.ndarray
            An array of length self.shape[1].

        Raises
        ------
        NotImplementedError
            If the operator does not implement a backward operator.
        """

        raise NotImplementedError

    def __mul__(self, a):
        return ScaledOperator(self, a)

    def __rmul__(self, a):
        return ScaledOperator(self, a)

    def __add__(self, op):
        return OperatorSum(self, op)

    def __repr__(self):
        try:
            arg = _getargspec(self.__init__)
        except TypeError:
            argnames = ()
        else:
            argnames = arg.args[1:]
            if arg.varargs is not None:
                argnames.append(arg.varargs)
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%r" % (key, getattr(self, key))
                      for key in argnames if hasattr(self, key)))


class OperatorSum(Operator):
    """The sum of multiple operators

    Parameters
    ----------
    *ops : list of `Operator`
        The operators to add.
    """

    def __init__(self, *ops):
        super(OperatorSum, self).__init__()

        shape = [None, None]
        for op in ops:
            if shape[0] is None:
                shape[0] = op.shape[0]
            elif op.shape[0] is not None and shape[0] != op.shape[0]:
                raise TypeError("incompatible output shapes")
            if shape[1] is None:
                shape[1] = op.shape[1]
            elif op.shape[1] is not None and shape[1] != op.shape[1]:
                raise TypeError("incompatible output shapes")
        self.shape = tuple(shape)

        self.ops = ops
        if all(op.gradient is not None for op in ops):
            self.gradient = OperatorSum(*[op.gradient for op in ops])

    def __call__(self, x):
        # (A + B)(x) = A(x) + B(x)
        return sum(op(x) for op in self.ops)

    def forward(self, x, tau):
        # (1 - tau (A + B))(x) = x - tau (A(x) + B(x))
        return x - tau * sum(op(x) for op in self.ops)

    def backward(self, x, tau):
        # (1 + tau (A + B))^-1(x) cannot easily be computed
        # from (1 + tau A)^-1(x) and (1 + tau B)^-1(x)
        raise NotImplementedError


class ScaledOperator(Operator):
    """A scalar multiple of an operator

    Parameters
    ----------
    op : `Operator`
        An operator.
    a : float
        A scalar scaling the result of the operator.
    b : float
        An optional scalar scaling the input of the operator.
    """

    def __init__(self, op, a, b=1):
        super(ScaledOperator, self).__init__()
        self.op = op
        self.a = a
        self.b = b
        if op.gradient is not None:
            # (grad (a A))(x) = a (grad A)(x)
            self.gradient = ScaledOperator(op.gradient, a)
        if op.conjugate is not None and a > 0:
            # (a A)*(x*) = a A*(x* / a) for a > 0
            self.conjugate = ScaledOperator(op.conjugate, a, 1.0 / a)

    def __call__(self, x):
        # (a A)(x) = a A(x)
        return self.a * self.op(x * self.b)

    def forward(self, x, tau):
        # (1 - tau (a A))(x) = (1 - (tau a) A)(x)
        return self.op.forward(x * self.b, self.a * tau)

    def backward(self, x, tau):
        # (1 + tau (a A))^-1(x) = (1 + (tau a) A)^-1(x)
        return self.op.backward(x * self.b, self.a * tau)
