from inspect import getargspec as _getargspec

import numpy as _np


def squared_operator_norm(A, niter=100):
    """Return largest eigenvalue of A.T A"""
    x = _np.random.randn(A.shape[1])
    for i in range(niter):
        x /= _np.linalg.norm(x)
        x = A.rmatvec(A.matvec(x))
    return _np.linalg.norm(x)


class Operator(object):
    shape = (None, None)
    lipschitz = None
    gradient = None

    def __call__(self, x):
        """Apply operator self(x)"""
        raise NotImplementedError

    def forward(self, x, tau):
        """Apply forward operator (1 - tau * self)(x)"""
        return x - tau * self(x)

    def backward(self, x, tau):
        """Apply backward operator (1 + tau * self)^-1(x)"""
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
    def __init__(self, op, a):
        super(ScaledOperator, self).__init__()
        self.op = op
        self.a = a
        if op.gradient is not None:
            self.gradient = ScaledOperator(op.gradient, a)

    def __call__(self, x):
        # (a A)(x) = a A(x)
        return self.a * self.op(x)

    def forward(self, x, tau):
        # (1 - tau (a A))(x) = (1 - (tau a) A)(x)
        return self.op.forward(x, self.a * tau)

    def backward(self, x, tau):
        # (1 + tau (a A))^-1(x) = (1 + (tau a) A)^-1(x)
        return self.op.backward(x, self.a * tau)
