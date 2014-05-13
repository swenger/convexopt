"""Separable norms (e.g., :math:`\\ell_{1,\infty}`
"""

import itertools as _it

import numpy as _np

from convexopt.operators.util import Operator

__all__ = ["GroupSeparableNorm", "SliceSeparableNorm"]


class SeparableNorm(Operator):
    """:math:`sum_i f_i(x_[i])`
    
    See "Sparse Reconstruction by Separable Approximation", section II D.
    """

    def __init__(self):
        self.gradient = SeparableNormGradient(self)

    def compose(self, args):
        raise NotImplementedError

    def decompose(self, arg):
        raise NotImplementedError

    def __call__(self, x):
        return sum(f(y) for f, y in zip(self.inner_ops, self.decompose(x)))


class SeparableNormGradient(Operator):
    def __init__(self, parent):
        self.parent = parent

    @property
    def shape(self):
        return (self.parent.shape[1], self.parent.shape[1])

    def backward(self, x, tau):
        return self.parent.compose(f.gradient.backward(y, tau)
                for f, y in zip(self.parent.inner_ops, self.parent.decompose(x))) \
                        .reshape(x.shape)


class GroupSeparableNorm(SeparableNorm):
    def __init__(self, inner_ops, groups):
        super(GroupSeparableNorm, self).__init__()
        self.groups = map(list, groups)
        all_indices = sorted(sum(self.groups, []))
        assert all_indices == range(len(all_indices)), "groups must be disjoint and complete"
        self.shape = (1, len(all_indices))
        try:
            self.inner_ops = list(inner_ops)
            assert len(self.inner_ops) == len(self.groups), "groups and inner_ops must have same size"
        except TypeError:
            self.inner_ops = _it.repeat(inner_ops)

    def compose(self, args):
        ret = _np.empty(self.shape[1])
        for group, arg in zip(self.groups, args):
            ret[group] = arg
        return ret

    def decompose(self, arg):
        return (arg.flat[x] for x in self.groups)


class SliceSeparableNorm(SeparableNorm):
    def __init__(self, inner_ops, input_shape=(-1,), axis=0):
        super(SliceSeparableNorm, self).__init__()
        assert axis < len(input_shape)
        self.input_shape = input_shape
        self.axis = axis if axis >= 0 else len(input_shape) + axis
        self.shape = (1, _np.prod(input_shape) if -1 not in input_shape else None)
        try:
            self.inner_ops = list(inner_ops)
            assert len(self.inner_ops) == input_shape[axis]
        except TypeError:
            self.inner_ops = _it.repeat(inner_ops)

    def compose(self, args):
        s, a = self.input_shape, self.axis
        return _np.asarray(list(args)) \
            .reshape(*((s[a],) + s[:a] + s[a+1:])) \
            .transpose(*(range(1, a + 1) + [0] + range(a + 1, len(s)))) \
            .ravel()

    def decompose(self, arg):
        s, a = self.input_shape, self.axis
        arg = arg.reshape(s) \
            .transpose(*([a] + range(0, a) + range(a + 1, len(s))))
        return arg.reshape(arg.shape[0], -1)
