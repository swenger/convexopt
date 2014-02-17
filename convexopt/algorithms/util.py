"""Utilities for algorithms
"""

import numpy as _np


class Callback(object):
    pass


class ErrorPrinter(Callback):
    def __init__(self, true_x):
        self.true_x = true_x

    def __call__(self, obj):
        print _np.linalg.norm(obj.x.ravel() - self.true_x.ravel())


class StepLimiter(Callback):
    def __init__(self, niter):
        self.step = 0
        self.niter = niter

    def __call__(self, obj):
        self.step += 1
        if self.step >= self.niter:
            return "maximum number of iterations reached (%d)" % self.niter


class Logger(object):
    """Logger for algorithm convergence.

    Usually passed to an algorithm as a callback.

    Parameters
    ----------
    objective : `Operator`, optional
        A function to evaluate the objective function value.
    true_x : `np.ndarray`, optional
        The true optimum, if known, to compute the reconstruction error.
    """

    def __init__(self, objective=None, true_x=None):
        super(Logger, self).__init__()

        self.objective = objective
        self.true_x = true_x

        self.objectives = []
        """The objective function values for each time step."""

        self.errors = []
        """The reconstruction errors for each time step."""

    def __call__(self, obj):
        self.objectives.append(self.objective(obj.x)
                               if self.objective is not None else float("nan"))
        self.errors.append(_np.linalg.norm(obj.x.ravel() - self.true_x.ravel())
                           if self.true_x is not None else float("nan"))


class _ClassOrInstanceMethod(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        from types import MethodType
        if obj is None:
            return MethodType(self.func, cls, type(cls))
        else:
            return MethodType(self.func, obj, cls)


def _find_subclasses_with_initparams(dct, cls):
    import inspect
    import warnings

    result = {}
    for cls in dct.values():
        if not (isinstance(cls, type) and issubclass(cls, Callback)):
            continue
        
        try:
            argspec = inspect.getargspec(cls.__init__)
        except TypeError:
            continue
        args = argspec.args[1:]
        if argspec.varargs:
            warnings.warn("varargs in %r" % cls)
        if argspec.keywords:
            warnings.warn("keywords in %r" % cls)

        for arg in args:
            if arg in result:
                warnings.warn("ambiguous constructor argument %r in %r" % (arg, cls))
            else:
                result[arg] = cls

    return result


class Algorithm(object):
    """Base class for iterative algorithms

    Parameters
    ----------
    callbacks : list of callable, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.

    Additional keyword arguments are passed to appropriate constructors of
    `Callback` subclasses.  These callback instances are then appended to the
    list of callbacks.

    Attributes
    ----------
    x : `np.ndarray`
        Solution vector.
    stopping_reasons : list of (callback, message)
        The stopping criteria that caused termination.
    """

    _delegate_kwargs = _find_subclasses_with_initparams(globals(), Callback)

    def __init__(self, callbacks=(), **kwargs):
        self._callbacks = list(callbacks)
        delegates = {}
        for key, value in kwargs.items():
            delegates.setdefault(self._delegate_kwargs[key], {})[key] = value
        self._callbacks.extend(k(**v) for k, v in delegates.items())

        self.stopping_reasons = []

    def step(self):
        """Compute an iteration step and store results in `self.x`
        """

        raise NotImplementedError

    @_ClassOrInstanceMethod
    def run(cls_or_self, *args, **kwargs):
        """Run the algorithm until a stopping criterion is satisfied.
        """

        if isinstance(cls_or_self, Algorithm):
            if args or kwargs:
                raise TypeError("run() does not take any arguments when called as an instance method")
            self = cls_or_self
        else:
            self = cls_or_self(*args, **kwargs)

        while not self.stopping_reasons:
            self.step()
            for callback in self._callbacks:
                message = callback(self)
                if message:
                    self.stopping_reasons.append((callback, message))
        
        return self.x
