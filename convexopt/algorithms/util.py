"""Utilities for algorithms
"""

import numpy as _np


class Callback(object):
    """Base class for callback objects to determine algorithm convergence
    """

    def __call__(self):
        """Called from within `Algorithm.run` to determine convergence

        Returns
        -------
        message : str or None
            Reason for termination, or `None` if not yet converged
        """

        raise NotImplementedError


class StepLimiter(Callback):
    """Limit the maximum number of steps of an algorithm.

    Parameters
    ----------
    maxiter : int
        Maximum number of steps.
    """

    def __init__(self, maxiter):
        super(StepLimiter, self).__init__()
        self.step = 0
        self.maxiter = maxiter

    def __call__(self, x):
        self.step += 1
        if self.step >= self.maxiter:
            return "maximum number of iterations reached (%d)" % self.maxiter


class Logger(list):
    """Base class for logger objects to log algorithm convergence
    """

    def __call__(self, x):
        """Log status for a new iterate
        """

        self.append(self.value(x))

    def value(self, x):
        """Compute status for a new iterate
        """

        raise NotImplementedError

    def __eq__(self, obj):
        return self is obj

    def __hash__(self):
        return id(self)


class ErrorLogger(Logger):
    """Log reconstruction errors for each time step

    Parameters
    ----------
    true_x : `np.ndarray`
        The true optimum to compute the reconstruction error.
    """

    def __init__(self, true_x):
        super(ErrorLogger, self).__init__()
        self.true_x = true_x

    def value(self, x):
        return _np.linalg.norm(x.ravel() - self.true_x.ravel())


class ObjectiveLogger(Logger):
    """Log objective function values for each time step

    Parameters
    ----------
    objective : `Operator`
        A function to evaluate the objective function value.
    """

    def __init__(self, objective):
        super(ObjectiveLogger, self).__init__()
        self.value = objective


class _ClassOrInstanceMethod(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        from types import MethodType
        if obj is None:
            return MethodType(self.func, cls, type(cls))
        else:
            return MethodType(self.func, obj, cls)


def _find_subclasses_with_initparams(dct, base_class, skip_args=1):
    import inspect
    import warnings

    result = {}
    for cls in dct.values():
        if not (isinstance(cls, type) and issubclass(cls, base_class)):
            continue

        try:
            argspec = inspect.getargspec(cls.__init__)
        except TypeError:
            continue
        args = argspec.args[skip_args:]
        if argspec.varargs:
            warnings.warn("varargs in %r" % cls)
        if argspec.keywords:
            warnings.warn("keywords in %r" % cls)

        for arg in args:
            if arg in result:
                warnings.warn("ambiguous constructor argument %r in %r"
                              % (arg, cls))
            else:
                result[arg] = cls

    return result


class Algorithm(object):
    """Base class for iterative algorithms

    Parameters
    ----------
    callbacks : list of `Callback`, optional
        Each callback is called once in each iteration with the current iterate
        as an argument.  May return a non-zero message to terminate the
        algorithm.
    loggers : list of `Logger`, optional
        Additional loggers to call once per iteration.

    Additional keyword arguments are passed to appropriate constructors of
    `Callback` subclasses.  These callback instances are then appended to the
    list of callbacks.  If the keyword arguments are `Logger` instances, they
    are instead added as attributes.

    Attributes
    ----------
    x : `np.ndarray`
        Solution vector.
    stopping_reasons : list of (callback, message)
        The stopping criteria that caused termination.

    When an attribute is set to a `Logger` instance, the logger is
    automatically called once per iteration.
    """

    _delegate_kwargs = _find_subclasses_with_initparams(globals(), Callback)

    def __init__(self, callbacks=(), loggers=(), **kwargs):
        self.callbacks = set(callbacks)
        self.loggers = set(loggers)
        self.stopping_reasons = set()

        delegates = {}
        for key, value in kwargs.items():
            if isinstance(value, Logger):
                setattr(self, key, value)
            else:
                delegates.setdefault(self._delegate_kwargs[key], {})[key] = value
        self.callbacks.update(k(**v) for k, v in delegates.items())

    def step(self):
        """Compute an iteration step and store results in `self.x`
        """

        raise NotImplementedError

    def __setattr__(self, key, value):
        try:
            self.loggers.remove(getattr(self, key))
        except (AttributeError, KeyError, TypeError):
            pass
        if isinstance(value, Logger):
            self.loggers.add(value)
        super(Algorithm, self).__setattr__(key, value)

    def __delattr__(self, key):
        value = getattr(self, key)
        if isinstance(value, Logger):
            self.loggers.remove(value)
        super(Algorithm, self).__delattr__(key, value)

    @_ClassOrInstanceMethod
    def run(cls_or_self, *args, **kwargs):
        """Run the algorithm until a stopping criterion is satisfied.
        """

        if isinstance(cls_or_self, Algorithm):
            if args or kwargs:
                raise TypeError("run() does not take any arguments " \
                                "when called as an instance method")
            self = cls_or_self
        else:
            self = cls_or_self(*args, **kwargs)

        while not self.stopping_reasons:
            self.step()
            for logger in self.loggers:
                logger(self.x)
            for callback in self.callbacks:
                message = callback(self.x)
                if message:
                    self.stopping_reasons.add((callback, message))

        return self.x
