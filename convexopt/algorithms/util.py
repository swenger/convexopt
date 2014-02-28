"""Utilities for algorithms
"""

from inspect import getargspec as _getargspec
from warnings import warn as _warn

import numpy as _np


class CallbackMeta(type):
    registry = {}

    def __new__(cls, name, bases, dct):
        newcls = super(cls, CallbackMeta).__new__(cls, name, bases, dct)
        try:
            argspec = _getargspec(dct["__init__"])
        except (KeyError, TypeError):
            pass
        else:
            args = argspec.args[1:]
            if argspec.varargs:
                _warn("varargs in %r" % cls)
            if argspec.keywords:
                _warn("keywords in %r" % cls)

            for arg in args:
                if arg in cls.registry:
                    _warn("ambiguous constructor argument %r in %r"
                                  % (arg, cls))
                else:
                    cls.registry[arg] = newcls
        return newcls


class Callback(object):
    """Base class for callback objects to determine algorithm convergence
    """

    __metaclass__ = CallbackMeta

    @classmethod
    def get_callback_for_initparam(cls, key):
        return cls.__metaclass__.registry[key]

    def __call__(self, x):
        """Called from within `Algorithm.run` to determine convergence

        Parameters
        ----------
        x : `np.ndarray`
            The current iterate.

        Returns
        -------
        message : str or None
            Reason for termination, or `None` if not yet converged
        """

        raise NotImplementedError


class StepLimiter(Callback):
    """Limit the maximum number of steps of an algorithm

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

    def __init__(self, callbacks=(), loggers=(), **kwargs):
        self.callbacks = set(callbacks)
        self.loggers = set(loggers)
        self.stopping_reasons = set()

        delegates = {}
        for key, value in kwargs.items():
            if isinstance(value, Logger):
                setattr(self, key, value)
            else:
                delegates.setdefault(Callback.get_callback_for_initparam(key), {})[key] = value
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
