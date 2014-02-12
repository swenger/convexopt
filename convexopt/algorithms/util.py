"""Utilities for algorithms
"""

import numpy as _np


class Logger(object):
    """Logger for algorithm convergence.

    Usually passed to an algorithm as a callback.

    Parameters
    ----------
    objective : `Operator`
        A function to evaluate the objective function value.
    true_x : `np.ndarray`
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

    def __call__(self, x):
        self.objectives.append(self.objective(x)
                              if self.objective is not None else float("nan"))
        self.errors.append(_np.linalg.norm(x.ravel() - self.true_x.ravel())
                           if self.true_x is not None else float("nan"))
