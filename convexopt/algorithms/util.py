import numpy as _np


class Logger(object):
    def __init__(self, objective=None, true_x=None):
        super(Logger, self).__init__()

        self.objective = objective
        self.true_x = true_x

        self.residuals = []
        self.errors = []

    def __call__(self, x):
        self.residuals.append(self.objective(x)
                              if self.objective is not None else float("nan"))
        self.errors.append(_np.linalg.norm(x.ravel() - self.true_x.ravel())
                           if self.true_x is not None else float("nan"))
