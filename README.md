convexopt
=========

Convex optimization framework for Python

TODO
----

- tests, examples, documentation
  - scipy's nosetests with np autoimport
  - build documentation using Sphinx
- constraints (proximal mapping is orthogonal projection)
  - PositiveIndicatorFunction
    - gradient: PositiveIndicatorFunctionGradient
      - backward: return np.maximum(x, 0)
  - how can constraints be combined with other functions?
    - e.g. ||x||_1 s.t. x >= 0
- stacked and reshaped operators
- inspection to find out what methods are implemented
- automatically apply the Moreau decomposition:
  x = prox_f(x) + prox_f*(x)
  => x = f.gradient.backward(x) + f.conjugate.gradient.backward(x)
  => If f.gradient or f.gradient.backward are not implemented, try
     f.conjugate.gradient.backward instead.
- logging to console using the logging module
- cached sparse decomposition for DataTerm backward operator

