import numpy as np

from convexopt.algorithms.forwardbackward import ForwardBackward
from convexopt.algorithms.util import ObjectiveLogger, ErrorLogger
from convexopt.operators import L1Norm, DataTerm

if __name__ == "__main__":
    from scipy.sparse.linalg import aslinearoperator

    # setup A and x
    np.random.seed(0)
    A = aslinearoperator(np.random.randn(15, 20))
    x = np.zeros(A.shape[1])
    p = np.random.permutation(len(x))[:3]
    x[p] = np.random.randn(len(p))

    # measure b = A x
    b = A.matvec(x)

    # setup problem argmin_x 0.5 ||A x - b||_2^2 + ||x||_1
    l1 = 1e-2 * L1Norm()
    l2 = DataTerm(A, b)

    # solve
    
    alg = ForwardBackward(l1, l2, maxiter=10000, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg.x, decimal=3)

    xr = ForwardBackward.run(l1, l2, maxiter=10000)
    np.testing.assert_array_equal(alg.x, xr)

    alg2 = ForwardBackward(l1, l2, maxiter=10000, alpha=0.9, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg2.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg2.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg2.x, decimal=3)
    
    alg3 = ForwardBackward(l1, l2, maxiter=10000, epsilon=1e-6, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg3.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg3.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg3.x, decimal=3)
    
    # plot convergence
    import pylab
    pylab.loglog(alg.errors, label="error")
    pylab.loglog(alg.objectives, label="residual")
    pylab.loglog(alg2.errors, label="error with alpha=%f" % alg2._alpha)
    pylab.loglog(alg2.objectives, label="residual with alpha=%f" % alg2._alpha)
    pylab.loglog(alg3.errors, label="error with alpha=%f" % alg3._alpha)
    pylab.loglog(alg3.objectives, label="residual with alpha=%f" % alg3._alpha)
    pylab.legend()
    pylab.show()
