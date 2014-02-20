import numpy as np

from convexopt.algorithms import ForwardBackward, forward_backward, FISTA
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

    alg1 = ForwardBackward(l1, l2, maxiter=10000, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg1.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg1.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg1.x, decimal=3)

    xr = forward_backward(l1, l2, maxiter=10000)
    np.testing.assert_array_equal(alg1.x, xr)

    alg2 = ForwardBackward(l1, l2, maxiter=10000, alpha=0.9, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg2.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg2.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg2.x, decimal=3)
    
    alg3 = ForwardBackward(l1, l2, maxiter=10000, epsilon=1e-6, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg3.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg3.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg3.x, decimal=3)
    
    alg4 = FISTA(l2, l1, maxiter=10000, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg4.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg4.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg4.x, decimal=3)
    
    # plot convergence
    import pylab
    pylab.figure("residuals")
    pylab.loglog(alg1.objectives, label="fwbw, alpha=%f" % alg1._alpha)
    pylab.loglog(alg2.objectives, label="fwbw, alpha=%f" % alg2._alpha)
    pylab.loglog(alg3.objectives, label="fwbw, alpha=%f" % alg3._alpha)
    pylab.loglog(alg4.objectives, label="fista")
    pylab.legend()

    pylab.figure("errors")
    pylab.loglog(alg1.errors, label="fwbw, alpha=%f" % alg1._alpha)
    pylab.loglog(alg2.errors, label="fwbw, alpha=%f" % alg2._alpha)
    pylab.loglog(alg3.errors, label="fwbw, alpha=%f" % alg3._alpha)
    pylab.loglog(alg4.errors, label="fista")
    pylab.legend()

    pylab.show()
