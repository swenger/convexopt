import numpy as np
import scipy.sparse as sp

from convexopt.algorithms import ForwardBackward, forward_backward, FISTA, ADMM, APGM
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
    maxiter = 5000

    apgm = APGM(l2, l1, rho=0.5, maxiter=maxiter, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    apgm.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in apgm.stopping_reasons)
    np.testing.assert_array_almost_equal(x, apgm.x, decimal=3)
    
    alg1 = ForwardBackward(l1, l2, maxiter=maxiter, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg1.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg1.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg1.x, decimal=3)

    xr = forward_backward(l1, l2, maxiter=maxiter)
    np.testing.assert_array_equal(alg1.x, xr)

    alg2 = ForwardBackward(l1, l2, maxiter=maxiter, alpha=0.9, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg2.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg2.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg2.x, decimal=3)
    
    alg3 = ForwardBackward(l1, l2, maxiter=maxiter, epsilon=1e-6, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg3.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg3.stopping_reasons)
    np.testing.assert_array_almost_equal(x, alg3.x, decimal=3)
        
    alg4 = FISTA(l2, l1, maxiter=maxiter, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    alg4.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in alg4.stopping_reasons)
    #np.testing.assert_array_almost_equal(x, alg4.x, decimal=3)

    admm1 = ADMM(l2, l1, rho=1.0, alpha=1.0, maxiter=maxiter, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    admm1.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in admm1.stopping_reasons)
    np.testing.assert_array_almost_equal(x, admm1.x, decimal=3)

    admm2 = ADMM(l2, l1, rho=0.5, alpha=1.8, maxiter=maxiter, objectives=ObjectiveLogger(l1 + l2), errors=ErrorLogger(x))
    admm2.run()
    print "reasons for stopping: " + ", ".join(message for cls, message in admm2.stopping_reasons)
    np.testing.assert_array_almost_equal(x, admm2.x, decimal=3)

    # plot convergence
    import pylab
    pylab.figure("residuals")
    pylab.loglog(alg1.objectives, label="fwbw, alpha=%f" % alg1._alpha)
    pylab.loglog(alg2.objectives, label="fwbw, alpha=%f" % alg2._alpha)
    pylab.loglog(alg3.objectives, label="fwbw, alpha=%f" % alg3._alpha)
    pylab.loglog(alg4.objectives, label="fista")
    pylab.loglog(admm1.objectives, label="admm")
    pylab.loglog(admm2.objectives, label="admm overrelax")
    pylab.loglog(apgm.objectives, label="apgm")
    pylab.legend()

    pylab.figure("errors")
    pylab.loglog(alg1.errors, label="fwbw, alpha=%f" % alg1._alpha)
    pylab.loglog(alg2.errors, label="fwbw, alpha=%f" % alg2._alpha)
    pylab.loglog(alg3.errors, label="fwbw, alpha=%f" % alg3._alpha)
    pylab.loglog(alg4.errors, label="fista")
    pylab.loglog(admm1.errors, label="admm")
    pylab.loglog(admm2.errors, label="admm overrelax")
    pylab.loglog(apgm.errors, label="apgm")
    pylab.legend()

    pylab.figure("admm diagnostics")
    pylab.subplot(211)
    pylab.title("primal residual")
    pylab.loglog(admm1._primal_residuals, label="admm")
    pylab.loglog(admm2._primal_residuals, label="admm overrelax")
    pylab.subplot(212)
    pylab.title("dual residual")
    pylab.loglog(admm1._dual_residuals, label="admm")
    pylab.loglog(admm2._dual_residuals, label="admm overrelax")

    pylab.show()

