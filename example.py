import numpy as np

from convexopt.algorithms.forwardbackward import ForwardBackward
from convexopt.algorithms.util import Logger, ErrorPrinter
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
    log = Logger(l1 + l2, x)

    # solve
    
    # xr = ForwardBackward.run(l1, l2, niter=10000, callbacks=[ErrorPrinter(x), log])
    
    alg = ForwardBackward(l1, l2, niter=10000, callbacks=[ErrorPrinter(x), log])
    alg.run()
    xr = alg.x
    print "reasons for stopping: " + ", ".join(message for cls, message in alg.stopping_reasons)

    np.testing.assert_array_almost_equal(x, xr, decimal=3)

    # plot convergence
    import pylab
    pylab.loglog(log.errors, label="error")
    pylab.loglog(log.objectives, label="residual")
    pylab.legend()
    pylab.show()
