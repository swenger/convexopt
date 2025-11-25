import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator
import pylab as pl

from convexopt.algorithms import APGM
from convexopt.algorithms.util import ObjectiveLogger, ErrorLogger
from convexopt.operators import L2Distance, SliceSeparableNorm, L2Norm, GroupL1Norm

def image_gradient_operator(h, w):
    """ 
    Given an image shape (height, width), return a scipy.sparse.LinearOperator
    that corresponds to computing the forward-difference gradient of a 2D image.
    In particular, if image is the flattened 2D image where image 
    has shape (height, width), a greyscale image, then
        grad = image_gradient_operator(*image.shape)
        dxdy = grad * image.ravel()
    gives the derivatives in x and y direction, in particular, dxdy now contains
    all derivatives in x, then in y direction. To get the derivatives per pixel,
        dx, dy = (grad * image.ravel()).reshape(2, h, w)
    """
    dy_row = sp.spdiags([-np.ones(w), np.ones(w)], [0, 1], w, w).tolil()
    dy_row[:, -1] = 0
    dy_row[-1, :] = 0
    dy = sp.kron(dy_row, sp.eye(w, w)).tocsr()
    dx_row = sp.spdiags([-np.ones(h), np.ones(h)], [0, 1], h, h).tolil()
    dx_row[:, -1] = 0
    dx_row[-1, :] = 0
    dx = sp.kron(sp.eye(w, w), dx_row).tocsr()
    return aslinearoperator(sp.vstack((dx, dy), 'csr'))


def main():
    image = pl.imread("lena.png").mean(axis=-1)
    h, w = image.shape

    y_true = image.ravel()
    noise = np.random.normal(0, 0.1, y_true.shape)
    y_noisy = y_true + noise

    grad = image_gradient_operator(h, w)

    l1 = 0.1 * GroupL1Norm((2, h, w), 0)
    #l1 = 0.1 * SliceSeparableNorm(L2Norm(), (2, h * w), 1)
    l2 = L2Distance(y_noisy)
    obj = lambda x: l1(grad * x) + l2(x)
    
    algo = APGM(l2, l1, A=grad, maxiter=50, x0=y_noisy, 
                objectives=ObjectiveLogger(obj), 
                errors=ErrorLogger(y_true) )

    t0 = time.time()
    y_denoised = algo.run()
    print("took %.3f seconds" % (time.time() - t0))

    pl.figure()
    pl.gray()

    for i, (title, v) in enumerate([('ground truth', y_true), 
                                    ('noisy', y_noisy),
                                    ('denoised', y_denoised)]):
        pl.subplot(2, 3, i+1)
        pl.title(title)
        pl.xticks([])
        pl.yticks([])
        pl.imshow(v.reshape(image.shape), vmin=0, vmax=1)

    pl.subplot(2, 3, 4)
    pl.title('Objective Function')
    pl.plot(algo.objectives)
    pl.ylabel('objective')
    pl.xlabel('iteration')

    pl.subplot(2, 3, 5)
    pl.title('Errors')
    pl.plot(algo.errors)
    pl.ylabel('error')
    pl.xlabel('iteration')

    pl.draw()
    pl.show()


if __name__ == '__main__':
    main()

