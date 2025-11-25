"""Linear Systems, Sparse Solutions, and Sudoku

Prabhu Babu, Kristiaan Pelckmans, Petre Stoica, and Jian Li
IEEE Signal Processing Letters, vol. 17, no. 1, January 2010
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from convexopt.algorithms import fista
from convexopt.operators import NonnegativeL1Norm, DataTerm

class LinearIndexer(object):
    def __init__(self, shape):
        self.shape = shape
        self.index = np.arange(np.prod(shape)).reshape(shape)
        
    def __getitem__(self, key):
        return np.ravel(self.index[key])

def solve_sudoku(clues, maxiter=1000, epsilon=0.5, threshold=1e-6, l1weight=1e-2):
    clues = np.asarray(clues, dtype=int)
    n = len(clues)
    assert clues.shape == (n, n), "sudoku must be square"
    assert int(np.sqrt(n)) ** 2 == n, "size of sudoku must be a square number"
    assert 0 < epsilon < 1, "epsilon must be between 0 and 1"

    idx = LinearIndexer((9, 9, 9))
    cs = [] # constraints: each entry is a list of components whose sum should be one

    for j in range(9): # in each row, ...
        for i in range(9): # ...each number...
            cs.append(idx[j, :, i]) # ...must occur exactly once

    for j in range(9): # in each column, ...
        for i in range(9): # ...each number...
            cs.append(idx[:, j, i]) # ...must occur exactly once

    for x in range(0, 9, 3): # in each box along x...
        for y in range(0, 9, 3): # ...and y, ...
            for i in range(9): # ...each number...
                cs.append(idx[x:x+3, y:y+3, i]) # ...must occur exactly once

    for x in range(9): # in each cell along x...
        for y in range(9): # ...and y, ...
            cs.append(idx[x, y, :]) # ...there must be exactly one number

    for i, row in enumerate(clues): # for each cell along x...
        for j, col in enumerate(row): # ...and y...
            if col: # ...for that a nonzero clue is given, ...
                cs.append(idx[i, j, col - 1]) # ...this number must occur in this cell

    ms = [sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)) for k in cs]
    A = sla.aslinearoperator(sp.vstack(ms))
    b = np.ones(len(cs))

    l1 = l1weight * NonnegativeL1Norm()
    l2 = DataTerm(A, b)

    # iterative reweighted l1-norm minimization
    x = fista(l2, l1, maxiter=maxiter)
    while True:
        tau = 1.0 / (x + epsilon)
        old_x = x
        x = fista(l2, l1 * tau, maxiter=maxiter)
        d = np.square(x - old_x).sum()
        print(d)
        if d < threshold:
            break
    x = x.reshape(9, 9, 9) # row, column, number -> probability

    return np.argmax(x, axis=2) + 1

def check_sudoku(solution, clues=None):
    if clues is not None:
        clues = np.asarray(clues, dtype=int)
        assert solution.shape == clues.shape, "solution shape does not match clues shape"
        assert all(solution[clues != 0] == clues[clues != 0]), "solution does not match clues"

    n = len(solution)
    assert solution.shape == (n, n), "sudoku must be square"
    assert int(np.sqrt(n)) ** 2 == n, "size of sudoku must be a square number"

    for i, row in enumerate(solution, 1):
        assert sorted(set(row)) == list(range(1, n + 1)), "row %d is invalid" % i
    for i, col in enumerate(solution.T, 1):
        assert sorted(set(col)) == list(range(1, n + 1)), "column %d is invalid" % i
    for i in range(3):
        for j in range(3):
            assert sorted(set(solution[3*i:3*i+3, 3*j:3*j+3].ravel())) == list(range(1, n + 1)), "box %d-%d is invalid" % (i + 1, j + 1)

if __name__ == "__main__":
    clues = [
        [0, 0, 3, 0, 0, 9, 0, 8, 1],
        [0, 0, 0, 2, 0, 0, 0, 6, 0],
        [5, 0, 0, 0, 1, 0, 7, 0, 0],
        [8, 9, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 6, 0, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 7],
        [0, 0, 9, 0, 2, 0, 0, 0, 8],
        [0, 7, 0, 0, 0, 4, 0, 0, 0],
        [2, 5, 0, 8, 0, 0, 6, 0, 0],
    ]

    solution = solve_sudoku(clues)
    print(solution)
    check_sudoku(solution, clues)

