import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from convexopt.algorithms import fista
from convexopt.operators import NonnegativeL1Norm, DataTerm

ijk = np.indices((9, 9, 9))
cs = [] # constraints

# x[j, 0, i] + x[j, 1, i] + x[j, 2, i] + ... + x[j, 8, i] = 1 # for all i, j: number of i's in j'ths row = 1
for j in range(9):
    for i in range(9):
        k = np.ravel_multi_index(ijk[:, j, :, i], (9, 9, 9))
        cs.append(sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)))

# x[0, j, i] + x[1, j, i] + x[2, j, i] + ... + x[8, j, i] = 1 # for all i, j: number of i's in j'ths column = 1
for j in range(9):
    for i in range(9):
        k = np.ravel_multi_index(ijk[:, :, j, i], (9, 9, 9))
        cs.append(sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)))

# box constraints
for j in range(9):
    x = 3 * (j % 3)
    y = 3 * (j // 3)
    for i in range(9):
        k = np.ravel_multi_index(ijk[:, x:x+3, y:y+3, i].reshape(3, -1), (9, 9, 9))
        cs.append(sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)))

# cell constraints: exactly one number in cell
for x in range(9):
    for y in range(9):
        k = np.ravel_multi_index(ijk[:, x, y, :].reshape(3, -1), (9, 9, 9))
        cs.append(sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)))

# clues
clues = [
        (1, 3, 3),
        (1, 6, 9),
        (1, 8, 8),
        (1, 9, 1),
        (2, 4, 2),
        (2, 8, 6),
        (3, 1, 5),
        (3, 5, 1),
        (3, 7, 7),
        (4, 1, 8),
        (4, 2, 9),
        (5, 3, 5),
        (5, 4, 6),
        (5, 6, 1),
        (5, 7, 2),
        (6, 8, 3),
        (6, 9, 7),
        (7, 3, 9),
        (7, 5, 2),
        (7, 9, 8),
        (8, 2, 7),
        (8, 6, 4),
        (9, 1, 2),
        (9, 2, 5),
        (9, 4, 8),
        (9, 7, 6),
        ]
for row, column, number in clues:
    k = np.ravel_multi_index(ijk[:, row - 1, column - 1, number - 1], (9, 9, 9))
    cs.append(sp.coo_matrix(([1], ([0], [k])), shape=(1, 9 * 9 * 9)))

# output clues
m = np.zeros((9, 9), dtype=np.uint8)
for row, column, number in clues:
    m[row - 1, column - 1] = number
print m

A = sla.aslinearoperator(sp.vstack(cs))
b = np.ones(len(cs))

l1 = 1e-2 * NonnegativeL1Norm()
l2 = DataTerm(A, b)
epsilon = 0.5

x = fista(l2, l1, maxiter=1000)
while True:
    tau = 1.0 / (x + epsilon)
    old_x = x
    x = fista(l2, l1 * tau, maxiter=1000)
    d = np.square(x - old_x).sum()
    print d
    if d < 1e-6:
        break
x = x.reshape(9, 9, 9) # row, column, number -> probability

solution = np.argmax(x, axis=2) + 1

print solution

