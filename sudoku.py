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
        cs.append(np.ravel_multi_index(ijk[:, j, :, i], (9, 9, 9)))

# x[0, j, i] + x[1, j, i] + x[2, j, i] + ... + x[8, j, i] = 1 # for all i, j: number of i's in j'ths column = 1
for j in range(9):
    for i in range(9):
        cs.append(np.ravel_multi_index(ijk[:, :, j, i], (9, 9, 9)))

# box constraints
for j in range(9):
    x = 3 * (j % 3)
    y = 3 * (j // 3)
    for i in range(9):
        cs.append(np.ravel_multi_index(ijk[:, x:x+3, y:y+3, i].reshape(3, -1), (9, 9, 9)))

# cell constraints: exactly one number in cell
for x in range(9):
    for y in range(9):
        cs.append(np.ravel_multi_index(ijk[:, x, y, :].reshape(3, -1), (9, 9, 9)))

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

for i, row in enumerate(clues):
    for j, col in enumerate(row):
        if col:
            cs.append([np.ravel_multi_index(ijk[:, i, j, col - 1], (9, 9, 9))])

ms = [sp.coo_matrix((np.ones(len(k)), (np.zeros(len(k)), k)), shape=(1, 9 * 9 * 9)) for k in cs]
A = sla.aslinearoperator(sp.vstack(ms))
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
for x in map(set, solution): print x
for x in map(set, solution.T): print x

