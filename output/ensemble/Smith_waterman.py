import itertools
import numpy as np


def _init_x(i, j):
    MIN = -float("inf")
    if i > 0 and j == 0:
        return MIN
    else:
        if j > 0:
            return -10 + (-0.5 * j)
        else:
            return 0


def _init_y(i, j):
    MIN = -float("inf")
    if j > 0 and i == 0:
        return MIN
    else:
        if i > 0:
            return -10 + (-0.5 * i)
        else:
            return 0


def _init_m(i, j):
    MIN = -float("inf")
    if j == 0 and i == 0:
        return 0
    else:
        if j == 0 or i == 0:
            return MIN
        else:
            return 0


def smith_waterman(a, b, match_score=3, gap_cost=2, penalty=False):
    H = np.zeros((len(a) + 1, len(b) + 1), np.int)

    before_c1 = 'O'
    before_c2 = 'O'
    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        point = 0
        if penalty:
            if i != 0 and before_c1[0] == 'I' and a[i - 1][0] == 'I':
                if a[i - 1] != before_c1:
                    point -= 1
            if i != 0 and before_c2[0] == 'I' and b[j - 1][0] == 'I':
                if b[j - 1] != before_c2:
                    point -= 1
            before_c1 = a[i - 1]
            before_c2 = b[j - 1]

        match = H[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else - match_score + penalty)
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H[-1][-1]


def smith_waterman_affine(a, b, match_score=3, gap_cost=2):
    dim_i = len(a) + 1
    dim_j = len(b) + 1
    # abuse list comprehensions to create matrices
    X = [[_init_x(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
    Y = [[_init_y(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]
    M = [[_init_m(i, j) for j in range(0, dim_j)] for i in range(0, dim_i)]

    S = -10.
    E = -0.5
    for j in range(1, dim_j):
        for i in range(1, dim_i):
            X[i][j] = max((S + E + M[i][j - 1]), (E + X[i][j - 1]), (S + E + Y[i][j - 1]))
            Y[i][j] = max((S + E + M[i - 1][j]), (S + E + X[i - 1][j]), (E + Y[i - 1][j]))
            if a[i - 1] == b[j - 1]:
                score = match_score
            else:
                score = -match_score
            M[i][j] = max(score + M[i - 1][j - 1], X[i][j], Y[i][j])

    return M[-1][-1]
