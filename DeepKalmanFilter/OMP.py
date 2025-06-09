import numpy as np

def OMP(K, y, A):
    Res = y
    m, n = A.shape

    Q = np.zeros((m, K))
    R = np.zeros((K, K))
    Rinv = np.zeros((K, K))
    w = np.zeros((m, K))
    x = np.zeros(n)

    kk = np.zeros(K, dtype=int)

    for J in range(K):
        # Index Search
        V, kkk = np.max(np.abs(A.T @ Res)), np.argmax(np.abs(A.T @ Res))
        kk[J] = kkk

        # Residual Update
        w[:,J:J+1] = A[:, kk[J]:kk[J]+1]
        for I in range(J):
            if J - 1 != 0:
                R[I:I+1,J:J+1] = Q[:,I:I+1].T @ w[:,J:J+1]
                w[:,J:J+1] = w[:,J:J+1] - R[I,J]*Q[:,I:I+1]
        R[J,J] = np.linalg.norm(w[:,J:J+1])
        Q[:,J:J+1] = w[:,J:J+1] / R[J,J]
        Res = Res - (Q[:,J:J+1]@Q[:,J:J+1].T) @ Res

    # Least Squares
    for J in range(K):
        Rinv[J,J] = 1 / R[J,J]
        if J != 0:
            for I in range(J):
                Rinv[I,J] = -Rinv[J,J] * (Rinv[I:I+1,:J] @ R[:J,J:J+1])

    xx = Rinv @ Q.T @ y

    for I in range(K):
        x[kk[I]] = xx[I]

    return x

