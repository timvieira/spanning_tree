import numpy as np


def matrix_tree_theorem(A, r):
    """
    Compute the log-partition function and its gradients (marginal probabilities)
    for non-projective dependency parser given a weighted adjacency matrix
    `A: head × modifier ↦ log-weight` (note: diagonal ignored)
    and root weight vector `r: head ↦ log-weight`.

    References

    - Koo, Globerson, Carreras and Collins (EMNLP'07)
      Structured Prediction Models via the Matrix-Tree Theorem
      https://www.aclweb.org/anthology/D07-1015

    - Smith and Smith (EMNLP'07)
      Probabilistic Models of Nonprojective Dependency Trees
      https://www.aclweb.org/anthology/D07-1014

    - McDonald & Satta (IWPT'07)
      https://www.aclweb.org/anthology/W07-2216

    """

    # Numerical stability trick: We use an extension of the log-sum-exp and
    # exp-normalize tricks to our log-det-exp setting. [I haven't never seen 
    # this trick elsewhere.]
    #
    # Note: The `exp` function below is point-wise exponential, not the matrix
    # exponential!
    #
    # for any value c,
    #
    #    log(det(exp(c) * exp(A - c)))
    #     = log(exp(c)^n * det(exp(A - c)))
    #     = c*n + log(det(exp(A - c)))
    #
    # Furthermore,
    #
    #    ∇ log(det(exp(c) * exp(A - c)))
    #     = ∇ [ c*n + log(det(exp(A - c))) ]
    #     = exp(A - c)⁻ᵀ
    #
    c = max(r.max(), A.max())

    r = np.exp(r - c)
    A = np.exp(A - c)
    np.fill_diagonal(A, 0)

    L = np.diag(A.sum(axis=0)) - A   # The Laplacian matrix of a graph
    L[0,:] = r                       # Koo et al.'s efficiency trick

    lnz = np.linalg.slogdet(L)[1] + c*len(r)

    dL = np.linalg.inv(L).T
    dr = r * dL[0,:]

    dA = A * 0
    N = len(r)
    for h in range(N):
        for m in range(N):
            dA[h,m] = A[h,m] * (dL[m,m] * (m!=0) - dL[h,m] * (h!=0))

    return lnz, dr, dA
