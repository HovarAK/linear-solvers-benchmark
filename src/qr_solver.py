"""
Title: qr_solver.py

Author: HovarAK
Date: 2026-08-20
Description: Linear solver based on QR Decomposition (Modified Gram-Schmidt) for Linear Least Squares Problems.
"""

import numpy as np
import scipy.linalg as la


def m_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description: Factors A = QR using Modified Gram-Schmidt (MGS).

    Parameters:
    -----------
    A : np.ndarray, shape (m,n)
        Matrix to factorize.

    Returns:
    --------
    Q : np.ndarray, shape (m,n)
        Matrix with orthonormal columns.

    R : np.ndarray, shape (n,n)
        Upper triangular matrix.
    """

    # Copies A so Q can be built in place without mutating the caller's matrix
    Q = A.copy()

    n_cols = int(A.shape[1])
    R = np.zeros((n_cols, n_cols))

    for j in range(n_cols):
        # Normalizes column j; its length becomes the diagonal entry R[j, j]
        R[j, j] = la.norm(Q[:, j])

        if np.isclose(R[j, j], 0):
            raise ValueError(
                f"Rank-deficient matrix: column {j} is linearly dependent on "
                f"the preceding columns (norm ~ 0); m_gram_schmidt requires "
                f"A to have full column rank."
            )

        Q[:, j] = Q[:, j] / R[j, j]

        # Removes the column j component from every remaining column
        for k in range(j + 1, n_cols):
            R[j, k] = Q[:, j] @ Q[:, k]
            Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]

    return Q, R


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Description: Solves the L.L.S problem Ax = b using QR decomposition.

    Parameters:
    -----------
    A : np.ndarray, shape (m,n)
        Design matrix (features).

    b : np.ndarray, shape (m,)
        Target vector.

    Returns:
    --------
    x : np.ndarray, shape (n,)
        Solution vector minimizing ||Ax - b||^2.
    """

    # Verifies the Pre-Requirment on Matrix A and Vector b
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for qr_solve: {A.shape} and {b.shape}"
            f" (first dimension of A must match first dim of B)"
        )

    """
    Solution: QR Factorization using Modified Gram-Schmidt (MGS)

    Why MGS over Classical Gram-Schmidt (CGS)?
        - MGS has the same asymptotic cost as CGS, but has significantly better numerical stability because
            it orthogonalizes each remaining column against the already computed, updated Q vector immediately,
            instead of against the original A columns.
    """

    Q, R = m_gram_schmidt(A)

    # Solves Rx = Q.T @ b by back substitution since R is upper triangular
    rhs = Q.T @ b
    x = la.solve_triangular(R, rhs)
    return x
