"""
Title: svd_solver.py

Author: HovarAK
Date: 2026-08-19
Description: Linear solver based on Singular Value Decomposition (SVD) for Linear Least Squeares Problems.
"""

import numpy as np
import scipy.linalg as la


def svd_solver(A: np.ndarray, b: np.ndarray, rcond=1e-10) -> np.ndarray:
    """
    Description: Solves the L.L.S problem Ax = b using SVD.

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
            f"Incompatible shapes for svd_solver: {A.shape} and {b.shape}"
            f" (first dimension of A must match first dim of B)"
        )

    """
    Solution: Truncated SVD (TSVD) with a tolerance
    
    Why did I choose TSVD over Normal SVD or Ridge Regularization?
        - Normal SVD requires (A.T @ A) to have full rank, equivalent to A having full rank.
            Thus, this can prevent the Pseudo-Inverse of A from being calculated.
        - Ridge Regularization will always ensure that (A.T @ A) will have full rank, i.e invertible.
            However, (A.T @ A + lambda * I_n) always introduces bias into the solution and you will
            have to choose lambda, which requires cross-validation or extra hypertunning.
    """

    U, S, Vt = la.svd(A, full_matrices=False, lapack_driver="gesdd")

    # Determines which singular values are "reliable"
    tol = rcond * S.max()
    reliable = S > tol
    S_inv = np.divide(1, S, out=np.zeros_like(S), where=reliable)

    x = Vt.T @ (S_inv * (U.T @ b))
    return x
