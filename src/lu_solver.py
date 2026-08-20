"""
Title: lu_solver.py

Author: HovarAK
Date: 2026-08-20
Description: Linear solvers based on LU Decomposition (with partial pivoting) and
    Cholesky (LL^T) Decomposition for square linear systems.
"""

import numpy as np
import scipy.linalg as la


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Description: Solves the square linear system Ax = b using LU decomposition.

    Parameters:
    -----------
    A : np.ndarray, shape (n, n)
        Square coefficient matrix.
        Note: LU requires a square system. For rectangular least-squares
        problems (m != n), route through the normal equations first
        (AᵀA x = Aᵀb) before calling this — see project notes on why
        that squares the condition number.

    b : np.ndarray, shape (n,)
        Right-hand side vector.

    Returns:
    --------
    x : np.ndarray, shape (n,)
        Solution vector.
    """

    # Verifies the Pre-Requirment on Matrix A: must be square to admit an LU factorization
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"lu_solve requires a square matrix A, got shape {A.shape}")

    # Verifies the Pre-Requirment on Matrix A and Vector b
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for lu_solve: {A.shape} and {b.shape}"
            f" (first dimension of A must match first dim of b)"
        )

    """
    Solution: LU Factorization with Partial Pivoting

    Why LU (with pivoting) over solving Ax = b directly?
        - Partial pivoting swaps rows to put the largest-magnitude entry on the
            diagonal at each step, which keeps the multipliers in L bounded and
            avoids the numerical blow-up that plain (unpivoted) Gaussian
            elimination can suffer from near-singular pivots.
        - Once A is factored, both triangular solves below are O(n^2) instead of
            the O(n^3) it would cost to re-eliminate for every new b.
    """

    # 1. Decompose A = P @ L @ U  (P is a permutation matrix)
    # pyright can't narrow la.lu's return type from permute_l=False alone
    # (it's a plain bool, not a Literal) — the 3-tuple unpacking is correct
    # at runtime; verified via the residual check in the module tests.
    P, L, U = la.lu(A, permute_l=False)  # type: ignore[reportAssignmentType]

    # 2. Apply the row permutation to b, then solve Ly = P.T @ b for y
    #    (forward substitution, L is lower triangular)
    y = la.solve_triangular(L, P.T @ b, lower=True)

    # 3. Solve Ux = y for x (back substitution, U is upper triangular)
    x = la.solve_triangular(U, y)
    return x


def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Description: Checks whether A is symmetric (A == A.T) within a tolerance.

    Parameters:
    -----------
    A : np.ndarray, shape (n, n)
        Matrix to check.

    tol : float
        Absolute tolerance passed to np.allclose.

    Returns:
    --------
    bool
        True if A is symmetric within tol, False otherwise.
    """
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Description: Checks whether A is positive-definite by attempting a Cholesky
        factorization (np.linalg.cholesky raises if it isn't).

    Parameters:
    -----------
    A : np.ndarray, shape (n, n)
        Matrix to check.

    Returns:
    --------
    bool
        True if A is positive-definite, False otherwise.
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def ll_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Description: Solves the linear system Ax = b using LL^T decomposition (Cholesky).

    Parameters:
    -----------
    A : np.ndarray, shape (n, n)
        Square coefficient matrix.
        Note: Cholesky requires A to be symmetric AND positive-definite.
        This is a stricter requirement than lu_solve's — decide where/how
        to check and raise a clear error if A doesn't qualify (similar to
        the rank-deficiency check you added to qr_solve).

    b : np.ndarray, shape (n,)
        Right-hand side vector.

    Returns:
    --------
    x : np.ndarray, shape (n,)
        Solution vector.
    """
    # Check that A is symmetric and positive-definite
    if not is_symmetric(A) or not is_positive_definite(A):
        raise ValueError(
            f"ll_solve requires A to be symmetric positive-definite, "
            f"got shape {A.shape} which failed that check"
        )

    """
    Solution: Cholesky (LL^T) Factorization

    Why Cholesky over general LU when A qualifies?
        - Exploiting symmetry halves the factorization work relative to LU
            (~n^3/3 flops vs ~2n^3/3), since only one triangular factor L is
            computed instead of separate L and U.
        - No pivoting is needed for a positive-definite A, so the factorization
            is both cheaper and numerically stable without row swaps.
    """

    # 1. Decompose A = L Lᵀ (via your own elimination, or scipy.linalg.cholesky)
    L = la.cholesky(A, lower=True)

    # 2. Solve Ly = b for y (forward substitution, L is lower triangular)
    y = la.solve_triangular(L, b, lower=True)

    # 3. Solve Lᵀx = y for x (back substitution against the transpose of L,
    #    which is upper triangular; trans=1 selects the 'T' system a^T x = b)
    x = la.solve_triangular(L, y, lower=True, trans=1)
    return x
