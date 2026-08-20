"""
Title: gd_solver.py

Author: HovarAK
Date: 2026-08-20
Description: Linear least-squares solver based on (batch) Gradient Descent for the
    problem Ax = b, minimizing f(x) = (1/2) * ||Ax - b||^2.
"""

import numpy as np


def _gradient(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of f(x) = (1/2) * ||Ax - b||^2 with respect to x.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Design matrix.
    b : np.ndarray, shape (m,)
        Target vector.
    x : np.ndarray, shape (n,)
        Current parameter estimate.

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient vector: Aᵀ(Ax - b)
    """
    # Residual r = Ax - b (shape (m,)); the gradient of (1/2)||Ax-b||^2 w.r.t. x
    # is A^T @ r by the chain rule.
    residual = A @ x - b
    return A.T @ residual


def gd_solve(
    A: np.ndarray,
    b: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Solve the linear least squares problem Ax = b using gradient descent.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Design matrix (features).
    b : np.ndarray, shape (m,)
        Target vector.
    lr : float, default=0.01
        Learning rate (step size per iteration).
    max_iters : int, default=1000
        Maximum number of iterations before stopping, even if not converged.
    tol : float, default=1e-6
        Convergence threshold. Stop early once progress between
        iterations falls below this value.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Approximate solution vector minimizing ||Ax - b||^2.
    """
    # Verifies the Pre-Requirment on Matrix A: must be 2-D to define a linear system
    if A.ndim != 2:
        raise ValueError(f"gd_solve requires A to be a 2-D matrix, got shape {A.shape}")

    # Verifies the Pre-Requirment on Vector b: must be 1-D to match Ax
    if b.ndim != 1:
        raise ValueError(f"gd_solve requires b to be a 1-D vector, got shape {b.shape}")

    # Verifies A isn't degenerate (no rows/columns) — nothing to solve otherwise
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError(f"gd_solve requires a non-empty A, got shape {A.shape}")

    # Verifies the Pre-Requirment on Matrix A and Vector b
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for gd_solve: {A.shape} and {b.shape}"
            f" (first dimension of A must match first dim of b)"
        )

    """
    Solution: Batch Gradient Descent

    Why gradient descent over a direct factorization (LU/QR/Cholesky)?
        - Each iteration only costs O(mn) (a matrix-vector product forward and
            back through A), versus the O(mn^2) or O(n^3) a one-shot
            factorization needs, so it scales to very large or sparse systems
            where even forming/factoring A directly would be too expensive.
        - The tradeoff is that it's iterative and approximate: whether/how fast
            it converges depends on lr and the conditioning of AᵀA, unlike the
            factorization solvers which are exact (up to floating-point error)
            in a fixed number of steps.
    """

    # 1. Initialize x (zeros, shape (n,))
    n = A.shape[1]
    x = np.zeros(n, dtype=float)

    # 2. Loop up to max_iters, taking a step down the gradient each time
    for _ in range(max_iters):
        grad = _gradient(A, b, x)
        x_new = x - lr * grad

        # Convergence check: stop early once the step itself (the progress
        # made this iteration) is smaller than tol — further iterations
        # wouldn't move x meaningfully.
        step_size = np.linalg.norm(x_new - x)
        x = x_new
        if step_size < tol:
            break

    # 3. Return x
    return x


if __name__ == "__main__":
    A_test = np.array([[1, 1], [1, 2], [1, 3]])
    b_test = np.array([6, 8, 10])
    x = gd_solve(A_test, b_test)
    print(x)
