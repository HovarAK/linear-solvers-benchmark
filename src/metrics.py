"""
Title: metrics.py

Author: HovarAK
Date: 2026-08-21
Description: Evaluation measures for regression, plus linear-system-specific
    diagnostics (residual/relative norms) for comparing the solvers in this repo.
"""

import numpy as np
import scipy.linalg as la


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Description: Computes the Mean Squared Error between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n,)
        Ground-truth values.

    y_pred : np.ndarray, shape (n,)
        Predicted values.

    Returns:
    --------
    float
        Mean of the squared elementwise errors.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"mse requires y_true and y_pred to have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )

    squared_error = (y_true - y_pred) ** 2
    return np.mean(squared_error)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Description: Computes the Mean Absolute Error between true and predicted values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n,)
        Ground-truth values.

    y_pred : np.ndarray, shape (n,)
        Predicted values.

    Returns:
    --------
    float
        Mean of the absolute elementwise errors.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"mae requires y_true and y_pred to have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )

    abs_error = np.abs(y_true - y_pred)
    return np.mean(abs_error)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Description: Computes the coefficient of determination R^2, i.e. the
        fraction of variance in y_true explained by y_pred.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n,)
        Ground-truth values.

    y_pred : np.ndarray, shape (n,)
        Predicted values.

    Returns:
    --------
    float
        R^2 score. 1.0 is a perfect fit; can be negative if the predictions
        are worse than just predicting the mean of y_true.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"r_squared requires y_true and y_pred to have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # ss_tot is 0 only when y_true is constant, in which case R^2 is undefined
    # (there's no variance for y_pred to explain).
    if np.isclose(ss_tot, 0):
        raise ValueError(
            "r_squared is undefined when y_true has zero variance "
            "(all elements are equal)"
        )

    return 1 - ss_res / ss_tot


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Description: Computes the absolute residual norm ||Ax - b|| for a solved
        linear system, i.e. how far Ax is from b in an absolute sense.

    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        Coefficient/design matrix.

    x : np.ndarray, shape (n,)
        Candidate solution vector.

    b : np.ndarray, shape (m,)
        Right-hand side vector.

    Returns:
    --------
    float
        The 2-norm of the residual A @ x - b.
    """
    m_rows, n_cols = A.shape

    if n_cols != x.shape[0]:
        raise ValueError(
            f"Incompatible shapes for residual_norm: A is {A.shape} but x is "
            f"{x.shape} (A's column count must match x's length)"
        )

    if m_rows != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for residual_norm: A is {A.shape} but b is "
            f"{b.shape} (A's row count must match b's length)"
        )

    return la.norm(A @ x - b)


def relative_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Description: Computes the relative residual norm ||Ax - b|| / ||b||, which
        scales the residual by the size of b so results are comparable across
        systems of different magnitude.

    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        Coefficient/design matrix.

    x : np.ndarray, shape (n,)
        Candidate solution vector.

    b : np.ndarray, shape (m,)
        Right-hand side vector.

    Returns:
    --------
    float
        The residual norm ||A @ x - b|| divided by ||b||.
    """
    m_rows, n_cols = A.shape

    if n_cols != x.shape[0]:
        raise ValueError(
            f"Incompatible shapes for relative_norm: A is {A.shape} but x is "
            f"{x.shape} (A's column count must match x's length)"
        )

    if m_rows != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for relative_norm: A is {A.shape} but b is "
            f"{b.shape} (A's row count must match b's length)"
        )

    # A zero b makes the ratio undefined (division by zero); the caller should
    # use residual_norm instead in that case.
    if np.allclose(b, 0):
        raise ValueError(
            "relative_norm is undefined when b is (numerically) the zero "
            "vector; use residual_norm instead"
        )

    return la.norm(A @ x - b) / la.norm(b)
