"""
Tests for gd_solver.py

Covers: the _gradient helper directly, input validation, exact-solve behavior
on simple well-conditioned systems, the tol/max_iters convergence controls,
correctness against reference solvers (np.linalg.lstsq, lu_solve) on random
well-conditioned systems using a learning rate scaled to the matrix's
spectrum, and basic non-mutation/shape guarantees.
"""

import numpy as np
import pytest

from src.gd_solver import _gradient, gd_solve
from src.lu_solver import lu_solve


def _safe_lr(A: np.ndarray, margin: float = 0.5) -> float:
    """
    Largest stable step size for gradient descent on (1/2)||Ax-b||^2 is
    2 / lambda_max(AᵀA). Scale by `margin` to stay comfortably inside that
    bound so tests converge without oscillating/diverging.
    """
    eigmax = np.max(np.linalg.eigvalsh(A.T @ A))
    return float(margin * 2.0 / eigmax)


# ---------------------------------------------------------------------------
# _gradient: direct correctness
# ---------------------------------------------------------------------------


def test_gradient_matches_direct_formula():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    b = rng.standard_normal(5)
    x = rng.standard_normal(3)

    grad = _gradient(A, b, x)
    expected = A.T @ (A @ x - b)
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_gradient_is_zero_at_least_squares_optimum():
    # At the normal-equations solution, A^T(Ax - b) == 0 by definition.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 3))
    b = rng.standard_normal(6)
    x_star = np.linalg.lstsq(A, b, rcond=None)[0]

    grad = _gradient(A, b, x_star)
    np.testing.assert_allclose(grad, np.zeros(3), atol=1e-8)


def test_gradient_output_shape():
    A = np.ones((4, 2))
    b = np.ones(4)
    x = np.zeros(2)
    assert _gradient(A, b, x).shape == (2,)


# ---------------------------------------------------------------------------
# gd_solve: input validation
# ---------------------------------------------------------------------------


def test_gd_solve_raises_on_incompatible_shapes():
    A = np.eye(3)
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        gd_solve(A, b)


def test_gd_solve_raises_on_non_2d_matrix():
    A = np.ones(3)  # 1-D, not a matrix
    b = np.ones(3)
    with pytest.raises(ValueError, match="2-D matrix"):
        gd_solve(A, b)


def test_gd_solve_raises_on_non_1d_vector():
    A = np.eye(3)
    b = np.ones((3, 1))  # column vector, not 1-D
    with pytest.raises(ValueError, match="1-D vector"):
        gd_solve(A, b)


def test_gd_solve_raises_on_empty_matrix():
    A = np.empty((0, 3))
    b = np.empty(0)
    with pytest.raises(ValueError, match="non-empty"):
        gd_solve(A, b)


def test_gd_solve_accepts_matching_shapes_without_error():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    # should not raise
    gd_solve(A, b, lr=0.5, max_iters=10)


# ---------------------------------------------------------------------------
# gd_solve: basic / exact solves on simple well-conditioned systems
# ---------------------------------------------------------------------------


def test_gd_solve_identity_matrix_converges_to_b():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 0.0])
    x = gd_solve(A, b, lr=0.5, max_iters=1000, tol=1e-12)
    np.testing.assert_allclose(x, b, atol=1e-6)


def test_gd_solve_scalar_1x1_system():
    A = np.array([[2.0]])
    b = np.array([4.0])
    x = gd_solve(A, b, lr=_safe_lr(A), max_iters=1000, tol=1e-12)
    np.testing.assert_allclose(x, [2.0], atol=1e-6)


def test_gd_solve_diagonal_system_matches_known_solution():
    A = np.diag([2.0, 4.0, 0.5])
    x_true = np.array([1.0, -1.0, 3.0])
    b = A @ x_true
    x = gd_solve(A, b, lr=_safe_lr(A), max_iters=5000, tol=1e-12)
    np.testing.assert_allclose(x, x_true, atol=1e-5)


def test_gd_solve_output_shape_matches_number_of_unknowns():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 4)) + 4 * np.eye(6, 4)
    b = rng.standard_normal(6)
    x = gd_solve(A, b, lr=_safe_lr(A), max_iters=50)
    assert x.shape == (4,)


# ---------------------------------------------------------------------------
# gd_solve: convergence controls (tol / max_iters / lr)
# ---------------------------------------------------------------------------


def test_gd_solve_lr_zero_never_moves_past_initial_x():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    x = gd_solve(A, b, lr=0.0, max_iters=100, tol=1e-9)
    np.testing.assert_allclose(x, np.zeros(3), atol=1e-12)


def test_gd_solve_max_iters_zero_returns_initial_zeros():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    x = gd_solve(A, b, lr=0.5, max_iters=0)
    np.testing.assert_allclose(x, np.zeros(3), atol=1e-12)


def test_gd_solve_loose_tol_stops_after_first_step():
    # With tol larger than the very first step's size, gd_solve should take
    # exactly one gradient step from x=0 and then stop, regardless of
    # max_iters. That first step is -lr * grad(A, b, 0) = lr * A^T @ b.
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([3.0, 4.0])
    lr = 0.1
    x = gd_solve(A, b, lr=lr, max_iters=1000, tol=1e10)
    expected_first_step = lr * (A.T @ b)
    np.testing.assert_allclose(x, expected_first_step, atol=1e-12)


def test_gd_solve_more_iterations_gets_closer_to_optimum():
    A = np.diag([1.0, 3.0])
    x_true = np.array([2.0, -1.0])
    b = A @ x_true
    lr = _safe_lr(A)

    x_few = gd_solve(A, b, lr=lr, max_iters=5, tol=0.0)
    x_many = gd_solve(A, b, lr=lr, max_iters=2000, tol=0.0)

    err_few = np.linalg.norm(x_few - x_true)
    err_many = np.linalg.norm(x_many - x_true)
    assert err_many < err_few


# ---------------------------------------------------------------------------
# gd_solve: correctness vs. reference solvers on random well-conditioned
# systems (learning rate scaled to the matrix so GD is guaranteed to converge)
# ---------------------------------------------------------------------------


def test_gd_solve_matches_lstsq_on_random_overdetermined_systems():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((10, 4)) + 6 * np.eye(10, 4)  # well-conditioned
        b = rng.standard_normal(10)

        x = gd_solve(A, b, lr=_safe_lr(A), max_iters=20000, tol=1e-14)
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(x, x_ref, atol=1e-4)


def test_gd_solve_matches_lu_solve_on_random_square_systems():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((5, 5)) + 8 * np.eye(5)  # diagonally dominant
        b = rng.standard_normal(5)

        x = gd_solve(A, b, lr=_safe_lr(A), max_iters=20000, tol=1e-14)
        x_ref = lu_solve(A, b)
        np.testing.assert_allclose(x, x_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# gd_solve: non-mutation of inputs
# ---------------------------------------------------------------------------


def test_gd_solve_inputs_are_not_mutated():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    A_copy, b_copy = A.copy(), b.copy()

    gd_solve(A, b, lr=_safe_lr(A), max_iters=100)

    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
