"""
Tests for svd_solver.py

Covers: input validation, correctness against known/reference solutions
(np.linalg.lstsq), the over-/under-determined and rank-deficient cases the
TSVD approach is specifically meant to handle, the rcond truncation
behavior, and basic non-mutation/shape guarantees.
"""

import numpy as np
import pytest

from src.svd_solver import svd_solver

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_incompatible_shapes():
    A = np.ones((5, 3))
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        svd_solver(A, b)


def test_accepts_matching_shapes_without_error():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    # should not raise
    svd_solver(A, b)


# ---------------------------------------------------------------------------
# Basic / exact solves
# ---------------------------------------------------------------------------


def test_identity_matrix_returns_b():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 0.0])
    x = svd_solver(A, b)
    np.testing.assert_allclose(x, b, atol=1e-10)


def test_scalar_1x1_system():
    A = np.array([[2.0]])
    b = np.array([4.0])
    x = svd_solver(A, b)
    np.testing.assert_allclose(x, [2.0], atol=1e-10)


def test_exact_square_system_matches_known_solution():
    # x = [1, 2] solves this system exactly
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true
    x = svd_solver(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-10)


def test_output_shape_matches_number_of_columns():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 4))
    b = rng.standard_normal(10)
    x = svd_solver(A, b)
    assert x.shape == (4,)


# ---------------------------------------------------------------------------
# Well-conditioned overdetermined / underdetermined vs. reference solver
# ---------------------------------------------------------------------------


def test_overdetermined_matches_numpy_lstsq():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((50, 5))
    b = rng.standard_normal(50)

    x = svd_solver(A, b)
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)

    np.testing.assert_allclose(x, x_ref, atol=1e-8)


def test_underdetermined_matches_numpy_minimum_norm_solution():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 8))  # fewer rows than columns
    b = rng.standard_normal(3)

    x = svd_solver(A, b)
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)

    # TSVD produces the minimum-norm solution, same as np.linalg.lstsq
    np.testing.assert_allclose(x, x_ref, atol=1e-8)
    np.testing.assert_allclose(A @ x, b, atol=1e-8)


def test_solution_minimizes_residual_norm():
    # For an overdetermined, well-conditioned system, no small perturbation
    # of x should reduce the residual below what svd_solver achieves.
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 4))
    b = rng.standard_normal(30)

    x = svd_solver(A, b)
    base_residual = np.linalg.norm(A @ x - b)

    for _ in range(20):
        perturbation = rng.standard_normal(4) * 1e-4
        perturbed_residual = np.linalg.norm(A @ (x + perturbation) - b)
        assert perturbed_residual >= base_residual - 1e-10


# ---------------------------------------------------------------------------
# Rank-deficient matrices (the case TSVD is specifically designed for)
# ---------------------------------------------------------------------------


def test_rank_deficient_matrix_gives_minimum_norm_solution():
    # Column 2 is exactly 2x column 1 -> rank 1, one singular value ~ 0
    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    b = np.array([1.0, 2.0, 3.0])

    x = svd_solver(A, b)

    # x should lie in the row space of A, i.e. be orthogonal to the
    # null-space direction [2, -1]
    null_vec = np.array([2.0, -1.0])
    assert abs(np.dot(x, null_vec)) < 1e-8

    # and it should still (approximately) solve the consistent system
    np.testing.assert_allclose(A @ x, b, atol=1e-8)


def test_singular_square_matrix_does_not_raise():
    # A singular 3x3 matrix (row3 = row1 + row2)
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]])
    b = np.array([1.0, 2.0, 3.0])
    # Should not raise (unlike e.g. np.linalg.solve, which would)
    x = svd_solver(A, b)
    assert x.shape == (3,)
    assert np.all(np.isfinite(x))


def test_zero_matrix_returns_zero_vector():
    A = np.zeros((4, 3))
    b = np.array([1.0, 2.0, 3.0, 4.0])
    x = svd_solver(A, b)
    np.testing.assert_allclose(x, np.zeros(3), atol=1e-10)


# ---------------------------------------------------------------------------
# rcond truncation behavior
# ---------------------------------------------------------------------------


def _make_matrix_with_singular_values(singular_values, m, n, seed):
    """Builds an (m, n) matrix with a prescribed singular value spectrum."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, len(singular_values))))
    V, _ = np.linalg.qr(rng.standard_normal((n, len(singular_values))))
    return U @ np.diag(singular_values) @ V.T


def test_small_rcond_keeps_small_singular_value_component():
    A = _make_matrix_with_singular_values([1.0, 1e-8], m=5, n=2, seed=4)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(5)

    x_keep = svd_solver(A, b, rcond=1e-10)  # tol=1e-10 < 1e-8, sv kept
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=1e-10)

    np.testing.assert_allclose(x_keep, x_ref, atol=1e-6)


def test_large_rcond_truncates_small_singular_value_component():
    A = _make_matrix_with_singular_values([1.0, 1e-8], m=5, n=2, seed=4)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(5)

    x_keep = svd_solver(A, b, rcond=1e-10)
    x_truncated = svd_solver(A, b, rcond=1e-6)  # tol=1e-6 > 1e-8, sv dropped

    # Truncating a component should change the solution and should not
    # increase the solution norm (TSVD favors the smaller-norm solution).
    assert not np.allclose(x_keep, x_truncated)
    assert np.linalg.norm(x_truncated) <= np.linalg.norm(x_keep) + 1e-8


# ---------------------------------------------------------------------------
# Non-mutation of inputs
# ---------------------------------------------------------------------------


def test_inputs_are_not_mutated():
    A = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    A_copy, b_copy = A.copy(), b.copy()

    svd_solver(A, b)

    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
