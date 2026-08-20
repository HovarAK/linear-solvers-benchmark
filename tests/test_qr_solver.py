"""
Tests for qr_solver.py

Covers: input validation, correctness against known/reference solutions
(np.linalg.lstsq), properties of the Modified Gram-Schmidt factorization
itself (orthonormality of Q, reconstruction A = QR, R upper triangular),
and basic non-mutation/shape guarantees.
"""

import numpy as np
import pytest
from src.qr_solver import m_gram_schmidt, qr_solve

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_incompatible_shapes():
    A = np.ones((5, 3))
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        qr_solve(A, b)


def test_accepts_matching_shapes_without_error():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    # should not raise
    qr_solve(A, b)


# ---------------------------------------------------------------------------
# Basic / exact solves
# ---------------------------------------------------------------------------


def test_identity_matrix_returns_b():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 0.0])
    x = qr_solve(A, b)
    np.testing.assert_allclose(x, b, atol=1e-10)


def test_scalar_1x1_system():
    A = np.array([[2.0]])
    b = np.array([4.0])
    x = qr_solve(A, b)
    np.testing.assert_allclose(x, [2.0], atol=1e-10)


def test_exact_square_system_matches_known_solution():
    # x = [1, 2] solves this system exactly
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true
    x = qr_solve(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-10)


def test_output_shape_matches_number_of_columns():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 4))
    b = rng.standard_normal(10)
    x = qr_solve(A, b)
    assert x.shape == (4,)


# ---------------------------------------------------------------------------
# Well-conditioned overdetermined system vs. reference solver
# ---------------------------------------------------------------------------


def test_overdetermined_matches_numpy_lstsq():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((50, 5))
    b = rng.standard_normal(50)

    x = qr_solve(A, b)
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)

    np.testing.assert_allclose(x, x_ref, atol=1e-8)


def test_solution_minimizes_residual_norm():
    # For an overdetermined, well-conditioned system, no small perturbation
    # of x should reduce the residual below what qr_solve achieves.
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 4))
    b = rng.standard_normal(30)

    x = qr_solve(A, b)
    base_residual = np.linalg.norm(A @ x - b)

    for _ in range(20):
        perturbation = rng.standard_normal(4) * 1e-4
        perturbed_residual = np.linalg.norm(A @ (x + perturbation) - b)
        assert perturbed_residual >= base_residual - 1e-10


# ---------------------------------------------------------------------------
# m_gram_schmidt factorization properties
# ---------------------------------------------------------------------------


def test_q_has_orthonormal_columns():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((20, 6))
    Q, _ = m_gram_schmidt(A)
    np.testing.assert_allclose(Q.T @ Q, np.eye(6), atol=1e-8)


def test_r_is_upper_triangular():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((10, 4))
    _, R = m_gram_schmidt(A)
    np.testing.assert_allclose(R, np.triu(R), atol=1e-12)


def test_qr_reconstructs_original_matrix():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((15, 5))
    Q, R = m_gram_schmidt(A)
    np.testing.assert_allclose(Q @ R, A, atol=1e-8)


def test_rank_deficient_matrix_raises():
    # Column 2 is exactly 2x column 1 -> rank 1, not full column rank
    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(ValueError, match="Rank-deficient"):
        m_gram_schmidt(A)


# ---------------------------------------------------------------------------
# Non-mutation of inputs
# ---------------------------------------------------------------------------


def test_inputs_are_not_mutated():
    A = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    A_copy, b_copy = A.copy(), b.copy()

    qr_solve(A, b)

    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
