"""
Tests for lu_solver.py

Covers: input validation, correctness against known/reference solutions
(np.linalg.solve), the row-pivoting case lu_solve depends on for stability,
the is_symmetric/is_positive_definite guards behind ll_solve, and basic
non-mutation/shape guarantees.
"""

import numpy as np
import pytest

from src.lu_solver import is_positive_definite, is_symmetric, ll_solve, lu_solve

# ---------------------------------------------------------------------------
# lu_solve: input validation
# ---------------------------------------------------------------------------


def test_lu_solve_raises_on_non_square_matrix():
    A = np.ones((3, 2))
    b = np.ones(3)
    with pytest.raises(ValueError, match="square"):
        lu_solve(A, b)


def test_lu_solve_raises_on_incompatible_shapes():
    A = np.eye(3)
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        lu_solve(A, b)


def test_lu_solve_accepts_matching_shapes_without_error():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    # should not raise
    lu_solve(A, b)


# ---------------------------------------------------------------------------
# lu_solve: basic / exact solves
# ---------------------------------------------------------------------------


def test_lu_solve_identity_matrix_returns_b():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 0.0])
    x = lu_solve(A, b)
    np.testing.assert_allclose(x, b, atol=1e-10)


def test_lu_solve_scalar_1x1_system():
    A = np.array([[2.0]])
    b = np.array([4.0])
    x = lu_solve(A, b)
    np.testing.assert_allclose(x, [2.0], atol=1e-10)


def test_lu_solve_exact_square_system_matches_known_solution():
    # x = [1, 2] solves this system exactly
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true
    x = lu_solve(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-10)


def test_lu_solve_output_shape_matches_number_of_unknowns():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    b = rng.standard_normal(6)
    x = lu_solve(A, b)
    assert x.shape == (6,)


# ---------------------------------------------------------------------------
# lu_solve: requires row pivoting (a zero on the natural diagonal)
# ---------------------------------------------------------------------------


def test_lu_solve_handles_zero_leading_pivot():
    # A[0, 0] == 0, so unpivoted elimination would divide by zero here;
    # this is exactly the case partial pivoting (and applying P to b) exists for.
    A = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 4.0], [3.0, 0.0, 1.0]])
    x_true = np.array([1.0, 2.0, -1.0])
    b = A @ x_true
    x = lu_solve(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-8)


def test_lu_solve_matches_numpy_solve_when_pivoting_required():
    A = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 4.0], [3.0, 0.0, 1.0]])
    b = np.array([5.0, 10.0, 8.0])
    x = lu_solve(A, b)
    x_ref = np.linalg.solve(A, b)
    np.testing.assert_allclose(x, x_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# lu_solve: random well-conditioned systems vs. reference solver
# ---------------------------------------------------------------------------


def test_lu_solve_matches_numpy_solve_on_random_systems():
    rng = np.random.default_rng(1)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((8, 8)) + 8 * np.eye(8)  # diagonally dominant
        b = rng.standard_normal(8)

        x = lu_solve(A, b)
        x_ref = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, x_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# lu_solve: non-mutation of inputs
# ---------------------------------------------------------------------------


def test_lu_solve_inputs_are_not_mutated():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    A_copy, b_copy = A.copy(), b.copy()

    lu_solve(A, b)

    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


# ---------------------------------------------------------------------------
# is_symmetric / is_positive_definite helpers
# ---------------------------------------------------------------------------


def test_is_symmetric_true_for_symmetric_matrix():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert is_symmetric(A) is True


def test_is_symmetric_false_for_asymmetric_matrix():
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    assert is_symmetric(A) is False


def test_is_positive_definite_true_for_spd_matrix():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    assert is_positive_definite(A) is True


def test_is_positive_definite_false_for_indefinite_matrix():
    # Symmetric but has a negative eigenvalue
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert is_positive_definite(A) is False


# ---------------------------------------------------------------------------
# ll_solve: input validation
# ---------------------------------------------------------------------------


def test_ll_solve_raises_on_asymmetric_matrix():
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="symmetric positive-definite"):
        ll_solve(A, b)


def test_ll_solve_raises_on_symmetric_indefinite_matrix():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, not PD
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="symmetric positive-definite"):
        ll_solve(A, b)


def test_ll_solve_accepts_spd_matrix_without_error():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    # should not raise
    ll_solve(A, b)


# ---------------------------------------------------------------------------
# ll_solve: basic / exact solves
# ---------------------------------------------------------------------------


def test_ll_solve_identity_matrix_returns_b():
    A = np.eye(4)
    b = np.array([1.0, -2.0, 3.5, 0.0])
    x = ll_solve(A, b)
    np.testing.assert_allclose(x, b, atol=1e-10)


def test_ll_solve_scalar_1x1_system():
    A = np.array([[2.0]])
    b = np.array([4.0])
    x = ll_solve(A, b)
    np.testing.assert_allclose(x, [2.0], atol=1e-10)


def test_ll_solve_exact_spd_system_matches_known_solution():
    # x = [1, 2] solves this SPD system exactly
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true
    x = ll_solve(A, b)
    np.testing.assert_allclose(x, x_true, atol=1e-10)


def test_ll_solve_output_shape_matches_number_of_unknowns():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((5, 5))
    A = M @ M.T + 5 * np.eye(5)  # guaranteed SPD
    b = rng.standard_normal(5)
    x = ll_solve(A, b)
    assert x.shape == (5,)


# ---------------------------------------------------------------------------
# ll_solve: random SPD systems vs. reference solver and lu_solve
# ---------------------------------------------------------------------------


def test_ll_solve_matches_numpy_solve_on_random_spd_systems():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((6, 6))
        A = M @ M.T + 6 * np.eye(6)  # guaranteed SPD
        b = rng.standard_normal(6)

        x = ll_solve(A, b)
        x_ref = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, x_ref, atol=1e-8)


def test_ll_solve_agrees_with_lu_solve_on_spd_system():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((6, 6))
    A = M @ M.T + 6 * np.eye(6)  # guaranteed SPD
    b = rng.standard_normal(6)

    x_ll = ll_solve(A, b)
    x_lu = lu_solve(A, b)
    np.testing.assert_allclose(x_ll, x_lu, atol=1e-8)


# ---------------------------------------------------------------------------
# ll_solve: non-mutation of inputs
# ---------------------------------------------------------------------------


def test_ll_solve_inputs_are_not_mutated():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    A_copy, b_copy = A.copy(), b.copy()

    ll_solve(A, b)

    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
