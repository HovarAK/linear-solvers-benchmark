"""
Tests for metrics.py

Covers: mse/mae/r_squared correctness and input validation, residual_norm/
relative_norm correctness (including against an actual solver's output) and
input validation, and the zero-variance / zero-b edge cases both modules
guard against.
"""

import numpy as np
import pytest

from src.lu_solver import lu_solve
from src.metrics import mae, mse, r_squared, relative_norm, residual_norm

# ---------------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------------


def test_mse_zero_for_identical_arrays():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_matches_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    # mean of [1, 4, 9] = 14/3
    np.testing.assert_allclose(mse(y_true, y_pred), 14.0 / 3.0)


def test_mse_is_symmetric_in_its_arguments():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 4.0])
    assert mse(y_true, y_pred) == mse(y_pred, y_true)


def test_mse_raises_on_shape_mismatch():
    y_true = np.ones(3)
    y_pred = np.ones(4)
    with pytest.raises(ValueError, match="same shape"):
        mse(y_true, y_pred)


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------


def test_mae_zero_for_identical_arrays():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_matches_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -2.0, 3.0])
    # mean of [1, 2, 3] = 2
    np.testing.assert_allclose(mae(y_true, y_pred), 2.0)


def test_mae_raises_on_shape_mismatch():
    y_true = np.ones(3)
    y_pred = np.ones(4)
    with pytest.raises(ValueError, match="same shape"):
        mae(y_true, y_pred)


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------


def test_r_squared_is_one_for_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y_true, y_true) == pytest.approx(1.0)


def test_r_squared_matches_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    # ss_res = (0^2 + 0^2 + 1^2) = 1, ss_tot = sum((y_true - 2)^2) = 1+0+1 = 2
    np.testing.assert_allclose(r_squared(y_true, y_pred), 1 - 1.0 / 2.0)


def test_r_squared_is_negative_when_worse_than_predicting_the_mean():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, -10.0, 20.0])
    assert r_squared(y_true, y_pred) < 0


def test_r_squared_raises_on_shape_mismatch():
    y_true = np.ones(3)
    y_pred = np.ones(4)
    with pytest.raises(ValueError, match="same shape"):
        r_squared(y_true, y_pred)


def test_r_squared_raises_when_y_true_has_zero_variance():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="zero variance"):
        r_squared(y_true, y_pred)


# ---------------------------------------------------------------------------
# residual_norm: correctness
# ---------------------------------------------------------------------------


def test_residual_norm_zero_for_exact_solution():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true

    x = lu_solve(A, b)
    np.testing.assert_allclose(residual_norm(A, x, b), 0.0, atol=1e-10)


def test_residual_norm_matches_known_value():
    A = np.eye(2)
    x = np.array([1.0, 1.0])
    b = np.array([1.0, 4.0])
    # A @ x - b = [0, -3], norm = 3
    np.testing.assert_allclose(residual_norm(A, x, b), 3.0)


def test_residual_norm_matches_manual_computation_on_random_system():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    x = rng.standard_normal(3)
    b = rng.standard_normal(5)

    expected = np.linalg.norm(A @ x - b)
    np.testing.assert_allclose(residual_norm(A, x, b), expected)


# ---------------------------------------------------------------------------
# residual_norm: input validation
# ---------------------------------------------------------------------------


def test_residual_norm_raises_on_incompatible_x_shape():
    A = np.eye(3)
    x = np.ones(4)  # wrong length
    b = np.ones(3)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        residual_norm(A, x, b)


def test_residual_norm_raises_on_incompatible_b_shape():
    A = np.eye(3)
    x = np.ones(3)
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        residual_norm(A, x, b)


def test_residual_norm_accepts_1d_x_without_error():
    # regression check: x must be treated as a 1-D vector (n,), matching
    # what every solver in this repo actually returns.
    A = np.eye(3)
    x = np.ones(3)
    b = np.ones(3)
    # should not raise
    residual_norm(A, x, b)


# ---------------------------------------------------------------------------
# relative_norm: correctness
# ---------------------------------------------------------------------------


def test_relative_norm_zero_for_exact_solution():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x_true = np.array([1.0, 2.0])
    b = A @ x_true

    x = lu_solve(A, b)
    np.testing.assert_allclose(relative_norm(A, x, b), 0.0, atol=1e-10)


def test_relative_norm_matches_residual_norm_divided_by_norm_b():
    A = np.eye(2)
    x = np.array([1.0, 1.0])
    b = np.array([1.0, 4.0])

    expected = residual_norm(A, x, b) / np.linalg.norm(b)
    np.testing.assert_allclose(relative_norm(A, x, b), expected)


# ---------------------------------------------------------------------------
# relative_norm: input validation
# ---------------------------------------------------------------------------


def test_relative_norm_raises_on_incompatible_x_shape():
    A = np.eye(3)
    x = np.ones(4)  # wrong length
    b = np.ones(3)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        relative_norm(A, x, b)


def test_relative_norm_raises_on_incompatible_b_shape():
    A = np.eye(3)
    x = np.ones(3)
    b = np.ones(4)  # wrong length
    with pytest.raises(ValueError, match="Incompatible shapes"):
        relative_norm(A, x, b)


def test_relative_norm_raises_when_b_is_zero_vector():
    A = np.eye(3)
    x = np.ones(3)
    b = np.zeros(3)
    with pytest.raises(ValueError, match="zero vector"):
        relative_norm(A, x, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
