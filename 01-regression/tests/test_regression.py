"""Tests for Project 01. Each one compares your from-scratch code with scikit-learn.

Run from the repo root:  .venv/Scripts/python -m pytest -v
They all fail at first (NotImplementedError) and turn green as you finish each milestone.
"""
import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from regression import (
    lasso_coordinate_descent,
    ols_closed_form,
    ols_gradient_descent,
    ridge_closed_form,
    soft_threshold,
)


@pytest.fixture(scope="module")
def data():
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)  # standardized features, raw (uncentered) target
    return X, y


def mse(X, y, intercept, coef):
    return np.mean((y - X @ coef - intercept) ** 2)


# --- Milestone 1: OLS closed form -------------------------------------------

def test_ols_closed_form_matches_sklearn(data):
    X, y = data
    intercept, coef = ols_closed_form(X, y)
    ref = LinearRegression().fit(X, y)
    assert np.allclose(coef, ref.coef_, atol=1e-6)
    assert intercept == pytest.approx(ref.intercept_, abs=1e-6)


# --- Milestone 2: gradient descent -------------------------------------------

def test_ols_gradient_descent_matches_closed_form(data):
    # Diabetes features are strongly correlated, so coefficients converge slowly.
    # We compare how well the model fits (MSE), not the coefficients themselves.
    X, y = data
    ref = LinearRegression().fit(X, y)
    intercept, coef = ols_gradient_descent(X, y)
    assert mse(X, y, intercept, coef) <= 1.01 * mse(X, y, ref.intercept_, ref.coef_)


# --- Milestone 3: ridge -------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_ridge_matches_sklearn(data, alpha):
    X, y = data
    intercept, coef = ridge_closed_form(X, y, alpha)
    ref = Ridge(alpha=alpha).fit(X, y)
    assert np.allclose(coef, ref.coef_, atol=1e-6)
    assert intercept == pytest.approx(ref.intercept_, abs=1e-6)


def test_ridge_alpha_zero_equals_ols(data):
    X, y = data
    b0_r, coef_r = ridge_closed_form(X, y, 0.0)
    b0_o, coef_o = ols_closed_form(X, y)
    assert np.allclose(coef_r, coef_o, atol=1e-6)
    assert b0_r == pytest.approx(b0_o, abs=1e-6)


def test_ridge_handles_duplicate_column(data):
    # A copied column makes X^T X singular. Ridge with alpha > 0 should still work.
    X, y = data
    X_dup = np.column_stack([X, X[:, 0]])
    _, coef = ridge_closed_form(X_dup, y, 1.0)
    assert np.all(np.isfinite(coef))


# --- Milestone 4: lasso --------------------------------------------------------

def test_soft_threshold():
    assert soft_threshold(3.0, 1.0) == pytest.approx(2.0)
    assert soft_threshold(-3.0, 1.0) == pytest.approx(-2.0)
    assert soft_threshold(0.5, 1.0) == 0.0


@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_lasso_matches_sklearn(data, alpha):
    X, y = data
    intercept, coef = lasso_coordinate_descent(X, y, alpha)
    ref = Lasso(alpha=alpha, tol=1e-10, max_iter=100_000).fit(X, y)
    assert np.allclose(coef, ref.coef_, atol=1e-4)
    assert intercept == pytest.approx(ref.intercept_, abs=1e-4)


def test_lasso_large_alpha_zeroes_everything(data):
    X, y = data
    intercept, coef = lasso_coordinate_descent(X, y, 1e4)
    assert np.all(coef == 0)
    assert intercept == pytest.approx(y.mean())


def test_lasso_produces_sparsity(data):
    X, y = data
    _, coef = lasso_coordinate_descent(X, y, 1.0)
    assert np.sum(coef == 0) >= 1
