"""Project 01: linear regression and regularization, from scratch.

Conventions for every function:
    X : np.ndarray of shape (n, p) — features (no column of ones; handle the intercept yourself)
    y : np.ndarray of shape (n,)   — target
    returns (intercept, coef) where intercept is a float and coef has shape (p,)

Use only NumPy here. scikit-learn is for checking your answers (see ../tests), not for solving them.
"""
import numpy as np


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Ordinary least squares, solved exactly.

    Minimizes ||y - X b - b0||^2.
    """
    raise NotImplementedError


def ols_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 5000
) -> tuple[float, np.ndarray]:
    """Ordinary least squares, solved iteratively with batch gradient descent.

    Minimizes the mean squared error (1/n) * ||y - X b - b0||^2.
    lr is the step size; n_iters is the number of steps. Start from all zeros.
    """
    raise NotImplementedError


def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Ridge regression, solved exactly.

    Minimizes ||y - X b - b0||^2 + alpha * ||b||^2.
    The intercept b0 is NOT penalized (same convention as sklearn.linear_model.Ridge).
    """
    raise NotImplementedError


def soft_threshold(z: float, gamma: float) -> float:
    """The soft-thresholding operator used by lasso.

    Moves z toward zero by gamma, and returns exactly 0 if that would cross zero.
    """
    raise NotImplementedError


def lasso_coordinate_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, n_iters: int = 1000, tol: float = 1e-6
) -> tuple[float, np.ndarray]:
    """Lasso regression, solved with cyclic coordinate descent.

    Minimizes (1/(2n)) * ||y - X b - b0||^2 + alpha * ||b||_1
    (same objective as sklearn.linear_model.Lasso). The intercept b0 is NOT penalized.
    Stop after n_iters full passes over the coefficients, or earlier once the largest
    coefficient change in a pass is below tol.
    """
    raise NotImplementedError
