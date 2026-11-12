# Project 01 — Regression & Regularization

**Time:** about a week of work, at your own pace. **Code you write:** `src/regression.py`.
**Done when:** all tests pass, `INTUITION.md` and `README.md` are filled in, and the three plots exist in the notebook.

## Why this project first

Penalized regression is at the heart of your internship pipeline: lasso, elastic-net and ridge Cox models are
these same penalties attached to a different loss. If you can derive and code plain ridge and lasso, the Cox
versions in Project 04 are a small step. You will also meet the ideas that come up in almost every DS interview:
least squares, gradient descent, the bias–variance trade-off, feature scaling, and why lasso selects features.

## Learning goals

By the end you should be able to explain, without notes:

1. What least squares minimizes, and where the closed-form solution (the "normal equations") comes from.
2. How the intercept is handled, and why centering X and y makes it simple.
3. What gradient descent is doing, how the learning rate matters, and **why unscaled features make it struggle**.
4. Why the ridge penalty shrinks coefficients, and why it fixes the "singular matrix" problem with collinear features.
5. Why lasso sets some coefficients **exactly** to zero while ridge only makes them small.
6. Why the intercept is never penalized.

## Dataset

`sklearn.datasets.load_diabetes`: 442 patients, 10 standardized baseline features (age, BMI, blood pressure,
six blood serum measurements), and a target measuring disease progression one year later. It is small, real,
and has strongly correlated features (look at `s1` and `s2`), which is exactly what makes regularization interesting.

## Starting analogies (your job: find where each one breaks)

- **OLS:** every data point is attached to the regression line by a spring. Longer springs pull harder. The line
  settles where all the pulls balance.
- **Ridge:** each coefficient is also on a leash tied to zero. The stiffer the leash (bigger alpha), the less the
  data can drag it away.
- **Lasso:** each coefficient must pay a fixed toll to be non-zero at all. Coefficients whose "pull" from the data
  is weaker than the toll can't afford to stay, so they drop out completely.

Write these up in `INTUITION.md` as you go. "Where it breaks" is the most important part.

## Milestones

Work in order. Each milestone turns specific tests green. Run tests from the repo root with
`.venv/Scripts/python -m pytest -v` (add `-k name` to run just one).

### 1. OLS, closed form
Implement `ols_closed_form`. Derive the solution on paper first: take the gradient of the squared error,
set it to zero. Then think about the intercept.
- Green: `test_ols_closed_form_matches_sklearn`
- Checkpoint: why is `np.linalg.solve` (or `lstsq`) preferred over `np.linalg.inv`?

### 2. OLS, gradient descent
Implement `ols_gradient_descent`. Derive the gradient of the mean squared error with respect to `b` and `b0`.
- Green: `test_ols_gradient_descent_matches_closed_form`
- In the notebook: **plot the loss curve** (MSE vs iteration) for a few learning rates. Then run it on the
  **unscaled** diabetes data (`load_diabetes(scaled=False)`) and see what happens. Explain why.

### 3. Ridge
Implement `ridge_closed_form`. Add the penalty to your milestone-1 derivation. Be careful not to penalize the intercept.
- Green: `test_ridge_matches_sklearn` (3 cases), `test_ridge_alpha_zero_equals_ols`, `test_ridge_handles_duplicate_column`
- In the notebook: **plot the ridge coefficient path**: each coefficient vs alpha on a log scale (try `np.logspace(-2, 4, 50)`).

### 4. Lasso
Implement `soft_threshold`, then `lasso_coordinate_descent`. The idea of coordinate descent: hold every coefficient
fixed except one, solve for that one exactly, move to the next, and repeat. Work out on paper what "solve for one
coefficient exactly" means when the objective includes `alpha * |b_j|`, and where soft-thresholding comes from.
- Green: `test_soft_threshold`, `test_lasso_matches_sklearn` (2 cases), `test_lasso_large_alpha_zeroes_everything`, `test_lasso_produces_sparsity`
- In the notebook: **plot the lasso path** next to the ridge path. Which features drop out first? Does that match
  what you'd expect from `s1` and `s2` being correlated?

### 5. Write-up
Fill in `README.md` (question, what you built, results with the plots, what surprised you, interview explanation)
and finish `INTUITION.md`. Update the project status in the root README.

## Stretch goals

- **Elastic net:** combine both penalties in your coordinate descent. This is the penalty your internship Cox models used.
- **Choosing alpha:** pick alpha with 5-fold cross-validation, written by hand. This leads straight into Project 02.
- **Interview drill:** explain the geometric picture (the L1 diamond vs the L2 circle touching the loss contours)
  and why it implies sparsity.

## Rules of engagement

Ask Claude for hints, not code. See `CLAUDE.md` at the repo root.
