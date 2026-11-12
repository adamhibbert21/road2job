# road2job

Learning data science — and, longer term, quantitative research — by building the core methods from scratch
in NumPy, then checking them against the standard libraries (scikit-learn, lifelines, statsmodels).

Each project starts from an intuition, implements the method by hand, verifies it with tests against a reference
implementation, and ends with a write-up of results and what I learned.

## Projects

| # | Project | Phase | Status |
|---|---------|-------|--------|
| 01 | [Regression & regularization](01-regression/) — OLS, gradient descent, ridge, lasso | Foundations | In progress |
| 02 | Model evaluation — nested CV, leakage, bootstrap, permutation tests | Foundations | Planned |
| 03 | Clustering & PCA — k-means, PCA via SVD, silhouette, choosing k | Foundations | Planned |
| 04 | Survival analysis — Kaplan–Meier, Cox partial likelihood, C-index | Foundations | Planned |
| 05 | Trees & ensembles — decision tree, random forest, gradient boosting, SHAP | Foundations | Planned |
| 06 | SQL + pandas analysis of a messy real dataset | Data science toolkit | Planned |
| 07 | A/B testing & causal inference — power by simulation | Data science toolkit | Planned |
| 08 | End-to-end data science project | Data science toolkit | Planned |
| 09 | Returns, volatility & time-series statistics | Quant research | Planned |
| 10 | Backtester from scratch — avoiding look-ahead bias | Quant research | Planned |
| 11 | Portfolio construction / pairs trading study | Quant research | Planned |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pytest
```

Python 3.14.
