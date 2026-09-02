# linear-solvers-benchmark

Implementation and runtime/accuracy comparison of five approaches to solving linear systems and least-squares problems: **SVD**, **QR decomposition**, **LU decomposition**, **Cholesky (LLᵀ) decomposition**, and **gradient descent**.

## Problem

Given a linear system (or least-squares regression problem) `Ax = b`, there are multiple numerical methods to solve for `x`, each with different tradeoffs in speed, numerical stability, and scalability. This project implements each solver from first principles (using NumPy/SciPy building blocks, not `np.linalg.solve` shortcuts), benchmarks their runtime as problem size grows, and compares solution accuracy.

**Solvers implemented** (all in `src/`, unit-tested in `tests/`):

| Solver | Function | Applies to | Notes |
|---|---|---|---|
| SVD | `svd_solver(A, b, rcond=1e-10)` in `svd_solver.py` | any `A` (m×n) | truncated SVD pseudo-inverse; most numerically stable, most expensive |
| QR decomposition | `qr_solve(A, b)` in `qr_solver.py` | full column-rank `A` (m×n) | Modified Gram-Schmidt; good stability, cheaper than SVD |
| LU decomposition | `lu_solve(A, b)` in `lu_solver.py` | square `A` (n×n) | partial pivoting; fastest general square solve, less stable when ill-conditioned |
| Cholesky (LLᵀ) | `ll_solve(A, b)` in `lu_solver.py` | symmetric positive-definite `A` (n×n) | ~half the flops of LU by exploiting symmetry; raises if `A` doesn't qualify |
| Gradient descent | `gd_solve(A, b, lr=0.01, max_iters=1000, tol=1e-6)` in `gd_solver.py` | any `A` (m×n) | iterative, O(mn) per step; convergence depends on `lr` and the conditioning of AᵀA |

`lu_solver.py` also exposes `is_symmetric()` and `is_positive_definite()` helper checks used to validate inputs to `ll_solve`.

## Dataset

Planned benchmark datasets, to show how each solver scales rather than relying on a single data point:

- `sklearn.datasets.make_regression()` — synthetic data, lets us control rows/features independently to isolate scaling behavior
- `sklearn.datasets.fetch_california_housing()` — real-valued regression, mid-sized
- `sklearn.datasets.load_diabetes()` — smaller real-world dataset for comparison

*(Not yet wired into a benchmark script — see Next Goal below.)*

## Next Goal

Run each solver against the datasets above and compare:

- **Viability** — which solvers can even be applied as-is (LU/Cholesky need a square/SPD system, so rectangular datasets have to go through the normal equations first) and how accurate each solution is against a reference (`np.linalg.lstsq`).
- **Time required under ideal conditions** — wall-clock runtime as problem size (rows/features) grows, on well-conditioned, full-rank inputs, to isolate each algorithm's scaling behavior from conditioning issues.

Output (timing data, plots, and a written comparison) will land in `benchmarks/`, `results/`, and `notebooks/` once this is built out.

## Project Structure

```
linear-solvers-benchmark/
├── src/                  # Solver implementations (SVD, QR, LU, gradient descent)
├── tests/                # pytest suite (72 tests, all passing)
├── benchmarks/           # (empty) scripts that will run and time each solver across datasets
├── data/                 # raw/interim/processed datasets used for benchmarking
├── notebooks/            # 01-download-data, 02-eda, 03-feature-prep, 04-benchmark-analysis
└── results/              # (empty) generated plots and benchmark output
```

## How to Run

```bash
git clone git@github.com:HovarAK/linear-solvers-benchmark.git
cd linear-solvers-benchmark
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest + notebook kernel deps

pytest
```

There is no benchmark script yet — once `benchmarks/` is populated, running it will write timing data and plots to `results/`, with the full writeup in `notebooks/`.

## Status

🚧 In progress — all five solvers are implemented and unit-tested; benchmarking against real datasets is the current focus. Part of a portfolio project series moving from linear algebra fundamentals (this project) through applied ML, SVMs, and dimensionality reduction.
