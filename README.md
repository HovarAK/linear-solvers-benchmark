# linear-solvers-benchmark

Implementation and runtime/accuracy comparison of four approaches to solving linear systems and least-squares problems: **SVD**, **QR decomposition**, **LU decomposition**, and **gradient descent**.

## Problem

Given a linear system (or least-squares regression problem) `Ax = b`, there are multiple numerical methods to solve for `x`, each with different tradeoffs in speed, numerical stability, and scalability. This project implements each solver from first principles (using NumPy, not `np.linalg.solve` shortcuts), benchmarks their runtime as problem size grows, and compares solution accuracy.

**Solvers implemented:**
- **SVD** — most numerically stable, most expensive
- **QR decomposition** — good stability, faster than SVD
- **LU decomposition** — fastest for square systems, less stable for ill-conditioned matrices
- **Gradient descent** — iterative, scales differently than the direct methods above

## Dataset

Benchmarked across datasets of increasing size to show how each solver scales, not just a single data point:

- `sklearn.datasets.make_regression()` — synthetic data, lets us control rows/features independently to isolate scaling behavior
- `sklearn.datasets.fetch_california_housing()` — real-valued regression, mid-sized
- `sklearn.datasets.load_diabetes()` — smaller real-world dataset for comparison

*(Update this section once final datasets are locked in.)*

## Key Result

*(Fill in once benchmarking is complete — e.g. "QR decomposition offered the best balance of speed and stability across all tested sizes; gradient descent runtime scaled linearly with rows but required tuning of learning rate to match direct-method accuracy." Include the runtime plot here.)*


## Project Structure

```
linear-solvers-benchmark/
├── src/                  # Solver implementations (SVD, QR, LU, gradient descent)
├── benchmarks/           # Scripts that run and time each solver across datasets
├── notebooks/            # Exploration, plotting, and written analysis
└── results/              # Generated plots and benchmark output
```

## How to Run

```bash
git clone git@github.com:HovarAK/linear-solvers-benchmark.git
cd linear-solvers-benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python benchmarks/runtime_comparison.py
```

Results (plots, timing data) will be written to `results/`. See `notebooks/analysis.ipynb` for the full writeup and discussion.

## Status

🚧 In progress — part of a portfolio project series moving from linear algebra fundamentals (this project) through applied ML, SVMs, and dimensionality reduction.
