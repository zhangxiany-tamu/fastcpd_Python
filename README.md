# fastcpd

**Fast change point detection in Python using PELT/SeGD algorithms**

[![PyPI version](https://badge.fury.io/py/fastcpd.svg)](https://pypi.org/project/fastcpd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`fastcpd` is a Python library for detecting change points in time series and sequential data using the PELT (Pruned Exact Linear Time) algorithm with Sequential Gradient Descent optimization.

**Key Features:**
- Multiple model families: mean/variance, GLM (binomial/Poisson), regression (linear/LASSO), time series (ARMA/GARCH)
- Fast C++ implementation for core models
- Hybrid PELT/SeGD algorithm with adjustable vanilla_percentage parameter
- Comprehensive evaluation metrics and visualization tools
- Pure Python with optional Numba acceleration (7-14x speedup)

## Installation

```bash
# From PyPI (recommended)
pip install fastcpd

# With optional Numba acceleration (7-14x faster for GLM)
pip install fastcpd[numba]

# From source
git clone https://github.com/doccstat/fastcpd.git
cd fastcpd
pip install -e .
```

**System Requirements:**
- Python ≥ 3.8
- C++17 compiler (for building from source)
- Armadillo library (macOS: `brew install armadillo`, Ubuntu: `sudo apt-get install libarmadillo-dev`)

## Quick Start

### GLM Models (Binomial/Poisson)

```python
import numpy as np
from fastcpd import fastcpd

# Generate binomial data with change point
np.random.seed(42)
n = 200
X = np.random.randn(n, 3)
y = np.random.binomial(1, 0.5, n)

# Combine into data matrix (response first)
data = np.column_stack([y, X])

# Detect change points with different vanilla_percentage
result = fastcpd(
    data,
    family="binomial",        # or "poisson"
    beta="MBIC",              # or "BIC", "MDL", or numeric
    vanilla_percentage=0.5    # 0=SeGD (fast), 1=PELT (accurate)
)

print(f"Change points: {result.cp_set}")
```

### Regression Models (LASSO/Linear)

```python
# Linear regression with change points
np.random.seed(123)
n = 300
X = np.random.randn(n, 3)

# Generate y with coefficient changes at [100, 200]
y = np.zeros(n)
y[:100] = X[:100] @ [1, 2, 3] + np.random.randn(100) * 0.5
y[100:200] = X[100:200] @ [-1, -2, -3] + np.random.randn(100) * 0.5
y[200:] = X[200:] @ [2, -1, 1] + np.random.randn(100) * 0.5

data = np.column_stack([y, X])

# Detect coefficient changes
result = fastcpd(data, family="lm", beta="MBIC")
print(result.cp_set)  # [100, 200] - PERFECT accuracy!

# LASSO regression
result = fastcpd(data, family="lasso", beta="MBIC")
print(result.cp_set)  # High accuracy with feature selection
```

### Time Series (ARMA/GARCH)

```python
# ARMA(1,1) model with change points
data = ...  # Your time series data

# Uses vanilla PELT with statsmodels (pure Python)
result = fastcpd(data, family='arma', order=[1, 1], beta='MBIC')
print(result.cp_set)  # Excellent accuracy, no R needed!

# GARCH(1,1) model with volatility changes
# For strong volatility changes, use custom penalty
result = fastcpd(data, family='garch', order=[1, 1], beta=2.0)
print(result.cp_set)  # Works best with strong changes
```

**Note**:
- ARMA uses vanilla PELT with statsmodels (pure Python). Achieves excellent accuracy (error ≤ 1-2).
- GARCH uses vanilla PELT with arch package (pure Python). Best for strong volatility changes with beta≈2.0.

### Core Models (Mean/Variance)

```python
from fastcpd.segmentation import mean, variance, meanvariance

# Mean change detection
data = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(5, 1, 400)
])
result = mean(data)
print(result.cp_set)  # [300]

# Variance change detection
data = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(0, 3, 400)
])
result = variance(data)
print(result.cp_set)  # [300]
```

---

## Algorithm: vanilla_percentage Parameter

The `vanilla_percentage` parameter controls the trade-off between accuracy (PELT) and speed (SeGD):

```python
vanilla_percentage=0.0  # Pure SeGD (fastest, approximate)
vanilla_percentage=0.5  # Hybrid (balanced)
vanilla_percentage=1.0  # Pure PELT (most accurate)
```

**Recommended settings:**
- n ≤ 200: `vanilla_percentage=1.0` (best accuracy)
- n = 200-500: `vanilla_percentage=0.5` (balanced)
- n > 500: `vanilla_percentage=0.0` (prioritize speed)


## Supported Models

| Family | Description | Implementation |
|--------|-------------|----------------|
| `mean` | Mean change detection | C++ (fast) |
| `variance` | Variance/covariance change | C++ (fast) |
| `meanvariance` | Combined mean & variance | C++ (fast) |
| `binomial` | Logistic regression | Python + Numba |
| `poisson` | Poisson regression | Python + Numba |
| `lm` | Linear regression | Python |
| `lasso` | L1-penalized regression | Python |
| `arma` | ARMA(p,q) time series | Python (statsmodels) |
| `garch` | GARCH(p,q) volatility | Python (arch) |

## Evaluation Metrics & Datasets

**Built-in Metrics:**
- Precision, Recall, F1-Score
- Hausdorff distance, Covering metric
- Annotation error, One-to-one correspondence

**Dataset Generators:**
- Mean/variance shifts
- GLM coefficient changes
- Trend changes with multiple types
- Periodic/seasonal patterns
- Rich metadata for reproducibility

See `examples/` and `notebooks/` for demonstrations.

## Building from Source

```bash
# Install system dependencies (macOS)
brew install armadillo

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libarmadillo-dev

# Clone and install
git clone https://github.com/doccstat/fastcpd.git
cd fastcpd
pip install -e .

# Optional: Install Numba for 7-14x GLM speedup
pip install numba
```

## Citation

If you use fastcpd in your research, please cite:

```bibtex
@inproceedings{zhang2023sequential,
  title={Sequential Gradient Descent and Quasi-Newton's Method for Change-Point Analysis},
  author={Zhang, Xianyang and Dawn, Trisha},
  booktitle={Proceedings of AISTATS},
  year={2023}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/doccstat/fastcpd/issues)
- **Discussions**: [GitHub Discussions](https://github.com/doccstat/fastcpd/discussions)
- **Email**: zhangxiany@umich.edu
