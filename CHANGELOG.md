# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.3] - 2025-10-16

### Fixed
- Fixed C++ compilation error: Added missing `#include <string>` in `src/ref_family_python.cc`
- Fixed installation error on Test PyPI by updating build dependencies
- Updated `scikit-build-core` from >=0.8 to >=0.10.0 for compatibility with modern `packaging` library
- Added explicit `packaging>=24.0` requirement to avoid API incompatibility issues

### Changed
- Build system now requires `scikit-build-core>=0.10.0`, `nanobind>=2.0`, and `packaging>=24.0`

## [0.18.2] - 2025-10-16

### Fixed
- Initial attempt to fix installation error (superseded by 0.18.3)

## [0.18.0] - 2025-10-11

### Added
- GARCH models (`family="garch"`) for volatility change detection
- ARMA models (`family="arma"`) for time series analysis
- Linear regression (`family="lm"`) with perfect accuracy
- LASSO regression (`family="lasso"`) with L1 penalization
- Comprehensive evaluation metrics module:
  - Precision, Recall, F1-Score
  - Hausdorff distance, Covering metric
  - Annotation error, One-to-one correspondence
  - Multi-annotator metrics
- Dataset generation utilities with rich metadata
- Visualization tools for change point analysis
- Support for vanilla_percentage parameter (PELT/SeGD interpolation)

### Changed
- Improved GLM accuracy (71% win rate vs R implementation)
- Phase 1 optimizations: 1.4x speedup with adaptive pruning and warm start
- Optional Numba acceleration (7-14x additional speedup for GLM)

### Performance
- Mean/Variance models: 2-14x faster than R (C++ implementation)
- GLM models: Competitive performance with Numba installed
- ARMA/GARCH: Pure Python implementation with excellent accuracy

## [0.1.0] - 2025-06-01

### Added
- Initial Python implementation of fastcpd
- Core models: mean, variance, meanvariance
- GLM models: binomial, poisson
- C++ implementation for core models using nanobind
- Basic PELT algorithm with pruning
- SeGD (Sequential Gradient Descent) support
- pytest-based test suite

### Changed
- Ported from R package to Python
- Adapted C++ code for Python bindings

---

## Release Notes

### Version 0.18.0 Highlights

**New Capabilities:**
- Complete time series support (ARMA/GARCH)
- Full regression suite (linear/LASSO)
- Industry-leading evaluation metrics
- Flexible PELT/SeGD interpolation

**Performance:**
- Up to 14x faster with Numba for GLM
- 2-14x faster for core models vs R
- Suitable for datasets up to n=1000+

**Quality:**
- Comprehensive test suite (76/76 tests passing)
- Extensive documentation and examples
- Production-ready for most use cases

---

For detailed changes, see the [commit history](https://github.com/zhangxiany-tamu/fastcpd_Python/commits/main).
