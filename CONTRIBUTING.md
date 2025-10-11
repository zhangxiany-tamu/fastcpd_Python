# Contributing to fastcpd-python

Thank you for your interest in contributing to fastcpd-python!

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Report issues to zhangxiany@umich.edu.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/doccstat/fastcpd/issues) to avoid duplicates.

When you create a bug report, please include:
- A clear and descriptive title
- Exact steps to reproduce the problem
- Expected vs. actual behavior
- Code samples (if applicable)
- Python version, OS, and fastcpd version
- Any relevant error messages or logs

Use the **Bug Report** template when creating an issue.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:
- A clear and descriptive title
- A detailed description of the proposed feature
- Explain why this enhancement would be useful
- Provide examples of how it would be used

Use the **Feature Request** template when creating an issue.

### Contributing Code

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our style guidelines
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request**

### Improving Documentation

Documentation improvements are always welcome! This includes:
- Fixing typos or clarifying existing docs
- Adding examples
- Improving API documentation
- Creating tutorials or guides

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/fastcpd.git
cd fastcpd
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Install Numba for faster computation (optional but recommended)
pip install numba

# Install documentation dependencies
pip install sphinx sphinx-rtd-theme nbsphinx sphinx-copybutton
```

### 4. Install C++ Dependencies (for core models)

**macOS**:
```bash
brew install armadillo
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install libarmadillo-dev
```

**Windows**: Requires Armadillo library and C++17 compiler. See README for details.

### 5. Build the Package

```bash
./build.sh
# Or manually:
pip install -e . --no-build-isolation
```

### 6. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastcpd --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::test_precision_recall
```

## Pull Request Process

### Before Submitting

1. **Update your fork** with the latest from `main`:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # Or for bug fixes:
   git checkout -b fix/bug-description
   ```

3. **Make your changes** with clear, descriptive commits:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub

3. **Fill out the PR template** completely

4. **Link related issues** using keywords like "Fixes #123"

### PR Requirements

Your PR must:
- Pass all CI checks (tests, linting)
- Include tests for new functionality
- Update documentation if needed
- Follow our style guidelines
- Have a clear description of changes

### Review Process

1. A maintainer will review your PR
2. You may be asked to make changes
3. Once approved, a maintainer will merge your PR

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Docstrings**: Google style
- **Type hints**: Encouraged for new code
- **Imports**: Organized with `isort`

### Formatting Tools

We use automated formatters:

```bash
# Install formatters
pip install black isort flake8

# Format code
black fastcpd/ tests/
isort fastcpd/ tests/

# Check style
flake8 fastcpd/ tests/
```

### Docstring Style

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description if needed. Explain what the function does,
    when to use it, and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    pass
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and PRs when relevant

Good commit messages:
```
Add covering metric for multi-annotator evaluation

Implement Binary Segmentation algorithm (#123)

Fix: Handle empty input in precision_recall (#456)

Docs: Add example for GARCH model usage
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

Example test:

```python
import numpy as np
from fastcpd.metrics import precision_recall

def test_precision_recall_perfect_match():
    """Test precision/recall with perfect match."""
    true_cps = [100, 200, 300]
    pred_cps = [100, 200, 300]

    result = precision_recall(true_cps, pred_cps, margin=0)

    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1_score'] == 1.0
    assert result['true_positives'] == 3
    assert result['false_positives'] == 0
    assert result['false_negatives'] == 0
```

### Running Tests Locally

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests for specific module
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=fastcpd --cov-report=term-missing

# Run only fast tests (skip slow tests)
pytest -m "not slow"
```

### Test Coverage

- Aim for >80% coverage for new code
- All new features must have tests
- Bug fixes should include regression tests

## Documentation

### Documentation Guidelines

- Add docstrings to all public functions/classes (Google style)
- Include examples in docstrings
- Update CHANGELOG.md for any changes
- Add usage examples in `examples/` or `notebooks/` directories

### Adding Examples

```python
"""
Example: Change Point Detection with Binomial Data
===================================================

Demonstrates basic usage of fastcpd for logistic regression.
"""

import numpy as np
from fastcpd import fastcpd

# Your example code...
```

## Release Process

(For maintainers only)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Build and upload to PyPI

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/doccstat/fastcpd/discussions)
- **Bugs**: Open an [issue](https://github.com/doccstat/fastcpd/issues)
- **Email**: Contact maintainers at zhangxiany@umich.edu

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Our documentation

Thank you for contributing to fastcpd-python!
