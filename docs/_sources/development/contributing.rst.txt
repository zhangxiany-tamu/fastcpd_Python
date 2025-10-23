Contributing
============

We welcome contributions to fastcpd-python! This guide will help you get started.

Ways to Contribute
------------------

- **Report bugs** via GitHub Issues
- **Suggest features** via GitHub Issues
- **Improve documentation**
- **Add tests**
- **Optimize performance**
- **Add examples**
- **Fix bugs**
- **Implement new features**

Getting Started
---------------

Fork and Clone
~~~~~~~~~~~~~~

.. code-block:: bash

   # Fork on GitHub first, then:
   git clone https://github.com/YOUR_USERNAME/fastcpd_Python.git
   cd fastcpd_Python

   # Add upstream remote
   git remote add upstream https://github.com/zhangxiany-tamu/fastcpd_Python.git

Development Setup
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install system dependencies
   # macOS:
   brew install armadillo

   # Ubuntu/Debian:
   sudo apt-get install libarmadillo-dev

   # Install in editable mode with dev dependencies
   pip install -e .[dev,test]

   # Install pre-commit hooks (optional but recommended)
   pre-commit install

Development Workflow
--------------------

1. Create a Branch
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Sync with upstream
   git fetch upstream
   git checkout main
   git merge upstream/main

   # Create feature branch
   git checkout -b feature/my-new-feature

2. Make Changes
~~~~~~~~~~~~~~~

Follow our coding standards:

**Python Code:**

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings (Google or NumPy style)
- Keep functions focused and small

**C++ Code:**

- Follow existing style
- Add comments for complex logic
- Ensure thread safety if applicable

3. Add Tests
~~~~~~~~~~~~

.. code-block:: python

   # Add tests to fastcpd/tests/
   def test_my_new_feature():
       from fastcpd.segmentation import mean
       # Your test here
       assert result is not None

4. Run Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest fastcpd/tests/test_basic.py

   # Run with coverage
   pytest --cov=fastcpd --cov-report=html

5. Format Code
~~~~~~~~~~~~~~

.. code-block:: bash

   # Format with black
   black fastcpd/

   # Lint with ruff
   ruff check fastcpd/

   # Type check with mypy
   mypy fastcpd/

6. Commit Changes
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "feat: add new feature for X"

   # Use conventional commits:
   # feat: new feature
   # fix: bug fix
   # docs: documentation
   # test: tests
   # refactor: code refactoring
   # perf: performance improvement

7. Push and Create PR
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/my-new-feature

Then create a Pull Request on GitHub.

Coding Standards
----------------

Python Style
~~~~~~~~~~~~

.. code-block:: python

   def detect_change_points(
       data: np.ndarray,
       beta: Union[str, float] = "MBIC",
       **kwargs
   ) -> FastcpdResult:
       """Detect change points in time series data.

       Args:
           data: Input array of shape (n, d)
           beta: Penalty parameter
           **kwargs: Additional arguments

       Returns:
           FastcpdResult object with detected change points

       Examples:
           >>> result = detect_change_points(data, beta="MBIC")
           >>> print(result.cp_set)
       """
       # Implementation
       pass

Documentation
~~~~~~~~~~~~~

- Use Google-style or NumPy-style docstrings
- Include examples in docstrings
- Update docs_sphinx/ if adding new features
- Add type hints

Testing
~~~~~~~

- Write tests for new features
- Maintain >80% code coverage
- Include edge cases
- Test both success and failure paths

Performance
~~~~~~~~~~~

- Profile before optimizing
- Add benchmarks for performance-critical code
- Document performance characteristics

Adding New Models
-----------------

To add a new model family:

1. **Implement cost function** in appropriate file:

   .. code-block:: python

      # fastcpd/pelt_newmodel.py
      def _fastcpd_newmodel(data, beta, trim):
          # Implementation
          return result_dict

2. **Add to segmentation.py**:

   .. code-block:: python

      def newmodel(data, beta="MBIC", **kwargs):
          """Detect changes in new model."""
          return fastcpd(data, family="newmodel", beta=beta, **kwargs)

3. **Add tests**:

   .. code-block:: python

      # fastcpd/tests/test_newmodel.py
      def test_newmodel():
          result = newmodel(data, beta="MBIC")
          assert len(result.cp_set) > 0

4. **Update documentation**:

   - Add to ``docs_sphinx/user_guide/models.rst``
   - Add example to ``examples/``
   - Update README.md

Documentation Contributions
----------------------------

Improving Docs
~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs_sphinx/
   make html
   open _build/html/index.html

Adding Examples
~~~~~~~~~~~~~~~

.. code-block:: python

   # examples/example_newfeature.py
   """
   Example: Using New Feature
   ===========================

   This example demonstrates...
   """
   import numpy as np
   from fastcpd.segmentation import mean

   # Your example code
   ...

Reporting Issues
----------------

Good Bug Reports
~~~~~~~~~~~~~~~~

Include:

- Python version
- fastcpd version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/stack traces

.. code-block:: python

   # Example bug report
   import fastcpd
   print(f"fastcpd version: {fastcpd.__version__}")

   # Minimal example
   data = ...
   result = ...  # Bug occurs here

Feature Requests
~~~~~~~~~~~~~~~~

Include:

- Use case description
- Why existing features don't solve it
- Proposed API (if applicable)
- Example usage

Pull Request Guidelines
------------------------

PR Checklist
~~~~~~~~~~~~

- [ ] Tests pass locally
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Followed coding standards
- [ ] Added entry to CHANGELOG.md
- [ ] PR description explains changes

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation
   - [ ] Performance improvement

   ## Testing
   How tested

   ## Related Issues
   Fixes #123

Code Review Process
-------------------

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Address feedback
4. Approval and merge

**Timeline:** Usually within 1-2 weeks

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

Communication
~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: Questions, ideas

Recognition
-----------

Contributors are:

- Listed in ``CONTRIBUTORS.md``
- Mentioned in release notes
- Acknowledged in documentation

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.

Getting Help
------------

- Read this guide
- Check existing issues
- Ask in GitHub Discussions
- Email: zhangxiany@stat.tamu.edu

Thank You!
----------

Your contributions make fastcpd-python better for everyone. We appreciate your time and effort!
