Algorithms
==========

Mathematical foundation of the PELT and SeGD algorithms used in fastcpd.

Overview
--------

fastcpd implements two main algorithms for change point detection:

1. **PELT** (Pruned Exact Linear Time) - Exact optimal segmentation via dynamic programming
2. **SeGD** (Sequential Gradient Descent) - Fast approximate method using sequential updates

Problem Formulation
-------------------

Change point detection aims to partition a sequence of observations :math:`z_1, z_2, \ldots, z_n` into :math:`m+1` homogeneous segments separated by change points :math:`\tau = (\tau_0, \tau_1, \ldots, \tau_m, \tau_{m+1})` where :math:`\tau_0 = 0` and :math:`\tau_{m+1} = n`.

Cost Function
~~~~~~~~~~~~~

For a segment :math:`[\tau_{i-1}+1, \tau_i]`, the cost function is defined as the minimized negative log-likelihood:

.. math::
   :label: cost-function

   C(z_{\tau_{i-1}+1:\tau_i}) = \min_{\theta} \sum_{j=\tau_{i-1}+1}^{\tau_i} l(z_j, \theta)

where :math:`l(z_j, \theta)` is the negative log-likelihood of observation :math:`z_j` under parameter :math:`\theta`.

Optimization Problem
~~~~~~~~~~~~~~~~~~~~

The change point detection problem is formulated as:

.. math::

   \min_{\tau, m} \left[ \sum_{i=1}^{m+1} C(z_{\tau_{i-1}+1:\tau_i}) + \beta \cdot m \right]

where :math:`\beta > 0` is a penalty parameter controlling the bias-variance trade-off.

PELT Algorithm
--------------

Dynamic Programming Recursion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define :math:`F(t)` as the minimum cost for segmenting the first :math:`t` observations:

.. math::

   F(0) = -\beta

.. math::

   F(t) = \min_{\tau < t} \left[ F(\tau) + C(z_{\tau+1:t}) + \beta \right], \quad t = 1, 2, \ldots, n

The optimal change points are recovered by backtracking through the minimizers at each step.

Pruning Technique
~~~~~~~~~~~~~~~~~

At time :math:`t`, if for some :math:`\tau < s < t`:

.. math::
   :label: pruning-condition-1

   F(\tau) + C(z_{\tau+1:t}) \geq F(s) + C(z_{s+1:t})

then :math:`\tau` can be pruned and will never be optimal for any :math:`t' > t`.

**Proof:**

For any :math:`t' > t`, by the additivity of the cost function:

.. math::
   :label: pruning-condition-2

   F(\tau) + C(z_{\tau+1:t'}) &= F(\tau) + C(z_{\tau+1:t}) + C(z_{t+1:t'})\\
   &\geq F(s) + C(z_{s+1:t}) + C(z_{t+1:t'})\\
   &\geq F(s) + C(z_{s+1:t'})

Thus :math:`s` always produces a better or equal cost than :math:`\tau` for all future time points.

SeGD Algorithm
--------------

Sequential Parameter Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a segment :math:`[\tau+1, t]` with parameter estimate :math:`\tilde{\theta}_{\tau+1:t-1}` based on observations :math:`z_{\tau+1}, \ldots, z_{t-1}`.

When a new observation :math:`z_t` arrives, we want to update the parameter estimate to :math:`\tilde{\theta}_{\tau+1:t}` without recomputing from scratch.

**Taylor Expansion:**

Using a first-order Taylor expansion around :math:`\tilde{\theta}_{\tau+1:t-1}`:

.. math::

   \nabla_\theta \sum_{i=\tau+1}^{t} l(z_i, \theta) \bigg|_{\theta=\tilde{\theta}_{\tau+1:t}} \approx \nabla_\theta \sum_{i=\tau+1}^{t-1} l(z_i, \theta) \bigg|_{\theta=\tilde{\theta}_{\tau+1:t-1}} + \nabla l(z_t, \tilde{\theta}_{\tau+1:t-1})

Setting this gradient to zero and using :math:`\nabla_\theta \sum_{i=\tau+1}^{t-1} l(z_i, \tilde{\theta}_{\tau+1:t-1}) = 0`:

.. math::

   \nabla l(z_t, \tilde{\theta}_{\tau+1:t-1}) \approx 0

**Newton-Raphson Update:**

Using a second-order Taylor expansion with preconditioning matrix :math:`H_{\tau+1:t-1}`:

.. math::

   \tilde{\theta}_{\tau+1:t} \approx \tilde{\theta}_{\tau+1:t-1} - H_{\tau+1:t-1}^{-1} \nabla l(z_t, \tilde{\theta}_{\tau+1:t-1})

where :math:`H_{\tau+1:t-1}` approximates the Hessian matrix.

Preconditioning Matrix Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hessian Update:**

.. math::

   H_{\tau+1:t} = H_{\tau+1:t-1} + \nabla^2 l(z_t, \tilde{\theta}_{\tau+1:t})

where :math:`\nabla^2 l` is the Hessian of the negative log-likelihood.

**Fisher Scoring Update:**

.. math::

   H_{\tau+1:t} = H_{\tau+1:t-1} + \mathbb{E}[\nabla^2 l(z_t, \tilde{\theta}_{\tau+1:t})]

**Initialization:**

.. math::

   H_{\tau+1:\tau+1} = \nabla^2 l(z_{\tau+1}, \tilde{\theta}_{\tau+1:\tau+1})

Multiple Passes (Epochs)
~~~~~~~~~~~~~~~~~~~~~~~~~

To improve the approximation quality, SeGD can make multiple passes over the data:

**Algorithm:**

For epoch :math:`e = 1, 2, \ldots, E`:

1. **Forward Pass** (left to right):

   For :math:`t = 1, 2, \ldots, n`:

   - For each segment endpoint :math:`\tau_k < t`:
     - Update :math:`\tilde{\theta}_{\tau_k+1:t}` using the sequential update
     - Update :math:`H_{\tau_k+1:t}` using Hessian or Fisher scoring

2. **Backward Pass** (right to left, optional):

   For :math:`t = n, n-1, \ldots, 1`:

   - Similar updates in reverse order

**Convergence:**

With multiple epochs, parameters converge to more accurate estimates, change point locations stabilize, and approximation quality improves.

Hybrid PELT/SeGD
----------------

The ``vanilla_percentage`` parameter controls interpolation between PELT and SeGD:

.. code-block:: python

   # Pure PELT (exact, slower)
   result = fastcpd(data, family="binomial", vanilla_percentage=1.0)

   # Pure SeGD (fast, approximate)
   result = fastcpd(data, family="binomial", vanilla_percentage=0.0)

   # Hybrid (50% PELT warmup, 50% SeGD)
   result = fastcpd(data, family="binomial", vanilla_percentage=0.5)

**How It Works:**

1. Run PELT on first :math:`\lfloor \alpha \cdot n \rfloor` samples where :math:`\alpha = \text{vanilla\_percentage}`
2. Use PELT solution to warm-start SeGD
3. Run SeGD on remaining :math:`(1-\alpha) \cdot n` samples

Model Selection
---------------

Penalty Parameter β
~~~~~~~~~~~~~~~~~~~

**Information Criteria:**

.. math::

   \text{BIC:} \quad \beta = \frac{p \log(n)}{2}

.. math::

   \text{MBIC:} \quad \beta = \frac{(p + 2) \log(n)}{2}

.. math::

   \text{MDL:} \quad \beta = \frac{p \log(n)}{2}

where :math:`p` is the number of parameters per segment and :math:`n` is the sample size.

References
----------

.. [Killick2012] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.

.. [Zhang2024] Zhang, X., & Dawn, T. (2024). Sequential Gradient Descent and Quasi-Newton's Method for Change-Point Analysis. arXiv:2404.05933v1.
