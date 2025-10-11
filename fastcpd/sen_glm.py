"""Sequential Gradient Descent (SEN) for GLM models.

This module implements the SEN algorithm for binomial and Poisson regression,
matching the R/C++ implementation for fast approximate change point detection.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SENState:
    """State for SEN algorithm at a given time point."""
    theta: np.ndarray  # Current parameter estimate
    theta_sum: np.ndarray  # Cumulative sum for averaging
    hessian: np.ndarray  # Accumulated Hessian matrix


class SENBinomial:
    """SEN implementation for binomial (logistic) regression."""

    def __init__(self, n_params: int, epsilon: float = 1e-10):
        """Initialize SEN for binomial regression.

        Parameters:
            n_params: Number of parameters (including intercept)
            epsilon: Small constant for numerical stability
        """
        self.n_params = n_params
        self.epsilon = epsilon

    def gradient(self, x: np.ndarray, y: float, theta: np.ndarray) -> np.ndarray:
        """Compute gradient for a single observation.

        Parameters:
            x: Features (1D array of length n_params)
            y: Binary response (0 or 1)
            theta: Current parameters

        Returns:
            Gradient vector

        Formula: -(y - p) * x where p = 1/(1 + exp(-x'theta))
        """
        u = np.dot(x, theta)
        # Clip to prevent overflow in exp
        u = np.clip(u, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-u))
        return -(y - prob) * x

    def hessian(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute Hessian for a single observation.

        Parameters:
            x: Features (1D array of length n_params)
            theta: Current parameters

        Returns:
            Hessian matrix (n_params x n_params)

        Formula: x.T * x * p * (1-p) where p = 1/(1 + exp(-x'theta))
        """
        u = np.dot(x, theta)
        # Clip to prevent overflow in exp
        u = np.clip(u, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-u))
        return np.outer(x, x) * prob * (1.0 - prob)

    def nll(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute negative log-likelihood for a segment.

        Parameters:
            X: Design matrix (n x n_params)
            y: Binary responses (n,)
            theta: Parameters

        Returns:
            Negative log-likelihood

        Formula: sum(-y*u + log(1 + exp(u))) where u = X*theta
        """
        u = X @ theta
        # Prevent overflow in exp
        u = np.clip(u, -500, 500)
        return np.sum(-y * u + np.log(1.0 + np.exp(u)))

    def initialize_state(self, x: np.ndarray, theta_init: np.ndarray) -> SENState:
        """Initialize SEN state for the first observation.

        Parameters:
            x: First observation features
            theta_init: Initial parameter estimate

        Returns:
            Initial SEN state
        """
        hess = self.hessian(x, theta_init)
        return SENState(
            theta=theta_init.copy(),
            theta_sum=theta_init.copy(),
            hessian=hess.copy()
        )

    def update_state(self, state: SENState, x: np.ndarray, theta_new: np.ndarray) -> SENState:
        """Update SEN state with new observation.

        Parameters:
            state: Current SEN state
            x: New observation features
            theta_new: New parameter estimate for this candidate

        Returns:
            Updated SEN state
        """
        hess_new = self.hessian(x, theta_new)
        return SENState(
            theta=theta_new.copy(),
            theta_sum=state.theta_sum + theta_new,
            hessian=state.hessian + hess_new
        )


class SENPoisson:
    """SEN implementation for Poisson regression."""

    def __init__(self, n_params: int, epsilon: float = 1e-10):
        """Initialize SEN for Poisson regression.

        Parameters:
            n_params: Number of parameters (including intercept)
            epsilon: Small constant for numerical stability
        """
        self.n_params = n_params
        self.epsilon = epsilon

    def gradient(self, x: np.ndarray, y: float, theta: np.ndarray) -> np.ndarray:
        """Compute gradient for a single observation.

        Parameters:
            x: Features (1D array of length n_params)
            y: Count response (non-negative integer)
            theta: Current parameters

        Returns:
            Gradient vector

        Formula: -(y - exp(x'theta)) * x
        """
        u = np.dot(x, theta)
        u = np.clip(u, -500, 500)  # Prevent overflow
        lambda_val = np.exp(u)
        return -(y - lambda_val) * x

    def hessian(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute Hessian for a single observation.

        Parameters:
            x: Features (1D array of length n_params)
            theta: Current parameters

        Returns:
            Hessian matrix (n_params x n_params)

        Formula: x.T * x * exp(x'theta)
        """
        u = np.dot(x, theta)
        u = np.clip(u, -500, 500)  # Prevent overflow
        lambda_val = np.exp(u)
        # Cap at 1e10 to match R implementation
        lambda_val = min(lambda_val, 1e10)
        return np.outer(x, x) * lambda_val

    def nll(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute negative log-likelihood for a segment.

        Parameters:
            X: Design matrix (n x n_params)
            y: Count responses (n,)
            theta: Parameters

        Returns:
            Negative log-likelihood

        Formula: sum(-y*u + exp(u) + log_factorial(y)) where u = X*theta
        """
        u = X @ theta
        # Prevent overflow
        u = np.clip(u, -500, 500)
        exp_u = np.exp(u)

        # Log factorial term
        log_fact = np.zeros(len(y))
        for i, yi in enumerate(y):
            if yi > 0:
                log_fact[i] = np.sum(np.log(np.arange(1, int(yi) + 1)))

        return np.sum(-y * u + exp_u + log_fact)

    def initialize_state(self, x: np.ndarray, theta_init: np.ndarray) -> SENState:
        """Initialize SEN state for the first observation.

        Parameters:
            x: First observation features
            theta_init: Initial parameter estimate

        Returns:
            Initial SEN state
        """
        hess = self.hessian(x, theta_init)
        return SENState(
            theta=theta_init.copy(),
            theta_sum=theta_init.copy(),
            hessian=hess.copy()
        )

    def update_state(self, state: SENState, x: np.ndarray, theta_new: np.ndarray) -> SENState:
        """Update SEN state with new observation.

        Parameters:
            state: Current SEN state
            x: New observation features
            theta_new: New parameter estimate for this candidate

        Returns:
            Updated SEN state
        """
        hess_new = self.hessian(x, theta_new)
        return SENState(
            theta=theta_new.copy(),
            theta_sum=state.theta_sum + theta_new,
            hessian=state.hessian + hess_new
        )


class SENLASSO:
    """SEN implementation for LASSO regression."""

    def __init__(self, n_params: int, epsilon: float = 1e-5):
        """Initialize SEN for LASSO regression.

        Parameters:
            n_params: Number of parameters
            epsilon: Small constant for numerical stability
        """
        self.n_params = n_params
        self.epsilon = epsilon

    def soft_threshold(self, a: np.ndarray, lambda_val: float) -> np.ndarray:
        """Soft thresholding operator for L1 penalty.

        Parameters:
            a: Input coefficients
            lambda_val: Threshold value

        Returns:
            Soft-thresholded coefficients

        Formula: sign(a) * max(|a| - lambda, 0)
        """
        return np.sign(a) * np.maximum(np.abs(a) - lambda_val, 0)

    def gradient(self, x: np.ndarray, y: float, theta: np.ndarray) -> np.ndarray:
        """Compute gradient for a single observation (least squares).

        Parameters:
            x: Features (1D array of length n_params)
            y: Response
            theta: Current parameters

        Returns:
            Gradient vector

        Formula: -(y - x'theta) * x
        """
        mu = np.dot(x, theta)
        return -(y - mu) * x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian for a single observation (least squares).

        Parameters:
            x: Features (1D array of length n_params)

        Returns:
            Hessian matrix (n_params x n_params)

        Formula: x.T * x
        """
        return np.outer(x, x)

    def nll(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_val: float) -> float:
        """Compute penalized least squares objective.

        Parameters:
            X: Design matrix (n x n_params)
            y: Responses (n,)
            theta: Parameters
            lambda_val: L1 penalty parameter

        Returns:
            Penalized objective: 0.5*RSS + lambda*|theta|_1

        Formula: sum((y - X*theta)^2)/2 + lambda*sum(|theta|)
        """
        residuals = y - X @ theta
        rss = np.sum(residuals ** 2) / 2
        l1_penalty = lambda_val * np.sum(np.abs(theta))
        return rss + l1_penalty

    def initialize_state(self, x: np.ndarray, theta_init: np.ndarray) -> SENState:
        """Initialize SEN state for the first observation.

        Parameters:
            x: First observation features
            theta_init: Initial parameter estimate

        Returns:
            Initial SEN state
        """
        hess = self.hessian(x) + self.epsilon * np.eye(self.n_params)
        return SENState(
            theta=theta_init.copy(),
            theta_sum=theta_init.copy(),
            hessian=hess.copy()
        )

    def update_state(self, state: SENState, x: np.ndarray, y: float,
                    lambda_val: float) -> SENState:
        """Update SEN state with new observation using proximal gradient descent.

        Parameters:
            state: Current SEN state
            x: New observation features
            y: New response
            lambda_val: L1 penalty parameter

        Returns:
            Updated SEN state

        Matches R's cost_lasso_update:
        - cmatrix = cmatrix + X_new %o% X_new
        - lik_dev = -(Y_new - mu) * X_new
        - coef = coef - solve(cmatrix, lik_dev)
        - nc = norm(cmatrix, type="F")
        - coef = soft(coef, lambda/nc)
        - cum_coef = cum_coef + coef
        """
        # Update Hessian matrix
        hess_new = self.hessian(x)
        hessian_updated = state.hessian + hess_new

        # Gradient step
        grad = self.gradient(x, y, state.theta)
        try:
            theta_new = state.theta - np.linalg.solve(hessian_updated, grad)
        except np.linalg.LinAlgError:
            theta_new = state.theta - grad / np.trace(hessian_updated)

        # Soft thresholding (proximal operator for L1)
        nc = np.linalg.norm(hessian_updated, ord='fro')  # Frobenius norm
        theta_new = self.soft_threshold(theta_new, lambda_val / nc)

        return SENState(
            theta=theta_new.copy(),
            theta_sum=state.theta_sum + theta_new,
            hessian=hessian_updated.copy()
        )


def newton_update(gradient: np.ndarray, hessian: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Perform Newton update step.

    Parameters:
        gradient: Gradient vector
        hessian: Hessian matrix
        epsilon: Regularization parameter

    Returns:
        Update direction (momentum)

    Formula: -solve(H + epsilon*I, g)
    """
    # Add epsilon to diagonal for positive definiteness
    hess_reg = hessian + epsilon * np.eye(hessian.shape[0])
    try:
        # Solve: (H + epsilon*I) * momentum = -gradient
        momentum = np.linalg.solve(hess_reg, -gradient)
        return momentum
    except np.linalg.LinAlgError:
        # If solve fails, use gradient descent
        return -gradient / (np.trace(hessian) + epsilon)
