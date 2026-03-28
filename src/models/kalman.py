"""
Kalman Filter implementation for tracking latent volatility states.

This module implements Kalman filtering approaches for:
- Tracking unobserved volatility states
- Filtering noisy price observations
- Providing smooth estimates of underlying dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from pykalman import KalmanFilter as PyKalmanFilter
from loguru import logger


@dataclass
class KalmanResults:
    """Container for Kalman filter results."""
    filtered_state_means: np.ndarray
    filtered_state_covariances: np.ndarray
    smoothed_state_means: Optional[np.ndarray]
    smoothed_state_covariances: Optional[np.ndarray]
    log_likelihood: float
    observation_matrix: np.ndarray
    transition_matrix: np.ndarray
    observation_noise: np.ndarray
    transition_noise: np.ndarray


class VolatilityKalmanFilter:
    """
    Kalman Filter for tracking latent volatility in stablecoin prices.
    
    State-space model:
        State equation:     x_t = A * x_{t-1} + w_t,  w_t ~ N(0, Q)
        Observation eq:     y_t = H * x_t + v_t,      v_t ~ N(0, R)
    
    Where:
        x_t: Latent volatility state
        y_t: Observed squared returns or realized volatility proxy
        A: State transition matrix (persistence of volatility)
        H: Observation matrix
        Q: State noise covariance (volatility of volatility)
        R: Observation noise covariance
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        observation_dim: int = 1,
        em_iterations: int = 50
    ):
        """
        Initialize Kalman Filter.
        
        Args:
            state_dim: Dimension of latent state
            observation_dim: Dimension of observations
            em_iterations: Number of EM iterations for parameter estimation
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.em_iterations = em_iterations
        
        self.kf = None
        self.is_fitted = False
        
    def _initialize_filter(
        self,
        initial_state_mean: float = 0.0,
        initial_state_covariance: float = 1.0,
        transition_coeff: float = 0.95,
        observation_coeff: float = 1.0,
        transition_noise: float = 0.1,
        observation_noise: float = 0.5
    ) -> PyKalmanFilter:
        """
        Initialize Kalman Filter with parameters.
        
        Args:
            initial_state_mean: Initial estimate of state
            initial_state_covariance: Initial uncertainty
            transition_coeff: AR(1) coefficient (persistence)
            observation_coeff: Observation loading
            transition_noise: State noise variance
            observation_noise: Observation noise variance
            
        Returns:
            Initialized PyKalmanFilter
        """
        kf = PyKalmanFilter(
            transition_matrices=np.array([[transition_coeff]]),
            observation_matrices=np.array([[observation_coeff]]),
            initial_state_mean=np.array([initial_state_mean]),
            initial_state_covariance=np.array([[initial_state_covariance]]),
            transition_covariance=np.array([[transition_noise]]),
            observation_covariance=np.array([[observation_noise]]),
            em_vars=['transition_covariance', 'observation_covariance',
                    'initial_state_mean', 'initial_state_covariance']
        )
        
        return kf
    
    def fit(
        self,
        observations: np.ndarray,
        estimate_params: bool = True
    ) -> KalmanResults:
        """
        Fit Kalman Filter to observations.
        
        Args:
            observations: Observation sequence (n_samples,) or (n_samples, 1)
            estimate_params: Whether to estimate parameters via EM
            
        Returns:
            KalmanResults object
        """
        # Reshape if necessary
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        logger.info(f"Fitting Kalman Filter on {len(observations)} observations")
        
        # Initialize filter
        self.kf = self._initialize_filter()
        
        # Estimate parameters using EM if requested
        if estimate_params:
            logger.info(f"Running EM algorithm for {self.em_iterations} iterations")
            self.kf = self.kf.em(observations, n_iter=self.em_iterations)
        
        # Run filter (forward pass)
        filtered_means, filtered_covs = self.kf.filter(observations)
        
        # Run smoother (backward pass)
        smoothed_means, smoothed_covs = self.kf.smooth(observations)
        
        # Compute log-likelihood
        log_likelihood = self.kf.loglikelihood(observations)
        
        self.is_fitted = True
        
        results = KalmanResults(
            filtered_state_means=filtered_means,
            filtered_state_covariances=filtered_covs,
            smoothed_state_means=smoothed_means,
            smoothed_state_covariances=smoothed_covs,
            log_likelihood=log_likelihood,
            observation_matrix=self.kf.observation_matrices,
            transition_matrix=self.kf.transition_matrices,
            observation_noise=self.kf.observation_covariance,
            transition_noise=self.kf.transition_covariance
        )
        
        logger.info(f"Kalman Filter fitted. Log-likelihood: {log_likelihood:.2f}")
        
        return results
    
    def filter_online(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filter to new observations (online filtering).
        
        Args:
            observations: New observations
            
        Returns:
            Tuple of (filtered_means, filtered_covariances)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before online filtering")
        
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        return self.kf.filter(observations)
    
    def forecast(
        self,
        n_steps: int,
        current_state: Optional[np.ndarray] = None,
        current_cov: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future states.
        
        Args:
            n_steps: Number of steps to forecast
            current_state: Current state estimate
            current_cov: Current state covariance
            
        Returns:
            Tuple of (forecast_means, forecast_covariances)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        A = self.kf.transition_matrices
        Q = self.kf.transition_covariance
        
        if current_state is None:
            current_state = self.kf.initial_state_mean
        if current_cov is None:
            current_cov = self.kf.initial_state_covariance
        
        forecasts = []
        forecast_covs = []
        
        state = current_state
        cov = current_cov
        
        for _ in range(n_steps):
            state = A @ state
            cov = A @ cov @ A.T + Q
            forecasts.append(state.copy())
            forecast_covs.append(cov.copy())
        
        return np.array(forecasts), np.array(forecast_covs)


class LocalLevelModel:
    """
    Local Level (Random Walk plus Noise) model for stablecoin deviations.
    
    y_t = mu_t + epsilon_t,  epsilon_t ~ N(0, sigma_e^2)
    mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma_n^2)
    
    This model captures:
    - mu_t: True underlying deviation level (latent)
    - y_t: Observed deviation (noisy)
    """
    
    def __init__(self):
        """Initialize Local Level Model."""
        self.kf = None
        self.is_fitted = False
        
    def fit(
        self,
        observations: np.ndarray,
        em_iterations: int = 50
    ) -> KalmanResults:
        """
        Fit Local Level model.
        
        Args:
            observations: Observed deviations
            em_iterations: EM iterations for parameter estimation
            
        Returns:
            KalmanResults object
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        
        # Local level model specification
        self.kf = PyKalmanFilter(
            transition_matrices=np.array([[1.0]]),  # Random walk
            observation_matrices=np.array([[1.0]]),  # Direct observation
            initial_state_mean=np.array([observations[0, 0]]),
            initial_state_covariance=np.array([[1.0]]),
            em_vars=['transition_covariance', 'observation_covariance']
        )
        
        # Estimate parameters
        self.kf = self.kf.em(observations, n_iter=em_iterations)
        
        # Filter and smooth
        filtered_means, filtered_covs = self.kf.filter(observations)
        smoothed_means, smoothed_covs = self.kf.smooth(observations)
        
        self.is_fitted = True
        
        return KalmanResults(
            filtered_state_means=filtered_means,
            filtered_state_covariances=filtered_covs,
            smoothed_state_means=smoothed_means,
            smoothed_state_covariances=smoothed_covs,
            log_likelihood=self.kf.loglikelihood(observations),
            observation_matrix=self.kf.observation_matrices,
            transition_matrix=self.kf.transition_matrices,
            observation_noise=self.kf.observation_covariance,
            transition_noise=self.kf.transition_covariance
        )


class StochasticVolatilityModel:
    """
    Stochastic Volatility model approximation using Kalman Filter.
    
    log(sigma_t^2) = mu + phi * (log(sigma_{t-1}^2) - mu) + eta_t
    r_t = sigma_t * epsilon_t
    
    This is approximated via quasi-maximum likelihood using
    log(r_t^2) as the observation.
    """
    
    def __init__(self):
        """Initialize Stochastic Volatility model."""
        self.kf = None
        self.is_fitted = False
        self.mu = None  # Long-run mean of log-variance
        self.phi = None  # Persistence parameter
        
    def fit(
        self,
        returns: np.ndarray,
        em_iterations: int = 100
    ) -> KalmanResults:
        """
        Fit Stochastic Volatility model.
        
        Args:
            returns: Return series
            em_iterations: EM iterations
            
        Returns:
            KalmanResults object
        """
        # Transform to log squared returns
        # Add small constant to avoid log(0)
        log_r2 = np.log(returns**2 + 1e-10)
        
        if log_r2.ndim == 1:
            log_r2 = log_r2.reshape(-1, 1)
        
        n_obs = len(log_r2)
        
        # Approximate observation equation:
        # log(r_t^2) = log(sigma_t^2) + log(epsilon_t^2)
        # where log(epsilon_t^2) has mean -1.27 and variance pi^2/2
        
        observation_offset = -1.27
        observation_variance = (np.pi**2) / 2
        
        # Initial parameter guesses
        initial_phi = 0.95
        initial_mu = np.mean(log_r2)
        initial_sigma_eta = 0.2
        
        self.kf = PyKalmanFilter(
            transition_matrices=np.array([[initial_phi]]),
            observation_matrices=np.array([[1.0]]),
            transition_offsets=np.array([initial_mu * (1 - initial_phi)]),
            observation_offsets=np.array([observation_offset]),
            initial_state_mean=np.array([initial_mu]),
            initial_state_covariance=np.array([[1.0]]),
            transition_covariance=np.array([[initial_sigma_eta**2]]),
            observation_covariance=np.array([[observation_variance]]),
            em_vars=['transition_matrices', 'transition_covariance',
                    'initial_state_mean', 'initial_state_covariance']
        )
        
        # Run EM
        try:
            self.kf = self.kf.em(log_r2, n_iter=em_iterations)
        except Exception as e:
            logger.warning(f"EM failed, using initial parameters: {e}")
        
        # Extract estimated parameters
        self.phi = float(self.kf.transition_matrices[0, 0])
        
        # Filter and smooth
        filtered_means, filtered_covs = self.kf.filter(log_r2)
        smoothed_means, smoothed_covs = self.kf.smooth(log_r2)
        
        # Convert log-variance to volatility
        filtered_vol = np.exp(filtered_means / 2)
        smoothed_vol = np.exp(smoothed_means / 2)
        
        self.is_fitted = True
        
        logger.info(f"Stochastic Volatility model fitted. phi={self.phi:.3f}")
        
        return KalmanResults(
            filtered_state_means=filtered_vol,
            filtered_state_covariances=filtered_covs,
            smoothed_state_means=smoothed_vol,
            smoothed_state_covariances=smoothed_covs,
            log_likelihood=self.kf.loglikelihood(log_r2),
            observation_matrix=self.kf.observation_matrices,
            transition_matrix=self.kf.transition_matrices,
            observation_noise=self.kf.observation_covariance,
            transition_noise=self.kf.transition_covariance
        )


def create_multivariate_kalman(
    n_assets: int,
    correlation_structure: str = "diagonal"
) -> PyKalmanFilter:
    """
    Create multivariate Kalman Filter for tracking multiple stablecoins.
    
    Args:
        n_assets: Number of assets to track
        correlation_structure: 'diagonal', 'full', or 'factor'
        
    Returns:
        Initialized multivariate Kalman Filter
    """
    if correlation_structure == "diagonal":
        # Independent dynamics for each asset
        A = np.eye(n_assets) * 0.95
        Q = np.eye(n_assets) * 0.1
        H = np.eye(n_assets)
        R = np.eye(n_assets) * 0.5
        
    elif correlation_structure == "full":
        # Allow cross-asset dependencies
        A = np.ones((n_assets, n_assets)) * 0.1 + np.eye(n_assets) * 0.85
        Q = np.eye(n_assets) * 0.1
        H = np.eye(n_assets)
        R = np.eye(n_assets) * 0.5
        
    else:
        raise ValueError(f"Unknown correlation structure: {correlation_structure}")
    
    kf = PyKalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=np.zeros(n_assets),
        initial_state_covariance=np.eye(n_assets),
        transition_covariance=Q,
        observation_covariance=R,
        em_vars=['transition_matrices', 'transition_covariance',
                'observation_covariance', 'initial_state_mean']
    )
    
    return kf


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic stochastic volatility data
    n = 500
    
    # True latent log-volatility process
    phi = 0.95
    sigma_eta = 0.15
    mu = -1.0
    
    h = np.zeros(n)
    h[0] = mu
    for t in range(1, n):
        h[t] = mu + phi * (h[t-1] - mu) + np.random.normal(0, sigma_eta)
    
    # Observed returns
    sigma = np.exp(h / 2)
    returns = sigma * np.random.normal(0, 1, n)
    
    # Fit models
    print("=" * 50)
    print("Volatility Kalman Filter")
    print("=" * 50)
    
    vol_kf = VolatilityKalmanFilter()
    vol_results = vol_kf.fit(np.abs(returns))  # Use absolute returns as proxy
    
    print(f"\nEstimated transition matrix:\n{vol_results.transition_matrix}")
    print(f"Estimated observation noise: {vol_results.observation_noise[0,0]:.4f}")
    print(f"Estimated transition noise: {vol_results.transition_noise[0,0]:.4f}")
    
    print("\n" + "=" * 50)
    print("Stochastic Volatility Model")
    print("=" * 50)
    
    sv_model = StochasticVolatilityModel()
    sv_results = sv_model.fit(returns)
    
    print(f"\nEstimated phi: {sv_model.phi:.4f} (true: {phi})")
    
    # Compare filtered vs true volatility
    correlation = np.corrcoef(
        sv_results.smoothed_state_means.flatten(),
        sigma
    )[0, 1]
    print(f"Correlation of smoothed vs true volatility: {correlation:.4f}")