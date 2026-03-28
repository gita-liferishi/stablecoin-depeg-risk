"""
Hidden Markov Model implementation for stablecoin regime detection.

This module implements HMM-based approaches for:
- Detecting latent market regimes (stable, stressed, crisis)
- Identifying regime transitions as early warning signals
- Modeling observation distributions under different states
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger


@dataclass
class HMMResults:
    """Container for HMM fitting results."""
    model: hmm.GaussianHMM
    n_states: int
    log_likelihood: float
    aic: float
    bic: float
    states: np.ndarray
    state_probs: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]


class StablecoinHMM:
    """
    Hidden Markov Model for stablecoin regime detection.
    
    This model identifies latent market states from observable features
    like price deviations, volatility, and volume patterns.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 0.01,
        random_state: int = 42,
        n_init: int = 10
    ):
        """
        Initialize HMM model.
        
        Args:
            n_states: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ('full', 'diag', 'spherical')
            n_iter: Maximum iterations for EM algorithm
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            n_init: Number of initializations to try
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def _create_model(self) -> hmm.GaussianHMM:
        """Create a new HMM model instance."""
        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params="stmc",  # Initialize all parameters
            params="stmc"  # Update all parameters during training
        )
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        lengths: Optional[List[int]] = None
    ) -> HMMResults:
        """
        Fit HMM to observation data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features
            lengths: Lengths of individual sequences (for multiple sequences)
            
        Returns:
            HMMResults object containing fitted model and diagnostics
        """
        logger.info(f"Fitting HMM with {self.n_states} states on data shape {X.shape}")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Try multiple initializations and keep the best
        best_model = None
        best_score = -np.inf
        
        for init in range(self.n_init):
            try:
                model = self._create_model()
                model.random_state = self.random_state + init
                model.fit(X_scaled, lengths=lengths)
                
                score = model.score(X_scaled, lengths=lengths)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Initialization {init} failed: {e}")
                continue
        
        if best_model is None:
            raise ValueError("All HMM initializations failed")
        
        self.model = best_model
        self.is_fitted = True
        
        # Compute model selection criteria
        n_params = self._count_parameters()
        n_samples = X.shape[0]
        
        log_likelihood = best_score
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        # Decode most likely state sequence
        states = self.model.predict(X_scaled, lengths=lengths)
        state_probs = self.model.predict_proba(X_scaled, lengths=lengths)
        
        results = HMMResults(
            model=self.model,
            n_states=self.n_states,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            states=states,
            state_probs=state_probs,
            transition_matrix=self.model.transmat_,
            means=self.model.means_,
            covariances=self.model.covars_,
            scaler=self.scaler,
            feature_names=self.feature_names
        )
        
        logger.info(f"HMM fitted. Log-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        return results
    
    def _count_parameters(self) -> int:
        """Count number of free parameters in the model."""
        n = self.n_states
        k = len(self.feature_names)  # Number of features
        
        # Initial state probabilities: n-1 (sum to 1)
        n_params = n - 1
        
        # Transition matrix: n * (n-1) (each row sums to 1)
        n_params += n * (n - 1)
        
        # Emission means: n * k
        n_params += n * k
        
        # Emission covariances
        if self.covariance_type == "full":
            n_params += n * k * (k + 1) // 2
        elif self.covariance_type == "diag":
            n_params += n * k
        elif self.covariance_type == "spherical":
            n_params += n
        
        return n_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of state probabilities (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        
        Args:
            X: Feature matrix
            
        Returns:
            Log-likelihood score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'n_states': self.n_states,
                'covariance_type': self.covariance_type
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'StablecoinHMM':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            n_states=data['n_states'],
            covariance_type=data['covariance_type']
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance


class HMMModelSelector:
    """
    Select optimal number of states using information criteria.
    """
    
    def __init__(
        self,
        state_range: List[int] = [2, 3, 4, 5],
        criterion: str = "bic",
        cv_folds: int = 5
    ):
        """
        Initialize model selector.
        
        Args:
            state_range: List of state counts to try
            criterion: Selection criterion ('aic', 'bic', 'cv')
            cv_folds: Number of cross-validation folds (for 'cv' criterion)
        """
        self.state_range = state_range
        self.criterion = criterion
        self.cv_folds = cv_folds
        self.results = {}
        
    def select(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[int, Dict]:
        """
        Select optimal number of states.
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            
        Returns:
            Tuple of (optimal_n_states, results_dict)
        """
        logger.info(f"Selecting optimal states from {self.state_range}")
        
        results = {}
        
        for n_states in self.state_range:
            try:
                model = StablecoinHMM(n_states=n_states)
                hmm_results = model.fit(X, feature_names)
                
                results[n_states] = {
                    'log_likelihood': hmm_results.log_likelihood,
                    'aic': hmm_results.aic,
                    'bic': hmm_results.bic,
                    'model': model
                }
                
                logger.info(f"  n_states={n_states}: LL={hmm_results.log_likelihood:.2f}, "
                           f"AIC={hmm_results.aic:.2f}, BIC={hmm_results.bic:.2f}")
                
            except Exception as e:
                logger.error(f"Failed for n_states={n_states}: {e}")
                continue
        
        self.results = results
        
        # Select based on criterion
        if self.criterion == "aic":
            optimal = min(results.keys(), key=lambda k: results[k]['aic'])
        elif self.criterion == "bic":
            optimal = min(results.keys(), key=lambda k: results[k]['bic'])
        else:
            optimal = min(results.keys(), key=lambda k: results[k]['bic'])
        
        logger.info(f"Selected n_states={optimal} based on {self.criterion}")
        
        return optimal, results
    
    def cross_validate(
        self,
        X: np.ndarray,
        n_states: int
    ) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X: Feature matrix
            n_states: Number of states to evaluate
            
        Returns:
            Dictionary with CV results
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        train_scores = []
        test_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            model = StablecoinHMM(n_states=n_states)
            model.fit(X_train)
            
            train_scores.append(model.score(X_train))
            test_scores.append(model.score(X_test))
        
        return {
            'train_scores': train_scores,
            'test_scores': test_scores,
            'mean_train': np.mean(train_scores),
            'mean_test': np.mean(test_scores),
            'std_test': np.std(test_scores)
        }


def interpret_hmm_states(
    results: HMMResults,
    original_df: pd.DataFrame,
    deviation_col: str = 'abs_pct_deviation',
    volatility_col: str = 'volatility_30d'
) -> Dict[int, str]:
    """
    Interpret HMM states based on feature characteristics.
    
    Args:
        results: Fitted HMM results
        original_df: Original DataFrame with features
        deviation_col: Column name for deviation
        volatility_col: Column name for volatility
        
    Returns:
        Dictionary mapping state index to interpretable label
    """
    interpretations = {}
    
    # Get mean characteristics for each state
    df_with_states = original_df.copy()
    df_with_states['state'] = results.states[:len(df_with_states)]
    
    state_stats = df_with_states.groupby('state').agg({
        deviation_col: 'mean',
        volatility_col: 'mean'
    }).to_dict('index')
    
    # Sort states by increasing deviation/volatility
    sorted_states = sorted(
        state_stats.keys(),
        key=lambda s: state_stats[s].get(deviation_col, 0) + state_stats[s].get(volatility_col, 0)
    )
    
    # Assign labels
    labels = ['stable', 'elevated', 'crisis'][:len(sorted_states)]
    
    for i, state in enumerate(sorted_states):
        if i < len(labels):
            interpretations[state] = labels[i]
        else:
            interpretations[state] = f'state_{state}'
    
    return interpretations


def compute_regime_transition_signals(
    state_probs: np.ndarray,
    crisis_state: int,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Compute early warning signals based on regime transition probabilities.
    
    Args:
        state_probs: State probability matrix (n_samples, n_states)
        crisis_state: Index of the crisis state
        threshold: Probability threshold for warning signal
        
    Returns:
        Binary warning signal array
    """
    crisis_prob = state_probs[:, crisis_state]
    
    # Signal when crisis probability exceeds threshold
    signals = (crisis_prob > threshold).astype(int)
    
    return signals


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic multi-regime data
    # n_samples = 1000
    
    # # State sequence (ground truth)
    # true_states = np.zeros(n_samples, dtype=int)
    # true_states[300:400] = 1  # Elevated period
    # true_states[600:650] = 2  # Crisis period
    # true_states[800:850] = 1  # Another elevated period
    
    # # Generate observations based on states
    # means = {0: [0.001, 0.02], 1: [0.01, 0.05], 2: [0.05, 0.15]}
    # stds = {0: [0.002, 0.01], 1: [0.01, 0.03], 2: [0.03, 0.05]}
    
    # X = np.zeros((n_samples, 2))
    # for t in range(n_samples):
    #     state = true_states[t]
    #     X[t] = np.random.normal(means[state], stds[state])
    
    # # Fit HMM
    # model = StablecoinHMM(n_states=3)
    # results = model.fit(X, feature_names=['deviation', 'volatility'])
    
    # print(f"\nTransition Matrix:\n{results.transition_matrix}")
    # print(f"\nState Means:\n{results.means}")
    
    # # Model selection
    # selector = HMMModelSelector(state_range=[2, 3, 4])
    # optimal_states, selection_results = selector.select(X)
    # print(f"\nOptimal number of states: {optimal_states}")
