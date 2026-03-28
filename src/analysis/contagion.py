"""
Contagion and spillover analysis for stablecoin networks.

This module implements:
- VAR-based spillover indices (Diebold-Yilmaz methodology)
- Granger causality network construction
- Cross-asset correlation dynamics
- Contagion risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from loguru import logger
import warnings


@dataclass
class SpilloverResults:
    """Container for spillover analysis results."""
    spillover_table: pd.DataFrame
    total_spillover_index: float
    directional_to: pd.Series  # Spillovers TO each asset
    directional_from: pd.Series  # Spillovers FROM each asset
    net_spillover: pd.Series  # Net spillover (to - from)
    pairwise_spillovers: pd.DataFrame
    forecast_horizon: int
    var_lags: int


@dataclass
class ContagionNetwork:
    """Container for contagion network analysis."""
    adjacency_matrix: pd.DataFrame
    graph: nx.DiGraph
    centrality_measures: Dict[str, pd.Series]
    communities: Optional[List[set]]
    risk_scores: pd.Series


class DieboldYilmazSpillover:
    """
    Implements Diebold-Yilmaz spillover index methodology.
    
    Reference: Diebold & Yilmaz (2012) "Better to give than to receive:
    Predictive directional measurement of volatility spillovers"
    """
    
    def __init__(
        self,
        var_lags: int = 5,
        forecast_horizon: int = 10,
        generalized: bool = True
    ):
        """
        Initialize spillover analyzer.
        
        Args:
            var_lags: Number of lags for VAR model
            forecast_horizon: Forecast horizon for variance decomposition
            generalized: Use generalized (order-invariant) decomposition
        """
        self.var_lags = var_lags
        self.forecast_horizon = forecast_horizon
        self.generalized = generalized
        self.var_model = None
        
    def fit(self, data: pd.DataFrame) -> SpilloverResults:
        """
        Compute spillover indices from multivariate time series.
        
        Args:
            data: DataFrame with assets as columns, time as index
            
        Returns:
            SpilloverResults object
        """
        logger.info(f"Computing spillovers for {data.shape[1]} assets")
        
        # Fit VAR model
        self.var_model = VAR(data.dropna())
        var_results = self.var_model.fit(self.var_lags)
        
        n_vars = data.shape[1]
        var_names = data.columns.tolist()
        
        # Get forecast error variance decomposition
        fevd = var_results.fevd(self.forecast_horizon)
        
        # Extract decomposition matrix at forecast horizon
        # fevd.decomp has shape (n_vars, forecast_horizon, n_vars)
        # We want the cumulative at the forecast horizon
        decomp = fevd.decomp
        
        # Sum over forecast horizons to get cumulative FEVD
        fevd_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                fevd_matrix[i, j] = decomp[i, self.forecast_horizon-1, j]
        
        # Normalize rows to sum to 100
        row_sums = fevd_matrix.sum(axis=1, keepdims=True)
        fevd_normalized = (fevd_matrix / row_sums) * 100
        
        # Create spillover table
        spillover_table = pd.DataFrame(
            fevd_normalized,
            index=var_names,
            columns=var_names
        )
        
        # Compute spillover measures
        # Directional spillovers TO asset i FROM all others
        directional_from = spillover_table.sum(axis=1) - np.diag(spillover_table.values)
        directional_from = pd.Series(directional_from, index=var_names)
        
        # Directional spillovers FROM asset i TO all others
        directional_to = spillover_table.sum(axis=0) - np.diag(spillover_table.values)
        directional_to = pd.Series(directional_to, index=var_names)
        
        # Net spillovers
        net_spillover = directional_to - directional_from
        
        # Total spillover index
        total_spillover = (spillover_table.sum().sum() - np.trace(spillover_table.values)) / n_vars
        
        # Pairwise spillovers (off-diagonal elements)
        pairwise = spillover_table.copy()
        np.fill_diagonal(pairwise.values, 0)
        
        logger.info(f"Total spillover index: {total_spillover:.2f}%")
        
        return SpilloverResults(
            spillover_table=spillover_table,
            total_spillover_index=total_spillover,
            directional_to=directional_to,
            directional_from=directional_from,
            net_spillover=net_spillover,
            pairwise_spillovers=pairwise,
            forecast_horizon=self.forecast_horizon,
            var_lags=self.var_lags
        )
    
    def rolling_spillover(
        self,
        data: pd.DataFrame,
        window: int = 200,
        step: int = 1
    ) -> pd.DataFrame:
        """
        Compute rolling spillover index over time.
        
        Args:
            data: DataFrame with assets as columns
            window: Rolling window size
            step: Step size for rolling window
            
        Returns:
            DataFrame with time-varying spillover indices
        """
        results = []
        dates = []
        
        for i in range(0, len(data) - window, step):
            window_data = data.iloc[i:i+window]
            
            try:
                spillover = self.fit(window_data)
                results.append({
                    'total_spillover': spillover.total_spillover_index,
                    **{f'to_{col}': spillover.directional_to[col] 
                       for col in data.columns},
                    **{f'from_{col}': spillover.directional_from[col] 
                       for col in data.columns}
                })
                dates.append(data.index[i + window - 1])
            except Exception as e:
                logger.warning(f"Rolling spillover failed at index {i}: {e}")
                continue
        
        return pd.DataFrame(results, index=dates)


class GrangerCausalityNetwork:
    """
    Construct directed network based on Granger causality tests.
    """
    
    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05
    ):
        """
        Initialize Granger causality analyzer.
        
        Args:
            max_lag: Maximum lag to test
            significance_level: Significance level for causality
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        
    def fit(self, data: pd.DataFrame) -> ContagionNetwork:
        """
        Construct Granger causality network.
        
        Args:
            data: DataFrame with assets as columns
            
        Returns:
            ContagionNetwork object
        """
        var_names = data.columns.tolist()
        n_vars = len(var_names)
        
        # Initialize adjacency matrix
        adjacency = pd.DataFrame(
            np.zeros((n_vars, n_vars)),
            index=var_names,
            columns=var_names
        )
        
        # Test all pairs
        for i, col_i in enumerate(var_names):
            for j, col_j in enumerate(var_names):
                if i == j:
                    continue
                
                try:
                    # Test if col_j Granger-causes col_i
                    test_data = data[[col_i, col_j]].dropna()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = grangercausalitytests(
                            test_data, 
                            maxlag=self.max_lag,
                            verbose=False
                        )
                    
                    # Get minimum p-value across all lags
                    p_values = [result[lag][0]['ssr_ftest'][1] 
                               for lag in range(1, self.max_lag + 1)]
                    min_p = min(p_values)
                    
                    if min_p < self.significance_level:
                        # Strength based on -log(p-value)
                        adjacency.loc[col_i, col_j] = -np.log10(min_p + 1e-10)
                        
                except Exception as e:
                    logger.debug(f"Granger test failed for {col_j}->{col_i}: {e}")
                    continue
        
        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(var_names)
        
        for i in var_names:
            for j in var_names:
                if adjacency.loc[i, j] > 0:
                    G.add_edge(j, i, weight=adjacency.loc[i, j])
        
        # Compute centrality measures
        centrality = {
            'in_degree': pd.Series(dict(G.in_degree(weight='weight'))),
            'out_degree': pd.Series(dict(G.out_degree(weight='weight'))),
            'betweenness': pd.Series(nx.betweenness_centrality(G, weight='weight')),
            'eigenvector': pd.Series(nx.eigenvector_centrality_numpy(G, weight='weight'))
            if len(G.edges()) > 0 else pd.Series({n: 0 for n in var_names})
        }
        
        # Risk score: combination of centrality measures
        risk_scores = (
            centrality['in_degree'] / centrality['in_degree'].max() +
            centrality['out_degree'] / centrality['out_degree'].max() +
            centrality['betweenness'] / (centrality['betweenness'].max() + 1e-10)
        ) / 3
        
        return ContagionNetwork(
            adjacency_matrix=adjacency,
            graph=G,
            centrality_measures=centrality,
            communities=None,  # Can add community detection later
            risk_scores=risk_scores
        )


class DynamicCorrelation:
    """
    Track dynamic correlations between stablecoins.
    """
    
    def __init__(self, window: int = 30):
        """
        Initialize dynamic correlation analyzer.
        
        Args:
            window: Rolling window for correlation
        """
        self.window = window
        
    def compute_rolling_correlation(
        self,
        data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling pairwise correlations.
        
        Args:
            data: DataFrame with assets as columns
            
        Returns:
            Dictionary with correlation time series
        """
        var_names = data.columns.tolist()
        
        # Pairwise rolling correlations
        correlations = {}
        for i, col_i in enumerate(var_names):
            for j, col_j in enumerate(var_names[i+1:], i+1):
                pair_name = f"{col_i}_{col_j}"
                correlations[pair_name] = data[col_i].rolling(self.window).corr(data[col_j])
        
        return correlations
    
    def compute_average_correlation(
        self,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute average pairwise correlation over time.
        
        Args:
            data: DataFrame with assets as columns
            
        Returns:
            Series with average correlation
        """
        corr_dict = self.compute_rolling_correlation(data)
        
        if not corr_dict:
            return pd.Series(dtype=float)
        
        corr_df = pd.DataFrame(corr_dict)
        return corr_df.mean(axis=1)
    
    def detect_correlation_regime_changes(
        self,
        data: pd.DataFrame,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Detect periods of unusually high or low correlation.
        
        Args:
            data: DataFrame with assets as columns
            threshold: Z-score threshold for regime detection
            
        Returns:
            DataFrame with regime indicators
        """
        avg_corr = self.compute_average_correlation(data)
        
        # Compute z-scores
        z_scores = (avg_corr - avg_corr.mean()) / avg_corr.std()
        
        # Identify regimes
        result = pd.DataFrame({
            'avg_correlation': avg_corr,
            'z_score': z_scores,
            'high_correlation_regime': z_scores > threshold,
            'low_correlation_regime': z_scores < -threshold
        })
        
        return result


class ContagionRiskAnalyzer:
    """
    Comprehensive contagion risk analysis combining multiple methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize contagion analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.spillover = DieboldYilmazSpillover(
            var_lags=self.config.get('var_lags', 5),
            forecast_horizon=self.config.get('forecast_horizon', 10)
        )
        self.granger = GrangerCausalityNetwork(
            max_lag=self.config.get('granger_max_lag', 10),
            significance_level=self.config.get('significance_level', 0.05)
        )
        self.dcc = DynamicCorrelation(
            window=self.config.get('correlation_window', 30)
        )
        
    def analyze(
        self,
        returns_data: pd.DataFrame,
        volatility_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run full contagion analysis.
        
        Args:
            returns_data: DataFrame with asset returns
            volatility_data: Optional DataFrame with volatility measures
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Running comprehensive contagion analysis")
        
        results = {}
        
        # 1. Spillover analysis on returns
        try:
            results['spillover_returns'] = self.spillover.fit(returns_data)
            logger.info(f"Return spillover index: {results['spillover_returns'].total_spillover_index:.2f}%")
        except Exception as e:
            logger.error(f"Spillover analysis failed: {e}")
        
        # 2. Spillover on volatility (if provided)
        if volatility_data is not None:
            try:
                results['spillover_volatility'] = self.spillover.fit(volatility_data)
                logger.info(f"Volatility spillover index: {results['spillover_volatility'].total_spillover_index:.2f}%")
            except Exception as e:
                logger.error(f"Volatility spillover failed: {e}")
        
        # 3. Granger causality network
        try:
            results['granger_network'] = self.granger.fit(returns_data)
            n_edges = results['granger_network'].graph.number_of_edges()
            logger.info(f"Granger network: {n_edges} significant causal links")
        except Exception as e:
            logger.error(f"Granger causality failed: {e}")
        
        # 4. Dynamic correlations
        try:
            results['correlation_regimes'] = self.dcc.detect_correlation_regime_changes(
                returns_data
            )
            high_corr_pct = results['correlation_regimes']['high_correlation_regime'].mean() * 100
            logger.info(f"High correlation regime: {high_corr_pct:.1f}% of time")
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
        
        # 5. Composite risk score
        if 'spillover_returns' in results and 'granger_network' in results:
            results['composite_risk'] = self._compute_composite_risk(results)
        
        return results
    
    def _compute_composite_risk(self, results: Dict) -> pd.Series:
        """Compute composite contagion risk score per asset."""
        spillover = results['spillover_returns']
        granger = results['granger_network']
        
        # Normalize and combine
        spillover_risk = spillover.directional_from / spillover.directional_from.max()
        granger_risk = granger.risk_scores / (granger.risk_scores.max() + 1e-10)
        
        # Weighted average
        composite = 0.5 * spillover_risk + 0.5 * granger_risk
        
        return composite.sort_values(ascending=False)


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    # n = 500
    
    # # Generate correlated returns for 4 stablecoins
    # cov = np.array([
    #     [1.0, 0.6, 0.5, 0.4],
    #     [0.6, 1.0, 0.7, 0.5],
    #     [0.5, 0.7, 1.0, 0.6],
    #     [0.4, 0.5, 0.6, 1.0]
    # ]) * 0.0001
    
    # returns = np.random.multivariate_normal([0, 0, 0, 0], cov, n)
    
    # # Add some lagged effects (contagion)
    # for t in range(1, n):
    #     returns[t, 1] += 0.3 * returns[t-1, 0]  # USDT -> USDC
    #     returns[t, 2] += 0.2 * returns[t-1, 1]  # USDC -> DAI
    
    # data = pd.DataFrame(
    #     returns,
    #     columns=['USDT', 'USDC', 'DAI', 'FRAX'],
    #     index=pd.date_range('2023-01-01', periods=n, freq='D')
    # )
    
    # # Run analysis
    # analyzer = ContagionRiskAnalyzer()
    # results = analyzer.analyze(data)
    
    # print("\n" + "=" * 50)
    # print("Spillover Table:")
    # print(results['spillover_returns'].spillover_table.round(2))
    
    # print("\n" + "=" * 50)
    # print("Net Spillovers (positive = net transmitter):")
    # print(results['spillover_returns'].net_spillover.round(2))
    
    # print("\n" + "=" * 50)
    # print("Granger Causality Centrality:")
    # print(results['granger_network'].centrality_measures['out_degree'].round(2))
    
    # if 'composite_risk' in results:
    #     print("\n" + "=" * 50)
    #     print("Composite Contagion Risk:")
    #     print(results['composite_risk'].round(3))
