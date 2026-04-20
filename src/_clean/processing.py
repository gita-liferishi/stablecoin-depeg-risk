"""
Data processing and feature engineering for stablecoin de-peg risk modeling.

This module handles:
- Data cleaning and validation
- Feature engineering for state-space models
- De-peg event labeling
- Train/test splitting for time series
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from loguru import logger


@dataclass
class DepegEvent:
    """Represents a de-peg event."""
    coin_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    min_price: float
    max_deviation: float
    duration_hours: float
    severity: str  # minor, moderate, severe, critical


class DataProcessor:
    """
    Process raw data into features suitable for state-space modeling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.depeg_thresholds = config.get("depeg", {}).get("thresholds", {
            "minor": 0.005,
            "moderate": 0.01,
            "severe": 0.03,
            "critical": 0.10
        })
        
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning price data: {len(df)} rows")
        
        df = df.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Handle missing values for price only (essential column)
        # Don't drop rows just because volume/high/low are missing
        if 'price' in df.columns:
            df['price'] = df['price'].ffill(limit=5).bfill(limit=5)
            
            # Only drop rows where price is still null
            null_price = df['price'].isnull().sum()
            if null_price > 0:
                logger.warning(f"Dropping {null_price} rows with null prices")
                df = df.dropna(subset=['price'])
        
        # Validate price ranges for stablecoins
        # Allow prices down to 0.01 to capture crash data (UST went near 0)
        # Upper bound 1.5 for depeg above peg
        if 'price' in df.columns:
            valid_mask = (df['price'] > 0.01) & (df['price'] < 1.5)
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(f"Removing {invalid_count} rows with prices outside 0.01-1.5 range")
                df = df[valid_mask]
        
        # Sort by index
        df = df.sort_index()
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        
        return df
    
    def compute_deviation_features(self, df: pd.DataFrame, peg_value: float = 1.0) -> pd.DataFrame:
        """
        Compute peg deviation features.
        
        Args:
            df: DataFrame with 'price' column
            peg_value: Target peg value (default 1.0 for USD)
            
        Returns:
            DataFrame with deviation features added
        """
        df = df.copy()
        
        # Absolute deviation
        df['deviation'] = df['price'] - peg_value
        df['abs_deviation'] = df['deviation'].abs()
        
        # Percentage deviation
        df['pct_deviation'] = df['deviation'] / peg_value
        df['abs_pct_deviation'] = df['pct_deviation'].abs()
        
        # Deviation direction (above/below peg)
        df['above_peg'] = (df['deviation'] > 0).astype(int)
        
        # Consecutive periods off-peg
        threshold = self.depeg_thresholds.get('minor', 0.005)
        df['off_peg'] = (df['abs_pct_deviation'] > threshold).astype(int)
        
        # Count consecutive off-peg periods
        df['off_peg_streak'] = df['off_peg'].groupby(
            (df['off_peg'] != df['off_peg'].shift()).cumsum()
        ).cumsum() * df['off_peg']
        
        return df
    
    def compute_volatility_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 30, 60, 90]
    ) -> pd.DataFrame:
        """
        Compute rolling volatility features.
        
        Args:
            df: DataFrame with 'price' column
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()
        
        # Log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        for window in windows:
            # Rolling standard deviation of returns
            df[f'volatility_{window}d'] = df['log_return'].rolling(window).std() * np.sqrt(252)
            
            # Rolling mean return
            df[f'mean_return_{window}d'] = df['log_return'].rolling(window).mean()
            
            # Rolling min/max from price (for range-based volatility)
            df[f'rolling_high_{window}d'] = df['price'].rolling(window).max()
            df[f'rolling_low_{window}d'] = df['price'].rolling(window).min()
            
            # Parkinson volatility estimator (uses rolling high-low range from price)
            df[f'parkinson_vol_{window}d'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(df[f'rolling_high_{window}d'] / df[f'rolling_low_{window}d']) ** 2)
            )
        
        return df
    
    def compute_volume_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Compute volume-based features.
        
        Args:
            df: DataFrame with 'volume' column
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with volume features added
        """
        df = df.copy()
        
        if 'volume' not in df.columns:
            logger.warning("No volume column found, skipping volume features")
            return df
        
        # Check if volume has any valid data
        if df['volume'].isna().all():
            logger.warning("Volume column is all NaN, skipping volume features")
            return df
        
        # Log volume
        df['log_volume'] = np.log1p(df['volume'])
        
        for window in windows:
            # Rolling mean volume
            df[f'avg_volume_{window}d'] = df['volume'].rolling(window).mean()
            
            # Volume relative to recent average
            df[f'volume_ratio_{window}d'] = df['volume'] / df[f'avg_volume_{window}d']
            
            # Volume momentum
            df[f'volume_momentum_{window}d'] = (
                df['volume'] / df['volume'].shift(window) - 1
            )
        
        return df
    
    def compute_momentum_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 3, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Compute price momentum features.
        
        Args:
            df: DataFrame with 'price' column
            periods: List of lookback periods
            
        Returns:
            DataFrame with momentum features added
        """
        df = df.copy()
        
        for period in periods:
            # Price momentum (percentage change)
            df[f'momentum_{period}d'] = df['price'].pct_change(period)
            
            # Price acceleration
            if period > 1:
                df[f'acceleration_{period}d'] = (
                    df[f'momentum_{period}d'] - 
                    df[f'momentum_{period}d'].shift(period)
                )
        
        return df
    
    def compute_market_cap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market cap related features.
        
        Args:
            df: DataFrame with 'market_cap' column
            
        Returns:
            DataFrame with market cap features added
        """
        df = df.copy()
        
        if 'market_cap' not in df.columns:
            logger.warning("No market_cap column found, skipping features")
            return df
        
        # Market cap changes
        df['mcap_change_1d'] = df['market_cap'].pct_change(1)
        df['mcap_change_7d'] = df['market_cap'].pct_change(7)
        df['mcap_change_30d'] = df['market_cap'].pct_change(30)
        
        # Market cap rolling stats
        df['mcap_volatility_30d'] = df['mcap_change_1d'].rolling(30).std()
        
        return df
    
    def label_depeg_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[DepegEvent]]:
        """
        Label de-peg events based on price deviations.
        
        Args:
            df: DataFrame with 'abs_pct_deviation' column
            
        Returns:
            Tuple of (DataFrame with labels, List of DepegEvent objects)
        """
        df = df.copy()
        events = []
        
        # Initialize labels
        df['depeg_label'] = 'stable'
        df['depeg_severity'] = 0.0
        
        for severity, threshold in sorted(
            self.depeg_thresholds.items(),
            key=lambda x: x[1],
            reverse=False
        ):
            mask = df['abs_pct_deviation'] >= threshold
            df.loc[mask, 'depeg_label'] = severity
            df.loc[mask, 'depeg_severity'] = threshold
        
        # Identify distinct de-peg episodes
        # A new episode starts when we transition from stable to any de-peg state
        df['in_depeg'] = (df['depeg_label'] != 'stable').astype(int)
        df['depeg_episode'] = (
            (df['in_depeg'] == 1) & (df['in_depeg'].shift(1) == 0)
        ).cumsum() * df['in_depeg']
        
        # Extract event details
        for episode_id in df[df['depeg_episode'] > 0]['depeg_episode'].unique():
            episode_data = df[df['depeg_episode'] == episode_id]
            
            if len(episode_data) > 0:
                coin_id = episode_data['coin_id'].iloc[0] if 'coin_id' in episode_data.columns else 'unknown'
                
                event = DepegEvent(
                    coin_id=coin_id,
                    start_time=episode_data.index[0],
                    end_time=episode_data.index[-1],
                    min_price=episode_data['price'].min(),
                    max_deviation=episode_data['abs_pct_deviation'].max(),
                    duration_hours=(episode_data.index[-1] - episode_data.index[0]).total_seconds() / 3600,
                    severity=episode_data['depeg_label'].value_counts().index[0]
                )
                events.append(event)
        
        logger.info(f"Identified {len(events)} de-peg events")
        
        return df, events
    
    def create_regime_labels(
        self,
        df: pd.DataFrame,
        n_regimes: int = 3
    ) -> pd.DataFrame:
        """
        Create discrete regime labels for HMM training validation.
        
        Uses volatility and deviation to assign approximate regime labels.
        These can be used to evaluate HMM's ability to recover true regimes.
        
        Args:
            df: DataFrame with volatility and deviation features
            n_regimes: Number of regimes to create
            
        Returns:
            DataFrame with regime labels
        """
        df = df.copy()
        
        # Default regime label
        df['regime_label'] = 0
        
        if n_regimes == 2:
            # Binary: stable vs stressed
            vol_col = 'volatility_30d' if 'volatility_30d' in df.columns else None
            dev_col = 'abs_pct_deviation' if 'abs_pct_deviation' in df.columns else None
            
            if vol_col and dev_col:
                vol_data = df[vol_col].dropna()
                if len(vol_data) > 0:
                    vol_threshold = vol_data.quantile(0.75)
                    dev_threshold = self.depeg_thresholds.get('minor', 0.005)
                    
                    df['regime_label'] = 0  # Stable
                    df.loc[
                        (df[vol_col] > vol_threshold) | (df[dev_col] > dev_threshold),
                        'regime_label'
                    ] = 1  # Stressed
                
        elif n_regimes == 3:
            # Three regimes: calm, elevated, crisis
            # Use deviation-based approach which is more robust
            dev_col = 'abs_pct_deviation' if 'abs_pct_deviation' in df.columns else None
            
            if dev_col:
                # Use thresholds to define regimes
                minor_thresh = self.depeg_thresholds.get('minor', 0.005)
                moderate_thresh = self.depeg_thresholds.get('moderate', 0.01)
                
                df['regime_label'] = 0  # Calm
                df.loc[df[dev_col] > minor_thresh, 'regime_label'] = 1  # Elevated
                df.loc[df[dev_col] > moderate_thresh, 'regime_label'] = 2  # Crisis
            else:
                # Fallback to volatility with error handling
                vol_col = 'volatility_30d' if 'volatility_30d' in df.columns else None
                
                if vol_col:
                    vol_data = df[vol_col].dropna()
                    if len(vol_data) > 10:  # Need enough data for quantiles
                        try:
                            df.loc[vol_data.index, 'regime_label'] = pd.qcut(
                                vol_data,
                                q=3,
                                labels=[0, 1, 2],
                                duplicates='drop'
                            ).astype(int)
                        except ValueError:
                            # If qcut fails, use simple percentile-based approach
                            p33 = vol_data.quantile(0.33)
                            p66 = vol_data.quantile(0.66)
                            df.loc[vol_data.index, 'regime_label'] = 0
                            df.loc[(df[vol_col] > p33) & (df[vol_col] <= p66), 'regime_label'] = 1
                            df.loc[df[vol_col] > p66, 'regime_label'] = 2
        
        return df
    
    def prepare_features_for_hmm(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for HMM training.
        
        Args:
            df: Processed DataFrame
            feature_cols: List of columns to use as features
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if feature_cols is None:
            # Default features for HMM
            feature_cols = [
                'abs_pct_deviation',
                'volatility_30d',
                'log_return',
                'volume_ratio_7d'
            ]
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing feature columns: {missing}")
        
        # Extract features and handle NaN
        X = df[available_cols].dropna().values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Prepared feature matrix: {X_scaled.shape}")
        
        return X_scaled, available_cols
    
    def train_test_split_timeseries(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data respecting temporal order.
        
        Args:
            df: DataFrame with datetime index
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.sort_index()
        n = len(df)
        
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - validation_size))
        
        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def filter_dead_stablecoins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Define collapse dates for dead stablecoins
        collapse_dates = {
            'UST': '2022-06-01',
            'terrausd': '2022-06-01',
            # Add others if needed (e.g., IRON)
        }
        
        initial_len = len(df)
        
        for symbol, cutoff_date in collapse_dates.items():
            # Check both 'symbol' and 'coin_id' columns
            for col in ['symbol', 'coin_id']:
                if col in df.columns:
                    mask = (df[col] == symbol) & (df.index > cutoff_date)
                    if mask.sum() > 0:
                        logger.info(f"Filtering {mask.sum()} rows of {symbol} after {cutoff_date}")
                        df = df[~mask]
        
        logger.info(f"Filtered {initial_len - len(df)} post-collapse rows")
        
        return df
    
    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full processing pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Fully processed DataFrame with all features
        """
        logger.info("Running full processing pipeline")
        
        # Clean data
        df = self.clean_price_data(df)
        df = self.filter_dead_stablecoins(df)
        # Compute features
        df = self.compute_deviation_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_volume_features(df)
        df = self.compute_momentum_features(df)
        df = self.compute_market_cap_features(df)
        
        # Label events
        df, events = self.label_depeg_events(df)
        
        # Create regime labels
        df = self.create_regime_labels(df)
        
        logger.info(f"Pipeline complete. Final shape: {df.shape}")
        
        return df


def pivot_multiasset_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot multi-asset data for cross-asset analysis.
    
    Args:
        df: DataFrame with 'coin_id' column
        
    Returns:
        Pivoted DataFrame with assets as columns
    """
    if 'coin_id' not in df.columns:
        return df
    
    # Pivot price data
    pivoted = df.reset_index().pivot(
        index='timestamp',
        columns='coin_id',
        values='price'
    )
    
    return pivoted


def compute_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that span multiple assets (for contagion analysis).
    
    Args:
        df: Pivoted DataFrame with assets as columns
        
    Returns:
        DataFrame with cross-asset features
    """
    features = pd.DataFrame(index=df.index)
    
    # Pairwise correlations (rolling)
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            corr_name = f'corr_{col1}_{col2}_30d'
            features[corr_name] = df[col1].rolling(30).corr(df[col2])
    
    # Average cross-correlation
    corr_cols = [c for c in features.columns if c.startswith('corr_')]
    if corr_cols:
        features['avg_cross_corr'] = features[corr_cols].mean(axis=1)
    
    # Dispersion (std across stablecoin prices)
    stablecoin_cols = [c for c in df.columns if c not in ['bitcoin', 'ethereum']]
    if stablecoin_cols:
        features['price_dispersion'] = df[stablecoin_cols].std(axis=1)
    
    return features


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Example usage
    processor = DataProcessor(config)
    
    # Load sample data
    raw_path = Path("./data/raw/prices_raw.parquet")
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        processed = processor.process_pipeline(df)
        
        # Save processed data
        processed.to_parquet("./data/processed/features.parquet")
        print(f"Processed data saved: {processed.shape}")