"""
Feature engineering script for stablecoin de-peg risk modeling.

This script:
1. Loads raw price and supply data
2. Computes features (deviation, volatility, momentum, etc.)
3. Labels de-peg events
4. Creates train/val/test splits
5. Saves processed data for model training
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from loguru import logger

from src._clean.processing import DataProcessor, pivot_multiasset_data, compute_cross_asset_features


def load_raw_data(data_dir: Path) -> dict:
    """Load all raw data files."""
    data = {}
    
    # Load prices
    prices_path = data_dir / "prices_raw.parquet"
    if prices_path.exists():
        data["prices"] = pd.read_parquet(prices_path)
        logger.info(f"Loaded prices: {data['prices'].shape}")
    else:
        logger.error(f"Prices file not found: {prices_path}")
        
    # Load supply
    supply_path = data_dir / "supply_raw.parquet"
    if supply_path.exists():
        data["supply"] = pd.read_parquet(supply_path)
        logger.info(f"Loaded supply: {data['supply'].shape}")
        
    return data


def process_single_asset(df: pd.DataFrame, processor: DataProcessor, symbol: str) -> pd.DataFrame:
    """Process features for a single stablecoin."""
    logger.info(f"Processing {symbol}...")
    
    # Filter to this asset
    asset_df = df[df["symbol"] == symbol].copy()
    
    if asset_df.empty:
        logger.warning(f"No data for {symbol}")
        return pd.DataFrame()
    
    # Sort by date
    asset_df = asset_df.sort_index()
    
    # Run processing pipeline
    try:
        processed = processor.process_pipeline(asset_df)
        processed["symbol"] = symbol
        logger.info(f"  ✓ {symbol}: {len(processed)} rows, {processed.shape[1]} features")
        return processed
    except Exception as e:
        logger.error(f"  ✗ {symbol} failed: {e}")
        return pd.DataFrame()


def merge_supply_features(prices_df: pd.DataFrame, supply_df: pd.DataFrame) -> pd.DataFrame:
    """Merge supply data into price features."""
    if supply_df is None or supply_df.empty:
        return prices_df
    
    # Prepare supply data
    supply_df = supply_df.reset_index()
    supply_df = supply_df.rename(columns={"date": "timestamp"})
    
    # Merge on date and symbol
    prices_df = prices_df.reset_index()
    
    merged = pd.merge(
        prices_df,
        supply_df[["timestamp", "symbol", "total_circulating"]],
        on=["timestamp", "symbol"],
        how="left"
    )
    
    # Compute supply-based features
    merged["supply_change_1d"] = merged.groupby("symbol")["total_circulating"].pct_change(1)
    merged["supply_change_7d"] = merged.groupby("symbol")["total_circulating"].pct_change(7)
    
    merged = merged.set_index("timestamp")
    
    logger.info(f"Merged supply features: {merged.shape}")
    
    return merged


def main():
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    # Paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "clean"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    logger.info("\n[1/5] Loading raw data...")
    raw_data = load_raw_data(raw_dir)
    
    if "prices" not in raw_data:
        logger.error("No price data found. Run extraction first.")
        return 1
    
    prices_df = raw_data["prices"]
    supply_df = raw_data.get("supply")
    
    # Initialize processor
    config = {
        "depeg": {
            "thresholds": {
                "minor": 0.005,      # 0.5%
                "moderate": 0.01,    # 1%
                "severe": 0.03,      # 3%
                "critical": 0.1     # 10%
            }
        }
    }
    processor = DataProcessor(config)
    
    # Process each stablecoin
    logger.info("\n[2/5] Processing individual assets...")
    symbols = prices_df["symbol"].unique()
    logger.info(f"Found {len(symbols)} assets: {list(symbols)}")
    
    processed_dfs = []
    for symbol in symbols:
        processed = process_single_asset(prices_df, processor, symbol)
        if not processed.empty:
            processed_dfs.append(processed)
    
    if not processed_dfs:
        logger.error("No assets processed successfully")
        return 1
    
    # Combine all processed data
    all_processed = pd.concat(processed_dfs)
    logger.info(f"Combined processed data: {all_processed.shape}")
    
    # Merge supply features
    logger.info("\n[3/5] Merging supply features...")
    if supply_df is not None:
        all_processed = merge_supply_features(all_processed, supply_df)
    
    # Compute cross-asset features
    logger.info("\n[4/5] Computing cross-asset features...")
    try:
        pivoted = pivot_multiasset_data(all_processed.reset_index())
        cross_features = compute_cross_asset_features(pivoted)
        
        # Save cross-asset features separately
        cross_features.to_parquet(processed_dir / "cross_asset_features.parquet")
        cross_features.to_csv(processed_dir / "cross_asset_features.csv")
        logger.info(f"Cross-asset features: {cross_features.shape}")
    except Exception as e:
        logger.warning(f"Cross-asset features failed: {e}")
    
    # Create train/val/test splits
    logger.info("\n[5/5] Creating train/val/test splits...")
    
    # Split each asset separately to maintain temporal order
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for symbol in all_processed["symbol"].unique():
        asset_df = all_processed[all_processed["symbol"] == symbol]
        train, val, test = processor.train_test_split_timeseries(
            asset_df, test_size=0.2, validation_size=0.1
        )
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
    
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    # Save processed data
    logger.info("\nSaving processed data...")
    
    # Full processed data
    all_processed.to_parquet(processed_dir / "features_all.parquet")
    all_processed.to_csv(processed_dir / "features_all.csv")
    
    # Splits
    train_df.to_parquet(processed_dir / "train.parquet")
    val_df.to_parquet(processed_dir / "val.parquet")
    test_df.to_parquet(processed_dir / "test.parquet")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Feature Engineering Summary")
    logger.info("=" * 60)
    logger.info(f"Total features: {all_processed.shape[1]}")
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Validation set: {len(val_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    logger.info(f"\nFiles saved to: {processed_dir}")
    
    # Dataset Statistics
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)
    
    # Per-asset stats
    logger.info("\n[Per-Asset Summary]")
    for symbol in all_processed["symbol"].unique():
        asset_df = all_processed[all_processed["symbol"] == symbol]
        logger.info(f"\n  {symbol}:")
        logger.info(f"    Rows: {len(asset_df)}")
        logger.info(f"    Date range: {asset_df.index.min().strftime('%Y-%m-%d')} to {asset_df.index.max().strftime('%Y-%m-%d')}")
        
        if "price" in asset_df.columns:
            logger.info(f"    Price - min: {asset_df['price'].min():.4f}, max: {asset_df['price'].max():.4f}, mean: {asset_df['price'].mean():.4f}")
        
        if "pct_deviation" in asset_df.columns:
            logger.info(f"    Max deviation: {asset_df['pct_deviation'].abs().max():.2%}")
        
        if "depeg_label" in asset_df.columns:
            depeg_counts = asset_df["depeg_label"].value_counts()
            logger.info(f"    Depeg events: {depeg_counts.to_dict()}")
    
    # Overall statistics
    logger.info("\n[Overall Statistics]")
    
    if "price" in all_processed.columns:
        logger.info(f"  Price range: {all_processed['price'].min():.4f} - {all_processed['price'].max():.4f}")
    
    if "pct_deviation" in all_processed.columns:
        logger.info(f"  Deviation range: {all_processed['pct_deviation'].min():.2%} to {all_processed['pct_deviation'].max():.2%}")
        logger.info(f"  Mean abs deviation: {all_processed['pct_deviation'].abs().mean():.4%}")
    
    if "volatility_30d" in all_processed.columns:
        logger.info(f"  Volatility (30d) - mean: {all_processed['volatility_30d'].mean():.4f}, max: {all_processed['volatility_30d'].max():.4f}")
    
    if "regime_label" in all_processed.columns:
        regime_counts = all_processed["regime_label"].value_counts().sort_index()
        logger.info(f"  Regime distribution: {regime_counts.to_dict()}")
    
    if "depeg_label" in all_processed.columns:
        depeg_counts = all_processed["depeg_label"].value_counts()
        logger.info(f"  Depeg label distribution: {depeg_counts.to_dict()}")
    
    # Missing values
    logger.info("\n[Missing Values]")
    missing = all_processed.isnull().sum()
    missing_pct = (missing / len(all_processed) * 100).round(2)
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        for col in cols_with_missing.index[:10]:
            logger.info(f"  {col}: {cols_with_missing[col]} ({missing_pct[col]:.1f}%)")
        if len(cols_with_missing) > 10:
            logger.info(f"  ... and {len(cols_with_missing) - 10} more columns with missing values")
    else:
        logger.info("  No missing values")
    
    # Feature correlations with deviation (top predictors)
    logger.info("\n[Top Features Correlated with Deviation]")
    if "abs_pct_deviation" in all_processed.columns:
        numeric_cols = all_processed.select_dtypes(include=[np.number]).columns
        correlations = all_processed[numeric_cols].corr()["abs_pct_deviation"].drop("abs_pct_deviation", errors="ignore")
        correlations = correlations.dropna().abs().sort_values(ascending=False)
        for col in correlations.head(10).index:
            logger.info(f"  {col}: {correlations[col]:.3f}")
    
    # List feature columns
    feature_cols = [c for c in all_processed.columns if c not in ["symbol", "coin_id"]]
    logger.info(f"\n[Feature Columns ({len(feature_cols)})]")
    for col in sorted(feature_cols)[:20]:
        logger.info(f"  - {col}")
    if len(feature_cols) > 20:
        logger.info(f"  ... and {len(feature_cols) - 20} more")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())