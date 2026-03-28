#!/usr/bin/env python
"""
Run contagion and spillover analysis.

Usage:
    python scripts/analyze_contagion.py --data data/processed/features.parquet
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.contagion import (
    DieboldYilmazSpillover,
    GrangerCausalityNetwork,
    ContagionRiskAnalyzer
)
from src.utils import setup_logging
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run contagion analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/clean/features_all.parquet",
        help="Path to processed features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/contagion",
        help="Output directory"
    )
    parser.add_argument(
        "--var-lags",
        type=int,
        default=5,
        help="VAR lags for spillover analysis"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=10,
        help="Forecast horizon for variance decomposition"
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=200,
        help="Rolling window for time-varying spillovers"
    )
    return parser.parse_args()


def prepare_returns_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare returns panel from multi-asset features.
    
    Args:
        df: DataFrame with features for multiple assets
        
    Returns:
        DataFrame with assets as columns, returns as values
    """
    if "symbol" not in df.columns:
        logger.warning("No 'symbol' column found, assuming single asset")
        if "log_return" in df.columns:
            return df[["log_return"]].dropna()
        return df
    
    # Pivot to wide format
    symbols = df["symbol"].unique()
    
    returns_dict = {}
    for symbol in symbols:
        symbol_data = df[df["symbol"] == symbol]
        if "log_return" in symbol_data.columns:
            returns_dict[symbol] = symbol_data["log_return"]
    
    if not returns_dict:
        logger.error("No return data found")
        return pd.DataFrame()
    
    returns_panel = pd.DataFrame(returns_dict)
    returns_panel = returns_panel.dropna()
    
    return returns_panel


def main():
    args = parse_args()
    
    setup_logging(log_level="INFO", log_file="logs/contagion.log")
    
    logger.info("=" * 60)
    logger.info("Contagion and Spillover Analysis")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return 1
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Prepare returns panel
    returns = prepare_returns_panel(df)
    logger.info(f"Returns panel: {returns.shape}")
    
    if returns.empty or returns.shape[1] < 2:
        logger.error("Need at least 2 assets for contagion analysis")
        return 1
    
    # Initialize analyzer
    config = {
        'var_lags': args.var_lags,
        'forecast_horizon': args.forecast_horizon,
        'granger_max_lag': 10,
        'correlation_window': 30
    }
    
    analyzer = ContagionRiskAnalyzer(config)
    
    # Run analysis
    results = analyzer.analyze(returns)
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("Saving Results")
    logger.info("=" * 60)
    
    # 1. Spillover table
    if 'spillover_returns' in results:
        spillover = results['spillover_returns']
        
        spillover.spillover_table.to_csv(output_dir / "spillover_table.csv")
        spillover.pairwise_spillovers.to_csv(output_dir / "pairwise_spillovers.csv")
        
        spillover_summary = {
            'total_spillover_index': float(spillover.total_spillover_index),
            'var_lags': spillover.var_lags,
            'forecast_horizon': spillover.forecast_horizon,
            'directional_to': spillover.directional_to.to_dict(),
            'directional_from': spillover.directional_from.to_dict(),
            'net_spillover': spillover.net_spillover.to_dict()
        }
        
        with open(output_dir / "spillover_summary.json", "w") as f:
            json.dump(spillover_summary, f, indent=2)
        
        logger.info(f"\nTotal Spillover Index: {spillover.total_spillover_index:.2f}%")
        logger.info("\nNet Spillovers (positive = net transmitter):")
        for asset, net in spillover.net_spillover.items():
            logger.info(f"  {asset}: {net:+.2f}%")
    
    # 2. Granger causality network
    if 'granger_network' in results:
        network = results['granger_network']
        
        network.adjacency_matrix.to_csv(output_dir / "granger_adjacency.csv")
        
        centrality_df = pd.DataFrame(network.centrality_measures)
        centrality_df.to_csv(output_dir / "granger_centrality.csv")
        
        network.risk_scores.to_csv(output_dir / "granger_risk_scores.csv")
        
        logger.info(f"\nGranger Network: {network.graph.number_of_edges()} significant links")
        logger.info("\nCentrality (Out-degree = influence):")
        for asset, score in network.centrality_measures['out_degree'].items():
            logger.info(f"  {asset}: {score:.3f}")
    
    # 3. Correlation regimes
    if 'correlation_regimes' in results:
        corr_regimes = results['correlation_regimes']
        corr_regimes.to_parquet(output_dir / "correlation_regimes.parquet")
        
        high_corr_pct = corr_regimes['high_correlation_regime'].mean() * 100
        logger.info(f"\nHigh correlation regime: {high_corr_pct:.1f}% of observations")
    
    # 4. Composite risk
    if 'composite_risk' in results:
        results['composite_risk'].to_csv(output_dir / "composite_risk.csv")
        
        logger.info("\nComposite Contagion Risk Ranking:")
        for asset, risk in results['composite_risk'].items():
            logger.info(f"  {asset}: {risk:.3f}")
    
    # 5. Rolling spillover analysis
    logger.info(f"\nComputing rolling spillovers (window={args.rolling_window})...")
    
    try:
        spillover_analyzer = DieboldYilmazSpillover(
            var_lags=args.var_lags,
            forecast_horizon=args.forecast_horizon
        )
        
        rolling_spillover = spillover_analyzer.rolling_spillover(
            returns,
            window=args.rolling_window,
            step=10  # Every 10 days to speed up
        )
        
        if not rolling_spillover.empty:
            rolling_spillover.to_parquet(output_dir / "rolling_spillover.parquet")
            
            logger.info(f"Rolling spillover range: {rolling_spillover['total_spillover'].min():.1f}% - {rolling_spillover['total_spillover'].max():.1f}%")
            logger.info(f"Mean rolling spillover: {rolling_spillover['total_spillover'].mean():.1f}%")
    
    except Exception as e:
        logger.warning(f"Rolling spillover analysis failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Contagion Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
