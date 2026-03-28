#!/usr/bin/env python
"""
Train Kalman Filter models for latent volatility tracking.

Usage:
    python scripts/train_kalman.py --data data/processed/features.parquet
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.kalman import (
    VolatilityKalmanFilter,
    LocalLevelModel,
    StochasticVolatilityModel
)
from src.utils import setup_logging, load_config
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Kalman Filter models"
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
        default="src/models/kalman",
        help="Output directory"
    )
    parser.add_argument(
        "--em-iterations",
        type=int,
        default=100,
        help="EM iterations for parameter estimation"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Specific stablecoin symbol"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(log_level="INFO", log_file="logs/kalman_training.log")
    
    logger.info("=" * 60)
    logger.info("Kalman Filter Training")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return 1
    
    df = pd.read_parquet(data_path)
    
    if args.symbol and "symbol" in df.columns:
        df = df[df["symbol"] == args.symbol]
        logger.info(f"Filtered to {args.symbol}: {len(df)} rows")
    
    results_summary = {}
    
    # 1. Local Level Model on price deviations
    logger.info("\n" + "-" * 40)
    logger.info("1. Local Level Model (Peg Deviation)")
    logger.info("-" * 40)
    
    if "deviation" in df.columns or "abs_pct_deviation" in df.columns:
        dev_col = "deviation" if "deviation" in df.columns else "abs_pct_deviation"
        deviations = df[dev_col].dropna().values
        
        ll_model = LocalLevelModel()
        ll_results = ll_model.fit(deviations, em_iterations=args.em_iterations)
        
        logger.info(f"Log-likelihood: {ll_results.log_likelihood:.2f}")
        logger.info(f"Observation noise: {ll_results.observation_noise[0,0]:.6f}")
        logger.info(f"Transition noise: {ll_results.transition_noise[0,0]:.6f}")
        
        # Save filtered states
        ll_states = pd.DataFrame({
            "timestamp": df[dev_col].dropna().index,
            "observed_deviation": deviations,
            "filtered_deviation": ll_results.filtered_state_means.flatten(),
            "smoothed_deviation": ll_results.smoothed_state_means.flatten()
        })
        ll_states.to_parquet(output_dir / "local_level_states.parquet")
        
        results_summary["local_level"] = {
            "log_likelihood": float(ll_results.log_likelihood),
            "observation_noise": float(ll_results.observation_noise[0, 0]),
            "transition_noise": float(ll_results.transition_noise[0, 0])
        }
    
    # 2. Volatility Kalman Filter
    logger.info("\n" + "-" * 40)
    logger.info("2. Volatility Kalman Filter")
    logger.info("-" * 40)
    
    if "log_return" in df.columns:
        returns = df["log_return"].dropna().values
        abs_returns = np.abs(returns)
        
        vol_kf = VolatilityKalmanFilter(em_iterations=args.em_iterations)
        vol_results = vol_kf.fit(abs_returns)
        
        logger.info(f"Log-likelihood: {vol_results.log_likelihood:.2f}")
        logger.info(f"Transition coefficient: {vol_results.transition_matrix[0,0]:.4f}")
        
        # Save filtered volatility
        vol_states = pd.DataFrame({
            "timestamp": df["log_return"].dropna().index,
            "abs_return": abs_returns,
            "filtered_volatility": vol_results.filtered_state_means.flatten(),
            "smoothed_volatility": vol_results.smoothed_state_means.flatten()
        })
        vol_states.to_parquet(output_dir / "volatility_kf_states.parquet")
        
        results_summary["volatility_kf"] = {
            "log_likelihood": float(vol_results.log_likelihood),
            "transition_coeff": float(vol_results.transition_matrix[0, 0]),
            "observation_noise": float(vol_results.observation_noise[0, 0])
        }
    
    # 3. Stochastic Volatility Model
    logger.info("\n" + "-" * 40)
    logger.info("3. Stochastic Volatility Model")
    logger.info("-" * 40)
    
    if "log_return" in df.columns:
        returns = df["log_return"].dropna().values
        
        # Remove zeros to avoid log issues
        returns_nonzero = returns[returns != 0]
        
        if len(returns_nonzero) > 100:
            sv_model = StochasticVolatilityModel()
            sv_results = sv_model.fit(returns_nonzero, em_iterations=args.em_iterations)
            
            logger.info(f"Log-likelihood: {sv_results.log_likelihood:.2f}")
            logger.info(f"Persistence (phi): {sv_model.phi:.4f}")
            
            # Save filtered volatility
            sv_states = pd.DataFrame({
                "return": returns_nonzero,
                "filtered_volatility": sv_results.filtered_state_means.flatten(),
                "smoothed_volatility": sv_results.smoothed_state_means.flatten()
            })
            sv_states.to_parquet(output_dir / "sv_model_states.parquet")
            
            results_summary["stochastic_volatility"] = {
                "log_likelihood": float(sv_results.log_likelihood),
                "phi": float(sv_model.phi)
            }
    
    # Save summary
    with open(output_dir / "kalman_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Kalman Filter Training Complete")
    logger.info("=" * 60)
    
    for model, metrics in results_summary.items():
        logger.info(f"\n{model}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
