#!/usr/bin/env python
"""
Train Hidden Markov Model for regime detection.

Usage:
    python scripts/train_hmm.py --data data/processed/features.parquet --n-states 2 3 4
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hmm import StablecoinHMM, HMMModelSelector, interpret_hmm_states
from src._clean.processing import DataProcessor
from src.utils import setup_logging, load_config
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HMM for regime detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/clean/features_all.parquet",
        help="Path to processed features"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Number of states to try"
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="Number of initializations per model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/models/hmm",
        help="Output directory for models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/run_models.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Specific stablecoin symbol (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(log_level="INFO", log_file="logs/hmm_training.log")
    
    logger.info("=" * 60)
    logger.info("HMM Training for Regime Detection")
    logger.info("=" * 60)
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return 1
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Filter to specific symbol if provided
    if args.symbol and "symbol" in df.columns:
        df = df[df["symbol"] == args.symbol]
        logger.info(f"Filtered to {args.symbol}: {len(df)} rows")
    
    # Prepare features for HMM
    feature_cols = [
        "abs_pct_deviation",
        "volatility_30d",
        "log_return",
        "volume_ratio_7d"
    ]
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using features: {available_features}")
    
    if len(available_features) < 2:
        logger.error("Not enough features available for HMM")
        return 1
    
    # Drop NaN and prepare feature matrix
    df_clean = df[available_features].dropna()
    X = df_clean.values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Model selection
    logger.info(f"\nTesting {len(args.n_states)} state configurations: {args.n_states}")
    
    selector = HMMModelSelector(
        state_range=args.n_states,
        criterion="bic"
    )
    
    optimal_states, selection_results = selector.select(X, feature_names=available_features)
    
    # Save selection results
    selection_summary = {
        "optimal_states": optimal_states,
        "results": {
            n: {
                "log_likelihood": float(r["log_likelihood"]),
                "aic": float(r["aic"]),
                "bic": float(r["bic"])
            }
            for n, r in selection_results.items()
        }
    }
    
    with open(output_dir / "model_selection.json", "w") as f:
        json.dump(selection_summary, f, indent=2)
    
    logger.info(f"\nOptimal number of states: {optimal_states}")
    
    # Train final model with optimal states
    logger.info(f"\nTraining final model with {optimal_states} states...")
    
    final_model = StablecoinHMM(
        n_states=optimal_states,
        n_init=args.n_init,
        covariance_type="full"
    )
    
    results = final_model.fit(X, feature_names=available_features)
    
    # Save model
    model_path = output_dir / f"hmm_{optimal_states}states.pkl"
    final_model.save(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Interpret states
    df_with_states = df_clean.copy()
    df_with_states["state"] = results.states
    
    interpretations = interpret_hmm_states(
        results,
        df_with_states,
        deviation_col="abs_pct_deviation" if "abs_pct_deviation" in df_with_states.columns else available_features[0],
        volatility_col="volatility_30d" if "volatility_30d" in df_with_states.columns else available_features[0]
    )
    
    logger.info("\nState Interpretations:")
    for state, label in interpretations.items():
        logger.info(f"  State {state}: {label}")
    
    # Save state assignments
    states_df = pd.DataFrame({
        "timestamp": df_clean.index,
        "state": results.states,
        "state_prob_0": results.state_probs[:, 0],
        "state_prob_1": results.state_probs[:, 1] if optimal_states > 1 else 0,
    })
    
    if optimal_states > 2:
        for i in range(2, optimal_states):
            states_df[f"state_prob_{i}"] = results.state_probs[:, i]
    
    states_df.to_parquet(output_dir / "state_assignments.parquet")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Optimal states: {optimal_states}")
    logger.info(f"Log-likelihood: {results.log_likelihood:.2f}")
    logger.info(f"AIC: {results.aic:.2f}")
    logger.info(f"BIC: {results.bic:.2f}")
    
    logger.info("\nTransition Matrix:")
    logger.info(np.array2string(results.transition_matrix, precision=3))
    
    logger.info("\nState Means:")
    for i, (state, label) in enumerate(interpretations.items()):
        logger.info(f"  {label}: {results.means[i]}")
    
    logger.info("\nState Proportions:")
    for state in range(optimal_states):
        prop = (results.states == state).mean() * 100
        label = interpretations.get(state, f"state_{state}")
        logger.info(f"  {label}: {prop:.1f}%")
    
    logger.info("\nHMM training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
