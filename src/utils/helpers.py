import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level
        log_file: Optional file path for log output
    """
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="7 days"
        )


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp."""
    return int(dt.timestamp())


def resample_to_daily(df: pd.DataFrame, agg_method: str = "last") -> pd.DataFrame:
    """
    Resample time series data to daily frequency.
    
    Args:
        df: DataFrame with datetime index
        agg_method: Aggregation method ('last', 'first', 'mean', 'ohlc')
        
    Returns:
        Resampled DataFrame
    """
    if agg_method == "ohlc":
        return df.resample("D").agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum",
            "market_cap": "last"
        })
    elif agg_method == "last":
        return df.resample("D").last()
    elif agg_method == "first":
        return df.resample("D").first()
    elif agg_method == "mean":
        return df.resample("D").mean()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def align_dataframes(
    dfs: List[pd.DataFrame],
    method: str = "inner"
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common index.
    
    Args:
        dfs: List of DataFrames with datetime indices
        method: Join method ('inner', 'outer')
        
    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []
    
    # Get common index
    if method == "inner":
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
    else:
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.union(df.index)
    
    return [df.reindex(common_index) for df in dfs]


def compute_information_criteria(
    log_likelihood: float,
    n_params: int,
    n_samples: int
) -> Dict[str, float]:
    """
    Compute AIC and BIC for model selection.
    
    Args:
        log_likelihood: Log-likelihood of fitted model
        n_params: Number of model parameters
        n_samples: Number of observations
        
    Returns:
        Dictionary with AIC and BIC values
    """
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return {"aic": aic, "bic": bic}


def validate_price_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate price data quality.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "n_rows": len(df),
        "n_missing": df.isnull().sum().to_dict(),
        "date_range": (df.index.min(), df.index.max()),
        "price_range": (df["price"].min(), df["price"].max()) if "price" in df.columns else None,
        "duplicates": df.index.duplicated().sum(),
        "gaps": None
    }
    
    # Check for gaps in daily data
    if len(df) > 1:
        expected_days = (df.index.max() - df.index.min()).days + 1
        actual_days = len(df.index.normalize().unique())
        results["gaps"] = expected_days - actual_days
    
    return results


def format_depeg_report(events: List, include_details: bool = True) -> str:
    """
    Format de-peg events into a readable report.
    
    Args:
        events: List of DepegEvent objects
        include_details: Whether to include detailed statistics
        
    Returns:
        Formatted string report
    """
    if not events:
        return "No de-peg events detected."
    
    lines = [f"De-Peg Event Report ({len(events)} events)", "=" * 50]
    
    for i, event in enumerate(events, 1):
        lines.append(f"\nEvent {i}:")
        lines.append(f"  Asset: {event.coin_id}")
        lines.append(f"  Period: {event.start_time} to {event.end_time}")
        lines.append(f"  Duration: {event.duration_hours:.1f} hours")
        lines.append(f"  Severity: {event.severity}")
        lines.append(f"  Max Deviation: {event.max_deviation*100:.2f}%")
        lines.append(f"  Min Price: ${event.min_price:.4f}")
    
    return "\n".join(lines)
