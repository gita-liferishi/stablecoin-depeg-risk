
import argparse
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src._get.extractors import DataExtractor
from src.utils.helpers import setup_logging, load_config
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract stablecoin data from APIs"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",  # Updated to 2020
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "src" / "config" / "get_data.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Output directory for raw data"
    )
    parser.add_argument(
        "--stablecoins",
        type=str,
        nargs="+",
        default=None,
        help="Specific stablecoins to extract (default: all in config)"
    )
    parser.add_argument(
        "--include-ust",
        action="store_true",
        default=True,
        help="Include UST historical data (up to May 2022 collapse)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(log_level="INFO", log_file="logs/extraction.log")
    
    logger.info("=" * 60)
    logger.info("Starting data extraction")
    logger.info("=" * 60)
    
    config = load_config(args.config)
    
    if args.start_date:
        config["data"]["date_range"]["start"] = args.start_date
    if args.end_date:
        config["data"]["date_range"]["end"] = args.end_date
    else:
        config["data"]["date_range"]["end"] = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {config['data']['date_range']['start']} to {config['data']['date_range']['end']}")
    
    extractor = DataExtractor(args.config)
    
    extractor.raw_dir = Path(args.output_dir)
    extractor.raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Run extraction
    try:
        results = extractor.run_full_extraction()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Extraction Summary")
        logger.info("=" * 60)
        
        for name, df in results.items():
            if df is not None and len(df) > 0:
                logger.info(f"  {name}: {len(df)} rows, {df.shape[1]} columns")
                if hasattr(df.index, 'min'):
                    try:
                        idx_min = pd.to_datetime(df.index.min()).strftime("%Y-%m-%d")
                        idx_max = pd.to_datetime(df.index.max()).strftime("%Y-%m-%d")
                        logger.info(f"    Date range: {idx_min} to {idx_max}")
                    except:
                        logger.info(f"    Index range: {df.index.min()} to {df.index.max()}")
            else:
                logger.warning(f"  {name}: No data extracted")
        
        logger.info("\nExtraction complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
