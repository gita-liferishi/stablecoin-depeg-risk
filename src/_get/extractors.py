"""
Data extraction module for stablecoin price and market data.

This module handles fetching data from:
- Binance API (prices - no auth needed, full history)
- DefiLlama API (supply, pools - no auth needed)
- Local CSV for UST (delisted asset)
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Global project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class BinanceClient:
    """
    Client for Binance API - no authentication needed.
    Full historical data available back to listing date.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Binance client."""
        self.cache_dir = cache_dir or PROJECT_ROOT / "data" / "cache" / "binance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        logger.info("Initialized Binance client (no auth needed)")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Union[Dict, List]:
        """Make request to Binance API."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance request error: {e}")
            raise
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str = "2020-01-01",
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'USDCUSDT')
            interval: Candle interval ('1d', '1h', '1m', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            limit: Max candles per request (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit
            }
            
            try:
                data = self._make_request("klines", params)
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1  # Next timestamp after last candle
                
                # Rate limiting - be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break
        
        if not all_data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df = df.set_index("timestamp")
        
        # Keep only relevant columns
        df = df[["open", "high", "low", "close", "volume"]]
        
        logger.info(f"  ✓ {symbol}: {len(df)} rows")
        
        return df
    
    def get_stablecoin_price(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get stablecoin price in USD terms.
        
        For stablecoins paired with USDT, we get the direct price.
        
        Args:
            symbol: Stablecoin symbol (e.g., 'USDC', 'DAI')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with price column
        """
        pair = f"{symbol}USDT"
        
        try:
            df = self.get_klines(pair, "1d", start_date, end_date)
            
            if df.empty:
                return df
            
            # Use close price as the daily price
            result = pd.DataFrame({
                "price": df["close"],
                "volume": df["volume"],
                "high": df["high"],
                "low": df["low"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()


class DefiLlamaClient:
    """
    Client for DefiLlama API - no rate limits, comprehensive DeFi data.
    """
    
    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"
    YIELDS_URL = "https://yields.llama.fi"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize DefiLlama client."""
        self.cache_dir = cache_dir or PROJECT_ROOT / "data" / "cache" / "defillama"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        logger.info("Initialized DefiLlama client")
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make request to DefiLlama API."""
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"DefiLlama request error: {e}")
            raise
    
    def get_stablecoins(self, include_prices: bool = True) -> pd.DataFrame:
        """Get list of all stablecoins with current data."""
        url = f"{self.STABLECOINS_URL}/stablecoins"
        params = {"includePrices": str(include_prices).lower()}
        
        data = self._make_request(url, params)
        
        records = []
        for coin in data.get("peggedAssets", []):
            record = {
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "gecko_id": coin.get("gecko_id"),
                "peg_type": coin.get("pegType"),
                "peg_mechanism": coin.get("pegMechanism"),
                "circulating": coin.get("circulating", {}).get("peggedUSD"),
                "price": coin.get("price")
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_stablecoin_history(
        self,
        stablecoin_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get historical supply data for a specific stablecoin."""
        url = f"{self.STABLECOINS_URL}/stablecoin/{stablecoin_id}"
        
        data = self._make_request(url)
        
        records = []
        for item in data.get("tokens", []):
            date = item.get("date")
            circulating = item.get("circulating", {})
            
            # Handle different response structures
            total = 0
            if isinstance(circulating, dict):
                # Try peggedUSD directly
                if "peggedUSD" in circulating:
                    total = circulating.get("peggedUSD", 0)
                else:
                    # Sum across chains (each chain has peggedUSD)
                    for chain, chain_data in circulating.items():
                        if isinstance(chain_data, dict):
                            total += chain_data.get("peggedUSD", 0)
                        elif isinstance(chain_data, (int, float)):
                            total += chain_data
            elif isinstance(circulating, (int, float)):
                total = circulating
            
            record = {
                "date": datetime.fromtimestamp(date) if date else None,
                "total_circulating": total,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("date").sort_index()
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
        
        return df
    
    def get_stablecoin_prices(self) -> pd.DataFrame:
        """Get historical prices for all stablecoins."""
        url = f"{self.STABLECOINS_URL}/stablecoinprices"
        data = self._make_request(url)
        
        records = []
        for item in data:
            record = {
                "date": datetime.fromtimestamp(item.get("date", 0)),
                "prices": item.get("prices", {})
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            prices_df = pd.json_normalize(df["prices"])
            df = pd.concat([df["date"], prices_df], axis=1)
            df = df.set_index("date")
        
        return df
    
    def get_pools(self, stablecoin_symbol: Optional[str] = None) -> pd.DataFrame:
        """Get liquidity pool data."""
        url = f"{self.YIELDS_URL}/pools"
        data = self._make_request(url)
        
        df = pd.DataFrame(data.get("data", []))
        
        if stablecoin_symbol and not df.empty:
            df = df[df["symbol"].str.contains(stablecoin_symbol, case=False, na=False)]
        
        return df


class DataExtractor:
    """
    Main class for extracting and combining data from multiple sources.
    Creates the MVP dataset for stablecoin de-peg risk modeling.
    """
    
    # Binance trading pairs for stablecoins
    BINANCE_PAIRS = {
        "USDT": None,  # Base quote currency - derive from BTCUSDT inverse or use 1.0
        "USDC": "USDCUSDT",
        "DAI": "DAIUSDT",
        "FRAX": "FRAXUSDT",
        "TUSD": "TUSDUSDT",
    }
    
    # Reference assets
    REFERENCE_PAIRS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
    }
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Initialize data extractor.
        
        Args:
            config: Optional configuration dict, path to YAML file, or None for defaults
        """
        # Handle different config input types
        if config is None:
            self.config = {
                "data": {
                    "date_range": {
                        "start": "2020-01-01",
                        "end": datetime.now().strftime("%Y-%m-%d")
                    }
                }
            }
        elif isinstance(config, str):
            # It's a file path - load YAML
            import yaml
            config_path = Path(config)
            if config_path.exists():
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {config}, using defaults")
                self.config = {
                    "data": {
                        "date_range": {
                            "start": "2020-01-01",
                            "end": datetime.now().strftime("%Y-%m-%d")
                        }
                    }
                }
        else:
            # It's already a dict
            self.config = config
        
        self.binance = BinanceClient()
        self.defillama = DefiLlamaClient()
        
        # Setup output directories
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.processed_dir = PROJECT_ROOT / "data" / "clean"
        self.cache_dir = PROJECT_ROOT / "data" / "cache"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DataExtractor")
    
    def load_coinmarketcap_csv(self, csv_path: Path, symbol: str) -> pd.DataFrame:
        """
        Load historical data from CoinMarketCap CSV.
        
        Args:
            csv_path: Path to CSV file
            symbol: Stablecoin symbol (USDT, USDC, etc.)
            
        Returns:
            DataFrame with OHLCV + market cap data
        """
        logger.info(f"Loading {symbol} from {csv_path}")
        
        try:
            # CoinMarketCap uses semicolon delimiter
            df = pd.read_csv(csv_path, delimiter=";")
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Parse date from timeOpen
            date_col = None
            for col in ["timeopen", "date", "timestamp"]:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.error(f"No date column found in {csv_path}")
                return pd.DataFrame()
            
            # Parse date and remove timezone
            df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df = df.set_index("date")
            
            # Rename columns to standard format
            rename_map = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "price",
                "volume": "volume",
                "marketcap": "market_cap",
                "circulatingsupply": "circulating_supply"
            }
            df = df.rename(columns=rename_map)
            
            # Ensure numeric columns
            numeric_cols = ["open", "high", "low", "price", "volume", "market_cap", "circulating_supply"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Sort by date
            df = df.sort_index()
            
            # Add identifiers
            df["symbol"] = symbol
            df["coin_id"] = symbol.lower()
            
            # Keep only relevant columns
            keep_cols = ["open", "high", "low", "price", "volume", "market_cap", "circulating_supply", "symbol", "coin_id"]
            df = df[[c for c in keep_cols if c in df.columns]]
            
            logger.info(f"  ✓ {symbol}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {symbol} CSV: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def extract_price_data(self) -> pd.DataFrame:
        """
        Extract price data for all stablecoins from CoinMarketCap CSVs.
        
        Looks for CSV files in data/raw/ with names like:
        - usdt_coinmarketcap.csv
        - usdc_coinmarketcap.csv
        - dai_coinmarketcap.csv
        - frax_coinmarketcap.csv
        - ust_coinmarketcap.csv
        
        Returns:
            DataFrame with OHLCV + market cap data for all assets
        """
        logger.info("Loading stablecoin data from CoinMarketCap CSVs...")
        
        # Get date range from config
        date_range = self.config.get("data", {}).get("date_range", {})
        start_date = date_range.get("start", "2020-01-01")
        end_date = date_range.get("end", datetime.now().strftime("%Y-%m-%d"))
        
        logger.info(f"Date range filter: {start_date} to {end_date}")
        
        # Define stablecoins and their possible CSV names
        stablecoins = {
            # "USDT": ["usdt_coinmarketcap.csv", "usdt.csv", "tether.csv", "USDT.csv"],
            # "USDC": ["usdc_coinmarketcap.csv", "usdc.csv", "usd-coin.csv", "USDC.csv"],
            "DAI": ["dai_coinmarketcap.csv", "dai.csv", "DAI.csv"],
            "FRAX": ["frax_coinmarketcap.csv", "frax.csv", "FRAX.csv"],
            "UST": ["ust_coinmarketcap.csv", "ust.csv", "terrausd.csv", "UST.csv"],
            "LUSD":["lusd_coinmarketcap.csv", "lusd.csv"],
            "MIM":["mim_coinmarketcap.csv", "mim.csv"],
            "sUSD":["susd_coinmarketcap.csv", "susd.csv"],
            "USDD":["usdd_coinmarketcap.csv", "usdd.csv"]
        }
        
        dfs = []
        
        for symbol, possible_names in stablecoins.items():
            csv_path = None
            
            # Find the CSV file
            for name in possible_names:
                path = self.raw_dir / name
                if path.exists():
                    csv_path = path
                    break
            
            if csv_path is None:
                logger.warning(f"  ✗ {symbol}: No CSV found (tried: {possible_names})")
                continue
            
            # Load the CSV
            df = self.load_coinmarketcap_csv(csv_path, symbol)
            if not df.empty:
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if not df.empty:
                    dfs.append(df)
                    logger.info(f"    After date filter: {len(df)} rows")
                else:
                    logger.warning(f"    No data in date range {start_date} to {end_date}")
        
        # Combine all
        if not dfs:
            raise ValueError("No price data loaded from any CSV")
        
        combined = pd.concat(dfs)
        combined.index.name = "timestamp"
        
        # Save
        output_path = self.raw_dir / "prices_raw.parquet"
        combined.to_parquet(output_path)
        logger.info(f"Saved price data to {output_path} ({len(combined)} total rows)")
        
        # Also save as CSV for inspection
        csv_path = self.raw_dir / "prices_raw.csv"
        combined.to_csv(csv_path)
        logger.info(f"Saved CSV copy to {csv_path}")
        
        return combined
    
    def extract_supply_data(self) -> pd.DataFrame:
        """
        Extract supply data from the loaded price CSVs.
        
        CoinMarketCap CSVs include circulatingSupply, so we extract it
        from the prices_raw.parquet file.
        
        Returns:
            DataFrame with supply metrics
        """
        logger.info("Extracting supply data from price CSVs...")
        
        # Load prices (which includes circulating_supply from CoinMarketCap)
        prices_path = self.raw_dir / "prices_raw.parquet"
        
        if not prices_path.exists():
            logger.warning("prices_raw.parquet not found. Run extract_price_data first.")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(prices_path)
            
            if "circulating_supply" not in df.columns:
                logger.warning("No circulating_supply column in price data")
                return pd.DataFrame()
            
            # Extract supply columns
            supply_df = df[["symbol", "circulating_supply", "market_cap"]].copy()
            supply_df = supply_df.rename(columns={"circulating_supply": "total_circulating"})
            
            # Save
            output_path = self.raw_dir / "supply_raw.parquet"
            supply_df.to_parquet(output_path)
            logger.info(f"Saved supply data to {output_path}")
            
            # CSV copy
            supply_df.to_csv(self.raw_dir / "supply_raw.csv")
            
            # Log per-asset counts
            for symbol in supply_df["symbol"].unique():
                count = len(supply_df[supply_df["symbol"] == symbol])
                logger.info(f"  ✓ {symbol}: {count} rows")
            
            return supply_df
            
        except Exception as e:
            logger.error(f"Supply extraction failed: {e}")
            return pd.DataFrame()
    
    def extract_pool_data(self) -> pd.DataFrame:
        """Extract liquidity pool data."""
        logger.info("Fetching liquidity pool data...")
        
        try:
            pools = self.defillama.get_pools()
            
            # Filter for stablecoin pools
            stablecoin_symbols = ["DAI", "FRAX", "UST", "LUSD", "MIM", "sUSD", "USDD"]
            pattern = "|".join(stablecoin_symbols)
            
            stablecoin_pools = pools[
                pools["symbol"].str.contains(pattern, case=False, na=False)
            ]
            
            if "tvlUsd" in stablecoin_pools.columns:
                stablecoin_pools = stablecoin_pools.sort_values("tvlUsd", ascending=False)
            
            output_path = self.raw_dir / "pools_raw.parquet"
            stablecoin_pools.to_parquet(output_path)
            logger.info(f"  ✓ Pools: {len(stablecoin_pools)} stablecoin pools")
            
            # CSV copy
            stablecoin_pools.to_csv(self.raw_dir / "pools_raw.csv", index=False)
            
            return stablecoin_pools
            
        except Exception as e:
            logger.error(f"Pool extraction failed: {e}")
            return pd.DataFrame()
    
    def run_full_extraction(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete MVP data extraction pipeline.
        
        Returns:
            Dictionary of DataFrames by data type
        """
        logger.info("=" * 60)
        logger.info("Starting MVP Data Extraction")
        logger.info("=" * 60)
        logger.info(f"Date range: {self.config['data']['date_range']['start']} to {self.config['data']['date_range']['end']}")
        logger.info(f"Output directory: {self.raw_dir}")
        
        results = {}
        
        # 1. Price data
        logger.info("\n[1/3] Extracting price data...")
        try:
            results["prices"] = self.extract_price_data()
        except Exception as e:
            logger.error(f"Price extraction failed: {e}")
        
        # 2. Supply data
        logger.info("\n[2/3] Extracting supply data...")
        try:
            results["supply"] = self.extract_supply_data()
        except Exception as e:
            logger.error(f"Supply extraction failed: {e}")
        
        # 3. Pool data
        logger.info("\n[3/3] Extracting pool data...")
        try:
            results["pools"] = self.extract_pool_data()
        except Exception as e:
            logger.error(f"Pool extraction failed: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Extraction Summary")
        logger.info("=" * 60)
        
        total_rows = 0
        for name, df in results.items():
            if df is not None and len(df) > 0:
                rows = len(df)
                total_rows += rows
                logger.info(f"  {name}: {rows:,} rows")
            else:
                logger.warning(f"  {name}: No data")
        
        logger.info(f"\nTotal: {total_rows:,} rows")
        logger.info(f"Files saved to: {self.raw_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    extractor = DataExtractor()
    data = extractor.run_full_extraction()
    
    for name, df in data.items():
        if df is not None and len(df) > 0:
            print(f"\n{name}:")
            print(df.head())