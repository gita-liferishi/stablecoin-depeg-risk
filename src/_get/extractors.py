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
    
    def load_ust_from_csv(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load UST historical data from CoinMarketCap CSV.
        
        Args:
            csv_path: Path to UST CSV file
            
        Returns:
            DataFrame with UST price data
        """
        if csv_path is None:
            # Look for UST CSV in raw data folder
            possible_names = [
                "ust_coinmarketcap.csv",
                "ust.csv",
                "terrausd.csv",
                "UST.csv",
                "TerraUSD.csv"
            ]
            
            for name in possible_names:
                path = self.raw_dir / name
                if path.exists():
                    csv_path = path
                    break
        
        if csv_path is None or not Path(csv_path).exists():
            logger.warning("UST CSV not found. Place it in data/raw/ust_coinmarketcap.csv")
            return pd.DataFrame()
        
        logger.info(f"Loading UST from {csv_path}")
        
        try:
            try:
                df = pd.read_csv(csv_path, delimiter=";")
            except:
                df = pd.read_csv(csv_path)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Handle different date column names
            date_col = None
            for col in ["timeopen", "date", "timestamp", "time"]:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                logger.error("No date column found in UST CSV")
                return pd.DataFrame()
            
            # Parse date
            df["date"] = pd.to_datetime(df[date_col])
            df = df.set_index("date")
            
            # Rename to standard format
            rename_map = {
                "close": "price",
                "open": "open",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "marketcap": "market_cap",
            }
            df = df.rename(columns=rename_map)
            
            # Ensure price column exists
            if "price" not in df.columns:
                logger.error("No price/close column found in UST CSV")
                return pd.DataFrame()
            
            # Clean numeric columns
            for col in ["price", "open", "high", "low", "volume", "market_cap"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.sort_index()
            df["symbol"] = "UST"
            df["coin_id"] = "terrausd"
            
            logger.info(f"  ✓ UST: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load UST CSV: {e}")
            return pd.DataFrame()
    
    def extract_price_data(self) -> pd.DataFrame:
        """
        Extract price data for all stablecoins and reference assets.
        
        Uses:
        - DefiLlama for stablecoin prices (USDT, USDC, DAI, FRAX)
        - CSV for UST (delisted)
        - Note: BTC/ETH not included (use separate source if needed)
        
        Returns:
            DataFrame with price data for all assets
        """
        date_range = self.config["data"]["date_range"]
        start_date = date_range["start"]
        end_date = date_range["end"]
        
        dfs = []
        
        # 1. Get stablecoin prices from DefiLlama
        logger.info("Fetching stablecoin prices from DefiLlama...")
        
        try:
            prices_df = self.defillama.get_stablecoin_prices()
            
            if not prices_df.empty:
                # DefiLlama returns prices with gecko_id as columns
                # Map to our symbols
                symbol_map = {
                    "tether": "USDT",
                    "usd-coin": "USDC",
                    "dai": "DAI",
                    "frax": "FRAX",
                }
                
                for gecko_id, symbol in symbol_map.items():
                    if gecko_id in prices_df.columns:
                        coin_df = pd.DataFrame({
                            "price": prices_df[gecko_id],
                            "volume": np.nan,
                            "high": np.nan,
                            "low": np.nan,
                            "symbol": symbol,
                            "coin_id": gecko_id
                        })
                        coin_df.index.name = "timestamp"
                        
                        # Filter by date range
                        coin_df = coin_df[
                            (coin_df.index >= start_date) & 
                            (coin_df.index <= end_date)
                        ]
                        
                        if not coin_df.empty:
                            dfs.append(coin_df)
                            logger.info(f"  ✓ {symbol}: {len(coin_df)} rows")
                        else:
                            logger.warning(f"  ✗ {symbol}: No data in date range")
                    else:
                        logger.warning(f"  ✗ {symbol} ({gecko_id}): Not found in DefiLlama")
                        
        except Exception as e:
            logger.error(f"DefiLlama prices failed: {e}")
        
        # 2. UST from CSV
        logger.info("Loading UST from CSV...")
        ust_df = self.load_ust_from_csv()
        if not ust_df.empty:
            # Standardize columns
            ust_clean = pd.DataFrame({
                "price": ust_df["price"] if "price" in ust_df.columns else ust_df.get("close"),
                "volume": ust_df.get("volume", np.nan),
                "high": ust_df.get("high", np.nan),
                "low": ust_df.get("low", np.nan),
                "symbol": "UST",
                "coin_id": "terrausd"
            }, index=ust_df.index)
            dfs.append(ust_clean)
        
        # Combine all
        if not dfs:
            raise ValueError("No price data fetched for any asset")
        
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
        Extract supply/circulating data from DefiLlama.
        
        Returns:
            DataFrame with supply metrics
        """
        logger.info("Fetching supply data from DefiLlama...")
        
        stablecoins_list = self.defillama.get_stablecoins()
        date_range = self.config["data"]["date_range"]
        
        target_symbols = ["USDT", "USDC", "DAI", "FRAX"]
        
        dfs = []
        for symbol in target_symbols:
            match = stablecoins_list[stablecoins_list["symbol"].str.upper() == symbol.upper()]
            
            if match.empty:
                logger.warning(f"  ✗ {symbol} not found in DefiLlama")
                continue
            
            sc_id = match.iloc[0]["id"]
            
            try:
                df = self.defillama.get_stablecoin_history(
                    sc_id,
                    start_date=date_range["start"],
                    end_date=date_range["end"]
                )
                df["symbol"] = symbol
                dfs.append(df)
                logger.info(f"  ✓ {symbol}: {len(df)} rows")
            except Exception as e:
                logger.error(f"  ✗ {symbol} supply failed: {e}")
        
        if dfs:
            combined = pd.concat(dfs)
            output_path = self.raw_dir / "supply_raw.parquet"
            combined.to_parquet(output_path)
            logger.info(f"Saved supply data to {output_path}")
            
            # CSV copy
            combined.to_csv(self.raw_dir / "supply_raw.csv")
            
            return combined
        
        return pd.DataFrame()
    
    def extract_pool_data(self) -> pd.DataFrame:
        """Extract liquidity pool data."""
        logger.info("Fetching liquidity pool data...")
        
        try:
            pools = self.defillama.get_pools()
            
            # Filter for stablecoin pools
            stablecoin_symbols = ["USDT", "USDC", "DAI", "FRAX", "UST"]
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