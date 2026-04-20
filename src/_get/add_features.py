import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
START, END = "2020-01-01", "2025-12-31"

def load_stablecoins(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    keep = [
        "timestamp", "symbol", "price", "log_return",
        "deviation", "abs_deviation",
        "depeg_label", "depeg_severity", "in_depeg", "depeg_episode",
    ]
    df = df[keep].copy()
    # df = df[df["symbol"] != "UST"].reset_index(drop=True)
    df = df.sort_values(["symbol", "timestamp"])
    recomputed = df.groupby("symbol")["price"].transform(
        lambda p: np.log(p).diff()
    )
    mismatch = (df["log_return"] - recomputed).abs().mean()
    if mismatch > 1e-6:
        print(f"[warn] log_return differs from log-diff by {mismatch:.2e}; overwriting")
        df["log_return"] = recomputed

    # Drop pre-warmup rows to avoid NaN in POMP observation
    print()
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df

def fetch_fgi() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = requests.get(url, timeout=30).json()
    fgi = pd.DataFrame(r["data"])
    fgi["timestamp"] = pd.to_datetime(fgi["timestamp"], unit="s").dt.normalize()
    fgi["fgi"] = fgi["value"].astype(int)
    fgi = fgi[["timestamp", "fgi"]].sort_values("timestamp").reset_index(drop=True)

    # Scale to [-1, 1] and first-difference for stationarity
    fgi["fgi_scaled"] = (fgi["fgi"] - 50) / 50
    fgi["delta_fgi_scaled"] = fgi["fgi_scaled"].diff()

    fgi = fgi[(fgi["timestamp"] >= START) & (fgi["timestamp"] <= END)]
    return fgi[["timestamp", "fgi_scaled", "delta_fgi_scaled"]]

def fetch_crypto_returns() -> pd.DataFrame:
    tickers = yf.download(
        "ETH-USD", start=START, end=END,
        interval="1d", progress=False, auto_adjust=True,
    )
    close = tickers["Close"].copy()
    close.index = pd.to_datetime(close.index).normalize()
    close = close.rename(columns={"ETH-USD": "eth_close"})

    out = pd.DataFrame(index=close.index)
    out["eth_return"]   = np.log(close["eth_close"]).diff()
    out["eth_drawdown"] = np.maximum(0.0, -out["eth_return"])

    out = out.reset_index().rename(columns={"Date": "timestamp", "index": "timestamp"})
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    return out.dropna().reset_index(drop=True)


def build_panel(stablecoin_csv: str, curve_csv: str | None = None) -> pd.DataFrame:
    stable = load_stablecoins(stablecoin_csv)
    fgi = fetch_fgi()
    crypto = fetch_crypto_returns()

    panel = stable.merge(fgi, on="timestamp", how="left")
    panel = panel.merge(crypto, on="timestamp", how="left")
    panel["fgi_scaled"] = panel.groupby("symbol")["fgi_scaled"].ffill(limit=1)
    panel["delta_fgi_scaled"] = panel.groupby("symbol")["delta_fgi_scaled"].ffill(limit=1)

    # POMP observation in basis points
    panel["y_bp"] = (panel["log_return"] * 1e4).clip(-5000, 5000)

    # 3-day rolling max ETH drawdown.
    # Smooths point-wise zeros on event-adjacent days (e.g. May 10 2022 or Mar 11 2023
    # had ETH up but the STRESS covers t-1, t, t+1). Per-asset groupby is defensive
    # even though the underlying eth_drawdown is identical across assets on each date.
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    panel["eth_drawdown_3d"] = (
        panel.groupby("symbol")["eth_drawdown"]
             .transform(lambda s: s.rolling(3, min_periods=1).max())
    )

    essentials = ["y_bp", "delta_fgi_scaled", "eth_drawdown", "eth_drawdown_3d"]
    panel = panel.dropna(subset=essentials).reset_index(drop=True)

    return panel

if __name__ == "__main__":
    panel = build_panel("data/clean/features_all.csv", curve_csv=None)
    out_path = DATA_DIR / "pomp_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"wrote {out_path}  shape={panel.shape}")
    print(panel.groupby("symbol").size())
    print(panel.columns.tolist())