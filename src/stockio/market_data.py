"""Fetch and prepare market data for analysis and ML training."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from stockio.config import get_logger

log = get_logger(__name__)


def fetch_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV history for *ticker* from Yahoo Finance.

    Returns a DataFrame indexed by date with columns:
        Open, High, Low, Close, Volume
    """
    log.info("Fetching %s history (period=%s, interval=%s)", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        log.warning("No data returned for %s", ticker)
        return df

    # yfinance may return MultiIndex columns for single tickers — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only what we need
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.dropna(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a rich set of technical indicators used as ML features.

    Operates in-place on *df* (which must have OHLCV columns) and returns it.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # --- Trend ---
    df["sma_10"] = ta.trend.sma_indicator(c, window=10)
    df["sma_30"] = ta.trend.sma_indicator(c, window=30)
    df["ema_10"] = ta.trend.ema_indicator(c, window=10)
    df["ema_30"] = ta.trend.ema_indicator(c, window=30)
    macd = ta.trend.MACD(c)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["adx"] = ta.trend.ADXIndicator(h, l, c).adx()

    # --- Momentum ---
    df["rsi"] = ta.momentum.rsi(c, window=14)
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["williams_r"] = ta.momentum.williams_r(h, l, c)
    df["roc"] = ta.momentum.roc(c, window=10)

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(c)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c).average_true_range()

    # --- Volume ---
    df["obv"] = ta.volume.on_balance_volume(c, v)
    df["vwap"] = ta.volume.volume_weighted_average_price(h, l, c, v)

    # --- Derived ---
    df["close_pct"] = c.pct_change()
    df["volume_pct"] = v.pct_change()
    df["high_low_range"] = (h - l) / c

    df.dropna(inplace=True)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    forecast_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X (features) and y (target) arrays from an indicator-enriched DataFrame.

    Target: 1 if Close is higher *forecast_horizon* days from now, else 0.
    Returns (X, y, feature_names).
    """
    feature_cols = [
        c
        for c in df.columns
        if c not in ("Open", "High", "Low", "Close", "Volume")
    ]

    # Target: will the price go up over the next N days?
    df = df.copy()
    df["target"] = (df["Close"].shift(-forecast_horizon) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols


def get_latest_price(ticker: str) -> float | None:
    """Return the most recent closing price for *ticker*."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        log.error("Failed to get price for %s: %s", ticker, exc)
        return None


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Batch-fetch latest closing prices for multiple tickers."""
    prices: dict[str, float] = {}
    for ticker in tickers:
        price = get_latest_price(ticker)
        if price is not None:
            prices[ticker] = price
    return prices
