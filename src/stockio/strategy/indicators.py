"""Technical indicator computation via pandas-ta."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from stockio.broker.models import Candle
from stockio.config import Settings


def compute_indicators(candles: list[Candle], settings: Settings) -> pd.DataFrame:
    """Convert candles to DataFrame and compute all configured indicators.

    Returns the DataFrame with indicator columns appended and NaN warmup rows dropped.
    """
    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # EMA
    for period in settings.ema_periods:
        df[f"ema_{period}"] = ta.ema(df["close"], length=period)

    # MACD
    macd = ta.macd(
        df["close"],
        fast=settings.macd_fast,
        slow=settings.macd_slow,
        signal=settings.macd_signal,
    )
    if macd is not None:
        df = df.join(macd)

    # RSI
    for period in settings.rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

    # Stochastic
    stoch = ta.stoch(
        df["high"],
        df["low"],
        df["close"],
        k=settings.stochastic_k,
        d=settings.stochastic_d,
        smooth_k=settings.stochastic_smooth,
    )
    if stoch is not None:
        df = df.join(stoch)

    # ATR
    df["atr"] = ta.atr(
        df["high"], df["low"], df["close"], length=settings.atr_period
    )

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=settings.bb_period, std=settings.bb_std)
    if bbands is not None:
        df = df.join(bbands)

    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"], length=settings.adx_period)
    if adx is not None:
        df = df.join(adx)

    df.dropna(inplace=True)
    return df


def build_feature_vector(df: pd.DataFrame, settings: Settings) -> dict[str, float]:
    """Extract a flat feature dict from the latest row of the indicator DataFrame."""
    if df.empty:
        return {}

    row = df.iloc[-1]
    features: dict[str, float] = {}

    # EMA cross signals
    emas = settings.ema_periods
    if len(emas) >= 2:
        features["ema_cross_short_mid"] = float(
            row.get(f"ema_{emas[0]}", 0) - row.get(f"ema_{emas[1]}", 0)
        )
    if len(emas) >= 3:
        features["ema_cross_mid_long"] = float(
            row.get(f"ema_{emas[1]}", 0) - row.get(f"ema_{emas[2]}", 0)
        )

    # MACD histogram
    macd_hist_col = f"MACDh_{settings.macd_fast}_{settings.macd_slow}_{settings.macd_signal}"
    features["macd_histogram"] = float(row.get(macd_hist_col, 0))

    # RSI
    for period in settings.rsi_periods:
        features[f"rsi_{period}"] = float(row.get(f"rsi_{period}", 50))

    # Stochastic
    sk, sd, ss = settings.stochastic_k, settings.stochastic_d, settings.stochastic_smooth
    stoch_k_col = f"STOCHk_{sk}_{sd}_{ss}"
    stoch_d_col = f"STOCHd_{sk}_{sd}_{ss}"
    features["stoch_k"] = float(row.get(stoch_k_col, 50))
    features["stoch_d"] = float(row.get(stoch_d_col, 50))

    # ATR
    features["atr"] = float(row.get("atr", 0))

    # Bollinger %B
    bb_lower = row.get(f"BBL_{settings.bb_period}_{settings.bb_std}", 0)
    bb_upper = row.get(f"BBU_{settings.bb_period}_{settings.bb_std}", 0)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        features["bb_percent_b"] = float((row["close"] - bb_lower) / bb_range)
    else:
        features["bb_percent_b"] = 0.5

    # ADX
    features["adx"] = float(row.get(f"ADX_{settings.adx_period}", 0))

    # Price relative to longest EMA
    longest_ema = f"ema_{emas[-1]}" if emas else None
    if longest_ema and row.get(longest_ema, 0) != 0:
        features["close_vs_ema_long"] = float(
            (row["close"] - row[longest_ema]) / row[longest_ema]
        )
    else:
        features["close_vs_ema_long"] = 0.0

    # Candle range relative to ATR
    atr_val = features["atr"]
    if atr_val > 0:
        features["range_vs_atr"] = float((row["high"] - row["low"]) / atr_val)
    else:
        features["range_vs_atr"] = 1.0

    # Temporal features
    features.update(build_temporal_features(df.index[-1]))

    return features


def build_temporal_features(timestamp) -> dict[str, float]:
    """Compute time-of-day and session features from a candle timestamp."""
    import math

    # Handle pandas Timestamp or datetime
    if hasattr(timestamp, "hour"):
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 4=Friday
    else:
        hour = 12
        weekday = 2

    features: dict[str, float] = {}

    # Session flags (UTC hours)
    features["session_asia"] = 1.0 if (hour >= 22 or hour < 7) else 0.0
    features["session_london"] = 1.0 if 8 <= hour < 17 else 0.0
    features["session_newyork"] = 1.0 if 13 <= hour < 22 else 0.0
    features["session_overlap"] = 1.0 if 13 <= hour < 17 else 0.0

    # Day of week (normalized 0-1)
    features["day_of_week"] = weekday / 4.0

    # Cyclical hour encoding (so 23:00 and 01:00 are close)
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)

    return features
