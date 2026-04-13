"""Generate ML training data from historical candles.

Fetches historical price data for all configured instruments, computes
indicators at each bar, and labels each bar with the actual outcome
(did price move up by ATR threshold within the horizon period?).

The output is a training_data.parquet file in the same format the live
OutcomeTracker produces, so retrain_model() can consume it directly.

Usage:
    stockio backtest              # Generate training data + train model
    stockio backtest --no-train   # Generate data only, skip training
    stockio backtest --bars 500   # Use 500 bars of history (default: 1000)
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import structlog

from stockio.broker.models import Direction
from stockio.config import Settings, load_instruments, load_settings
from stockio.strategy.indicators import (
    build_feature_vector,
    build_temporal_features,
    compute_indicators,
)
from stockio.strategy.scorer import FEATURE_NAMES, retrain_model

log = structlog.get_logger()


def _build_features_at_bar(df: pd.DataFrame, idx: int, settings: Settings) -> dict[str, float]:
    """Build a feature vector from a slice of the indicator DataFrame ending at idx."""
    row = df.iloc[idx]
    features: dict[str, float] = {}

    emas = settings.ema_periods
    if len(emas) >= 2:
        features["ema_cross_short_mid"] = float(
            row.get(f"ema_{emas[0]}", 0) - row.get(f"ema_{emas[1]}", 0)
        )
    if len(emas) >= 3:
        features["ema_cross_mid_long"] = float(
            row.get(f"ema_{emas[1]}", 0) - row.get(f"ema_{emas[2]}", 0)
        )

    macd_hist_col = f"MACDh_{settings.macd_fast}_{settings.macd_slow}_{settings.macd_signal}"
    features["macd_histogram"] = float(row.get(macd_hist_col, 0))

    for period in settings.rsi_periods:
        features[f"rsi_{period}"] = float(row.get(f"rsi_{period}", 50))

    sk, sd, ss = settings.stochastic_k, settings.stochastic_d, settings.stochastic_smooth
    features["stoch_k"] = float(row.get(f"STOCHk_{sk}_{sd}_{ss}", 50))
    features["stoch_d"] = float(row.get(f"STOCHd_{sk}_{sd}_{ss}", 50))

    features["atr"] = float(row.get("atr", 0))

    bb_lower = row.get(f"BBL_{settings.bb_period}_{settings.bb_std}", 0)
    bb_upper = row.get(f"BBU_{settings.bb_period}_{settings.bb_std}", 0)
    bb_range = bb_upper - bb_lower
    features["bb_percent_b"] = float((row["close"] - bb_lower) / bb_range) if bb_range > 0 else 0.5

    features["adx"] = float(row.get(f"ADX_{settings.adx_period}", 0))

    longest_ema = f"ema_{emas[-1]}" if emas else None
    if longest_ema and row.get(longest_ema, 0) != 0:
        features["close_vs_ema_long"] = float(
            (row["close"] - row[longest_ema]) / row[longest_ema]
        )
    else:
        features["close_vs_ema_long"] = 0.0

    atr_val = features["atr"]
    features["range_vs_atr"] = float((row["high"] - row["low"]) / atr_val) if atr_val > 0 else 1.0

    # Temporal features from the bar's timestamp
    features.update(build_temporal_features(df.index[idx]))

    # Calendar features (not available historically, zero-fill)
    features.setdefault("hours_until_high_event", 1.0)
    features.setdefault("hours_until_medium_event", 1.0)
    features.setdefault("is_event_window", 0.0)
    features.setdefault("events_next_4h", 0.0)

    # Sentiment (not available historically, zero-fill)
    features.setdefault("sentiment", 0.0)

    return features


def generate_training_data(
    settings: Settings,
    instruments: list[str],
    bars: int = 1000,
) -> pd.DataFrame:
    """Fetch historical data and generate labelled training rows."""
    from stockio.broker.oanda import OandaBroker

    # Use practice account for historical data
    class _BacktestSettings:
        pass

    s = _BacktestSettings()
    s.oanda_account_id = settings.oanda_practice_account_id or settings.oanda_account_id
    s.oanda_api_token = settings.oanda_practice_api_token or settings.oanda_api_token
    s.oanda_environment = "practice"

    broker = OandaBroker(s)

    horizon = settings.label_horizon_bars
    atr_mult = settings.label_atr_mult
    all_rows: list[dict] = []

    for instrument in instruments:
        log.info("backtest_fetching", instrument=instrument, bars=bars)

        try:
            candles = broker.get_candles(
                instrument=instrument,
                granularity=settings.granularity,
                count=bars,
            )
        except Exception as exc:
            log.warning("backtest_fetch_failed", instrument=instrument, error=str(exc))
            continue

        if len(candles) < settings.lookback_bars + horizon:
            log.warning(
                "backtest_insufficient_data",
                instrument=instrument,
                candles=len(candles),
            )
            continue

        # Compute indicators for the full series
        df = compute_indicators(candles, settings)

        if df.empty:
            continue

        # Walk through each bar (leaving room for the horizon lookahead)
        generated = 0
        for i in range(len(df) - horizon):
            features = _build_features_at_bar(df, i, settings)
            entry_price = df.iloc[i]["close"]
            atr = features.get("atr", 0)

            if atr <= 0:
                continue

            # Look ahead: did price go up by threshold?
            future_prices = df.iloc[i + 1 : i + 1 + horizon]["close"]
            max_future = future_prices.max()
            threshold = atr * atr_mult
            label = 1 if (max_future - entry_price) >= threshold else 0

            row = {
                "timestamp": df.index[i].isoformat() if hasattr(df.index[i], "isoformat") else str(df.index[i]),
                "instrument": instrument,
                "direction": "BUY" if label == 1 else "SELL",
                "confidence": 0.0,  # Not applicable for historical
                "entry_price": entry_price,
                "exit_price": float(df.iloc[i + horizon]["close"]),
                "atr": atr,
                "label": label,
            }
            for name in FEATURE_NAMES:
                row[name] = features.get(name, 0.0)

            all_rows.append(row)
            generated += 1

        log.info(
            "backtest_generated",
            instrument=instrument,
            rows=generated,
            label_1_pct=round(
                sum(1 for r in all_rows[-generated:] if r["label"] == 1) / max(generated, 1) * 100, 1
            ),
        )

    return pd.DataFrame(all_rows)


def run_backtest(bars: int = 1000, train: bool = True) -> None:
    """Generate historical training data and optionally train the model."""
    settings = load_settings()
    instruments = load_instruments()

    log.info("backtest_start", instruments=len(instruments), bars=bars)

    df = generate_training_data(settings, instruments, bars=bars)

    if df.empty:
        log.error("backtest_no_data")
        return

    # Save to parquet (same location the live system uses)
    parquet_path = settings.data_dir / "training_data.parquet"
    df.to_parquet(parquet_path, index=False)
    log.info(
        "backtest_saved",
        path=str(parquet_path),
        total_rows=len(df),
        instruments=df["instrument"].nunique(),
        label_balance=f"{df['label'].mean():.1%} positive",
    )

    if train:
        log.info("backtest_training_model")
        success = retrain_model(settings, settings.data_dir, settings.models_dir)
        if success:
            log.info("backtest_model_trained")
        else:
            log.warning("backtest_training_failed")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Backtest complete: {len(df)} training samples from {df['instrument'].nunique()} instruments")
    print(f"Label balance: {df['label'].mean():.1%} positive (price went up by ATR threshold)")
    print(f"Saved to: {parquet_path}")
    if train:
        model_path = settings.models_dir / "lightgbm_model.txt"
        print(f"Model: {'trained' if model_path.exists() else 'NOT trained (insufficient data?)'}")
    print(f"{'='*60}\n")
