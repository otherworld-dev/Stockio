"""Backfill ML training data from historical OANDA candles.

Replaces the slow drip of live-collected samples (~10/day) with months of
historical data labelled identically to the live pipeline: symmetric
±label_atr_mult × ATR direction barriers over label_horizon_bars, with
sign-of-move labelling at horizon timeout and ambiguous bars dropped.

The existing training_data.parquet is archived (not deleted) before the
new data is written, then the model is retrained immediately so the CV
scores are visible without waiting for a bot cycle.

Usage (on the server):
    sudo -u stockio /opt/stockio/.venv/bin/python scripts/backfill_training_data.py [days]

Defaults to 180 days of M15 candles for all configured instruments.
Sentiment and calendar features are set to their no-signal defaults
(0 sentiment, no upcoming events) since historical values aren't available.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from stockio.broker.oanda import OandaBroker
from stockio.config import load_instruments, load_settings
from stockio.strategy.indicators import compute_indicators
from stockio.strategy.scorer import FEATURE_NAMES, retrain_model

_GRAN_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}


def _make_practice_broker(settings) -> OandaBroker:
    class _S:
        pass

    s = _S()
    s.oanda_account_id = settings.oanda_practice_account_id or settings.oanda_account_id
    s.oanda_api_token = settings.oanda_practice_api_token or settings.oanda_api_token
    s.oanda_environment = "practice"
    if not s.oanda_account_id or not s.oanda_api_token:
        raise SystemExit("No OANDA practice credentials found in .env")
    return OandaBroker(s)


def _fetch_history(broker, instrument: str, granularity: str, days: int) -> list:
    """Fetch historical candles in paginated 5000-bar batches."""
    bar_minutes = _GRAN_MINUTES.get(granularity, 15)
    now = datetime.now(UTC)
    cursor = now - timedelta(days=days)
    candles = []

    while cursor < now:
        batch = broker.get_candles(
            instrument=instrument,
            granularity=granularity,
            count=5000,
            from_time=cursor,
        )
        if not batch:
            break
        candles.extend(batch)
        last_ts = batch[-1].timestamp
        if last_ts <= cursor:
            break  # No forward progress — bail out
        cursor = last_ts + timedelta(minutes=bar_minutes)
        time.sleep(0.2)  # Be polite to the API

    return candles


def _build_feature_frame(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Vectorised version of indicators.build_feature_vector for all rows."""
    emas = settings.ema_periods
    feat = pd.DataFrame(index=df.index)

    feat["ema_cross_short_mid"] = df[f"ema_{emas[0]}"] - df[f"ema_{emas[1]}"]
    feat["ema_cross_mid_long"] = df[f"ema_{emas[1]}"] - df[f"ema_{emas[2]}"]

    macd_col = f"MACDh_{settings.macd_fast}_{settings.macd_slow}_{settings.macd_signal}"
    feat["macd_histogram"] = df.get(macd_col, 0.0)

    for period in settings.rsi_periods:
        feat[f"rsi_{period}"] = df.get(f"rsi_{period}", 50.0)

    sk, sd, ss = settings.stochastic_k, settings.stochastic_d, settings.stochastic_smooth
    feat["stoch_k"] = df.get(f"STOCHk_{sk}_{sd}_{ss}", 50.0)
    feat["stoch_d"] = df.get(f"STOCHd_{sk}_{sd}_{ss}", 50.0)

    feat["atr"] = df["atr"]

    bbl = df.get(f"BBL_{settings.bb_period}_{settings.bb_std}")
    bbu = df.get(f"BBU_{settings.bb_period}_{settings.bb_std}")
    if bbl is not None and bbu is not None:
        bb_range = bbu - bbl
        feat["bb_percent_b"] = np.where(
            bb_range > 0, (df["close"] - bbl) / bb_range.replace(0, np.nan), 0.5
        )
    else:
        feat["bb_percent_b"] = 0.5

    feat["adx"] = df.get(f"ADX_{settings.adx_period}", 0.0)

    ema_long = df[f"ema_{emas[-1]}"]
    feat["close_vs_ema_long"] = np.where(
        ema_long != 0, (df["close"] - ema_long) / ema_long, 0.0
    )
    feat["range_vs_atr"] = np.where(
        df["atr"] > 0, (df["high"] - df["low"]) / df["atr"], 1.0
    )

    # Sentiment — unavailable historically
    feat["sentiment"] = 0.0

    # Temporal features (mirrors indicators.build_temporal_features)
    hours = df.index.hour
    weekday = df.index.weekday
    feat["session_asia"] = ((hours >= 22) | (hours < 7)).astype(float)
    feat["session_london"] = ((hours >= 8) & (hours < 17)).astype(float)
    feat["session_newyork"] = ((hours >= 13) & (hours < 22)).astype(float)
    feat["session_overlap"] = ((hours >= 13) & (hours < 17)).astype(float)
    feat["day_of_week"] = weekday / 4.0
    feat["hour_sin"] = np.sin(2 * math.pi * hours / 24)
    feat["hour_cos"] = np.cos(2 * math.pi * hours / 24)

    # Calendar features — no-event defaults (matches calendar.py)
    feat["hours_until_high_event"] = 1.0
    feat["hours_until_medium_event"] = 1.0
    feat["is_event_window"] = 0.0
    feat["events_next_4h"] = 0.0

    return feat


def _label_rows(df: pd.DataFrame, settings) -> tuple[np.ndarray, np.ndarray]:
    """Label each row with direction outcome. Returns (labels, exit_prices).

    label 1 = price hit +label_atr_mult*ATR before -label_atr_mult*ATR
    label 0 = the reverse
    label -1 = ambiguous (both barriers in one bar) or unresolvable — drop
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    n = len(df)
    horizon = settings.label_horizon_bars
    mult = settings.label_atr_mult

    labels = np.full(n, -1, dtype=np.int8)
    exit_prices = np.zeros(n)

    for i in range(n - horizon):
        if atr[i] <= 0:
            continue
        up = close[i] + atr[i] * mult
        down = close[i] - atr[i] * mult

        resolved = False
        for j in range(i + 1, i + 1 + horizon):
            hit_up = high[j] >= up
            hit_down = low[j] <= down
            if hit_up and hit_down:
                resolved = True  # Ambiguous bar — leave label -1 (dropped)
                break
            if hit_up:
                labels[i] = 1
                exit_prices[i] = up
                resolved = True
                break
            if hit_down:
                labels[i] = 0
                exit_prices[i] = down
                resolved = True
                break

        if not resolved:
            # Horizon timeout — label by sign of net move
            labels[i] = 1 if close[i + horizon] > close[i] else 0
            exit_prices[i] = close[i + horizon]

    return labels, exit_prices


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = load_settings()
    instruments = load_instruments()
    broker = _make_practice_broker(settings)
    granularity = settings.granularity

    print(f"Backfilling {days} days of {granularity} candles "
          f"for {len(instruments)} instruments...")

    all_rows: list[pd.DataFrame] = []
    for name in instruments:
        candles = _fetch_history(broker, name, granularity, days)
        if len(candles) < 100:
            print(f"  {name}: only {len(candles)} candles — skipped")
            continue

        df = compute_indicators(candles, settings)
        feat = _build_feature_frame(df, settings)
        labels, exit_prices = _label_rows(df, settings)

        keep = labels >= 0
        rows = feat[keep].copy()
        rows.insert(0, "timestamp", df.index[keep].map(lambda t: t.isoformat()))
        rows.insert(1, "instrument", name)
        rows.insert(2, "direction", "HOLD")
        rows.insert(3, "confidence", 0.0)
        rows.insert(4, "entry_price", df["close"].to_numpy()[keep])
        rows.insert(5, "exit_price", exit_prices[keep])
        rows["label"] = labels[keep].astype(int)

        up_pct = rows["label"].mean()
        dropped = int((labels == -1).sum())
        print(f"  {name}: {len(candles)} candles -> {len(rows)} samples "
              f"(label balance: {up_pct:.1%} up, {dropped} dropped)")
        all_rows.append(rows)

    if not all_rows:
        raise SystemExit("No data collected — check OANDA credentials/connectivity")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Verify all expected feature columns exist
    missing = [f for f in FEATURE_NAMES if f not in combined.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from backfill: {missing}")

    # Archive existing training data (mislabelled legacy rows — don't mix)
    parquet_path = settings.data_dir / "training_data.parquet"
    if parquet_path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        archive = parquet_path.with_name(f"training_data_legacy_{stamp}.parquet")
        parquet_path.rename(archive)
        print(f"Archived old training data to {archive.name}")

    combined.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(combined)} samples "
          f"(overall label balance: {combined['label'].mean():.1%} up) "
          f"to {parquet_path}")

    print("Retraining model...")
    ok = retrain_model(settings, settings.data_dir, settings.models_dir)
    if ok:
        meta = (settings.models_dir / "model_meta.json").read_text()
        print(f"Retrain succeeded. Metadata:\n{meta}")
        print("\nRestart stockio-web so the bots load the new model.")
    else:
        print("Retrain was rejected by the quality gate or failed — "
              "check logs. The bots will fall back to rules-based scoring.")


if __name__ == "__main__":
    main()
