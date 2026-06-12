"""Sweep label horizon × barrier configurations to find one with real edge.

The 1-hour / ±1 ATR config showed CV AUC 0.515 (no edge). Before giving up
on the ML model, this tests whether the same features predict direction at
longer horizons / wider barriers — where trend persistence actually lives.

Fetches candles once per instrument, then for each (horizon, barrier_mult)
combination relabels the data and runs walk-forward CV, reporting AUC.
Nothing is deployed — this is a read-only experiment.

Usage (on the server):
    sudo -u stockio /opt/stockio/.venv/bin/python /opt/stockio/scripts/sweep_label_configs.py [days]

If a config clears AUC 0.53, update label_horizon_bars / label_atr_mult in
config/settings.toml, re-run backfill_training_data.py, and restart.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from stockio.config import load_instruments, load_settings
from stockio.strategy.indicators import compute_indicators
from stockio.strategy.scorer import FEATURE_NAMES

# Sibling import — works when run as a script from the scripts/ directory
from backfill_training_data import (
    _build_feature_frame,
    _fetch_history,
    _label_rows,
    _make_practice_broker,
)

# Horizons in M15 bars: 1h, 2h, 4h, 12h, 24h
HORIZONS = [4, 8, 16, 48, 96]
BARRIER_MULTS = [1.0, 1.5, 2.0]

_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_threads": 2,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}


def _cv_auc(features: np.ndarray, labels: np.ndarray, gap: int) -> float:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5, gap=gap)
    aucs = []
    for train_idx, test_idx in tscv.split(features):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        if len(set(y_test)) < 2:
            continue
        train_data = lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_NAMES)
        valid_data = lgb.Dataset(x_test, label=y_test, feature_name=FEATURE_NAMES)
        model = lgb.train(
            _LGB_PARAMS,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        aucs.append(roc_auc_score(y_test, model.predict(x_test)))
    return sum(aucs) / len(aucs) if aucs else 0.0


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = load_settings()
    instruments = load_instruments()
    broker = _make_practice_broker(settings)
    granularity = settings.granularity

    print(f"Fetching {days} days of {granularity} candles "
          f"for {len(instruments)} instruments (once)...")

    data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name in instruments:
        candles = _fetch_history(broker, name, granularity, days)
        if len(candles) < 200:
            print(f"  {name}: only {len(candles)} candles — skipped")
            continue
        df = compute_indicators(candles, settings)
        feat = _build_feature_frame(df, settings)
        data[name] = (df, feat)
        print(f"  {name}: {len(df)} usable bars")

    if not data:
        raise SystemExit("No data collected")

    n_instruments = len(data)
    results = []
    total = len(HORIZONS) * len(BARRIER_MULTS)
    print(f"\nSweeping {total} configurations "
          f"({len(HORIZONS)} horizons x {len(BARRIER_MULTS)} barrier mults)...\n")

    for horizon in HORIZONS:
        for mult in BARRIER_MULTS:
            frames = []
            for name, (df, feat) in data.items():
                labels, _ = _label_rows(df, horizon, mult)
                keep = labels >= 0
                f = feat[keep].copy()
                f["label"] = labels[keep]
                f["_ts"] = df.index[keep]
                frames.append(f)

            combined = pd.concat(frames).sort_values("_ts")
            x = combined[FEATURE_NAMES].fillna(0).to_numpy()
            y = combined["label"].to_numpy()

            # Gap must exceed the label lookahead so test folds can't see
            # outcomes that overlap training rows (rows are pooled across
            # instruments, so the gap is horizon × instrument count).
            gap = max(48, horizon * n_instruments)
            auc = _cv_auc(x, y, gap)
            balance = y.mean()
            results.append((horizon, mult, len(y), balance, auc))
            hours = horizon * 0.25
            print(f"  horizon={horizon:>3} bars ({hours:>4.1f}h)  "
                  f"barrier={mult:.1f} ATR  samples={len(y):>7}  "
                  f"up-balance={balance:.1%}  CV AUC={auc:.4f}")

    results.sort(key=lambda r: r[4], reverse=True)
    best = results[0]
    print("\n=== Results (best first) ===")
    for horizon, mult, n, balance, auc in results:
        marker = " <-- has edge" if auc >= 0.53 else ""
        print(f"  horizon={horizon:>3} barrier={mult:.1f}  AUC={auc:.4f}{marker}")

    if best[4] >= 0.53:
        print(f"\nBest config: label_horizon_bars={best[0]}, "
              f"label_atr_mult={best[1]} (AUC {best[4]:.4f})")
        print("Update these in config/settings.toml, re-run "
              "backfill_training_data.py, and restart stockio-web.")
    else:
        print(f"\nNo configuration cleared AUC 0.53 (best: {best[4]:.4f}).")
        print("These features don't predict direction at any tested horizon — "
              "the honest conclusion is to keep the paper bot on rules-based "
              "scoring, or invest in better features (multi-timeframe, "
              "cross-pair, returns-based).")


if __name__ == "__main__":
    main()
