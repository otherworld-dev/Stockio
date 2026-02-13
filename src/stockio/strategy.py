"""ML-based trading strategy engine.

Trains a Gradient Boosting classifier on technical indicators + sentiment to
predict whether a stock's price will rise over the next N days.  The model is
periodically retrained on fresh data so it improves over time.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from stockio import config
from stockio.config import get_logger
from stockio.market_data import (
    add_technical_indicators,
    build_feature_matrix,
    fetch_history,
)
from stockio.sentiment import SentimentScore

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    ticker: str
    signal: Signal
    confidence: float  # 0–1
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

_MODEL_PATH = config.MODEL_DIR / "gb_model.joblib"
_SCALER_PATH = config.MODEL_DIR / "scaler.joblib"
_META_PATH = config.MODEL_DIR / "model_meta.json"


def _save_model(model, scaler, feature_names: list[str], accuracy: float) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _MODEL_PATH)
    joblib.dump(scaler, _SCALER_PATH)
    meta = {
        "trained_at": dt.datetime.utcnow().isoformat(),
        "features": feature_names,
        "accuracy": accuracy,
    }
    _META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("Model saved (accuracy=%.4f)", accuracy)


def _load_model():
    if not _MODEL_PATH.exists():
        return None, None, None
    model = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)
    meta = json.loads(_META_PATH.read_text())
    return model, scaler, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    tickers: list[str],
    period: str = "2y",
    forecast_horizon: int = 5,
) -> tuple[GradientBoostingClassifier, StandardScaler, list[str], float]:
    """Train (or retrain) the prediction model on recent data for *tickers*.

    Returns (model, scaler, feature_names, cv_accuracy).
    """
    log.info("Training model on %d tickers (period=%s) ...", len(tickers), period)

    all_X, all_y = [], []
    feature_names: list[str] = []

    for ticker in tickers:
        df = fetch_history(ticker, period=period)
        if df.empty or len(df) < 60:
            log.warning("Skipping %s — insufficient data (%d rows)", ticker, len(df))
            continue
        df = add_technical_indicators(df)
        X, y, feat = build_feature_matrix(df, forecast_horizon=forecast_horizon)
        all_X.append(X)
        all_y.append(y)
        if not feature_names:
            feature_names = feat

    if not all_X:
        raise RuntimeError("No usable training data for any ticker")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series aware cross-validation
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy")
    accuracy = float(cv_scores.mean())
    log.info("Cross-val accuracy: %.4f (+/- %.4f)", accuracy, cv_scores.std())

    # Final fit on all data
    model.fit(X_scaled, y)

    _save_model(model, scaler, feature_names, accuracy)
    return model, scaler, feature_names, accuracy


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    ticker: str,
    sentiment: SentimentScore | None = None,
) -> TradeSignal:
    """Generate a BUY / SELL / HOLD signal for *ticker*.

    Combines the ML model's prediction with sentiment analysis and
    basic risk rules to produce a final recommendation.
    """
    model, scaler, meta = _load_model()
    if model is None:
        log.warning("No trained model found — returning HOLD")
        return TradeSignal(ticker=ticker, signal=Signal.HOLD, confidence=0.0,
                           reasons=["No trained model available"])

    # Build features from latest data
    df = fetch_history(ticker, period="6mo")
    if df.empty or len(df) < 40:
        return TradeSignal(ticker=ticker, signal=Signal.HOLD, confidence=0.0,
                           reasons=["Insufficient market data"])

    df = add_technical_indicators(df)
    feature_cols = [c for c in df.columns
                    if c not in ("Open", "High", "Low", "Close", "Volume")]
    latest = df[feature_cols].iloc[[-1]].values
    latest_scaled = scaler.transform(latest)

    prob = model.predict_proba(latest_scaled)[0]  # [P(down), P(up)]
    prob_up = float(prob[1]) if len(prob) > 1 else 0.5

    reasons: list[str] = []

    # --- ML signal ---
    ml_score = (prob_up - 0.5) * 2  # map 0.5–1 → 0–1, 0–0.5 → -1–0

    # --- Sentiment adjustment ---
    sentiment_adj = 0.0
    if sentiment and sentiment.num_articles > 0:
        sentiment_adj = sentiment.score * 0.3  # 30 % weight
        reasons.append(f"Sentiment: {sentiment.score:+.2f} ({sentiment.num_articles} articles)")

    # --- Technical confirmation ---
    last_row = df.iloc[-1]
    rsi = last_row.get("rsi", 50)
    macd_diff = last_row.get("macd_diff", 0)
    if rsi < 30:
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        reasons.append(f"RSI overbought ({rsi:.1f})")
    if macd_diff > 0:
        reasons.append("MACD bullish crossover")
    elif macd_diff < 0:
        reasons.append("MACD bearish crossover")

    # --- Composite score ---
    composite = ml_score * 0.5 + sentiment_adj + (0.2 if macd_diff > 0 else -0.2 if macd_diff < 0 else 0.0)
    composite = max(-1.0, min(1.0, composite))

    # Decision thresholds
    BUY_THRESHOLD = 0.25
    SELL_THRESHOLD = -0.25

    if composite >= BUY_THRESHOLD:
        signal = Signal.BUY
    elif composite <= SELL_THRESHOLD:
        signal = Signal.SELL
    else:
        signal = Signal.HOLD

    confidence = abs(composite)
    reasons.insert(0, f"ML P(up)={prob_up:.2f}, composite={composite:+.3f}")

    log.info(
        "Signal for %s: %s (confidence=%.2f, composite=%.3f)",
        ticker, signal.value, confidence, composite,
    )
    return TradeSignal(ticker=ticker, signal=signal, confidence=confidence, reasons=reasons)


def generate_signals(
    tickers: list[str],
    sentiments: dict[str, SentimentScore] | None = None,
) -> list[TradeSignal]:
    """Generate trade signals for all *tickers*."""
    signals: list[TradeSignal] = []
    for ticker in tickers:
        sent = sentiments.get(ticker) if sentiments else None
        sig = predict(ticker, sentiment=sent)
        signals.append(sig)
    return signals
