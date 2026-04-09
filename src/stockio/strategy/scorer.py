"""Instrument scoring — rules-based fallback and LightGBM ML mode."""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import structlog

from stockio.broker.models import Direction, Signal
from stockio.config import Settings

log = structlog.get_logger()

# Feature names expected by the LightGBM model (order matters for prediction)
FEATURE_NAMES = [
    # Technical (13)
    "ema_cross_short_mid",
    "ema_cross_mid_long",
    "macd_histogram",
    "rsi_7",
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "atr",
    "bb_percent_b",
    "adx",
    "close_vs_ema_long",
    "range_vs_atr",
    "sentiment",
    # Temporal (7)
    "session_asia",
    "session_london",
    "session_newyork",
    "session_overlap",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    # Calendar (4)
    "hours_until_high_event",
    "hours_until_medium_event",
    "is_event_window",
    "events_next_4h",
]

MIN_TRAINING_SAMPLES = 200


# ---------------------------------------------------------------------------
# Outcome tracking
# ---------------------------------------------------------------------------


@dataclass
class PendingOutcome:
    """A prediction waiting to be resolved after label_horizon_bars."""

    instrument: str
    features: dict[str, float]
    direction: Direction
    confidence: float
    entry_price: float
    atr: float
    timestamp: datetime
    horizon_cycle: int  # Cycle number when this outcome should be resolved


class OutcomeTracker:
    """Tracks pending predictions and records resolved outcomes to parquet."""

    def __init__(self, settings: Settings, data_dir: Path) -> None:
        self._settings = settings
        self._data_dir = data_dir
        self._pending: collections.deque[PendingOutcome] = collections.deque(maxlen=500)
        self._parquet_path = data_dir / "training_data.parquet"

        # Buffer resolved outcomes before flushing to disk
        self._write_buffer: list[dict] = []
        self._flush_threshold = 20

        # In-memory count to avoid reading parquet on every access
        self._row_count = self._count_existing_rows()

        # Rolling accuracy tracking for degradation detection
        self._recent_outcomes: collections.deque[bool] = collections.deque(
            maxlen=settings.degradation_window
        )

    def record_pending(
        self,
        instrument: str,
        features: dict[str, float],
        direction: Direction,
        confidence: float,
        entry_price: float,
        atr: float,
        current_cycle: int,
    ) -> None:
        """Record a new prediction to be resolved later."""
        self._pending.append(
            PendingOutcome(
                instrument=instrument,
                features=features,
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                atr=atr,
                timestamp=datetime.now(UTC),
                horizon_cycle=current_cycle + self._settings.label_horizon_bars,
            )
        )

    def resolve_outcomes(self, current_cycle: int, get_price_fn) -> int:
        """Check pending outcomes that have reached their horizon.

        get_price_fn(instrument) -> float should return current mid price.
        Returns number of outcomes resolved.
        """
        resolved = 0
        remaining: list[PendingOutcome] = []

        for outcome in self._pending:
            if current_cycle < outcome.horizon_cycle:
                remaining.append(outcome)
                continue

            try:
                current_price = get_price_fn(outcome.instrument)
            except Exception:
                log.exception("outcome_price_fetch_failed", instrument=outcome.instrument)
                remaining.append(outcome)  # Try again next cycle
                continue

            move = current_price - outcome.entry_price
            threshold = outcome.atr * self._settings.label_atr_mult

            if outcome.direction == Direction.BUY:
                correct = move >= threshold
            else:
                correct = -move >= threshold

            self._recent_outcomes.append(correct)
            self._append_training_row(outcome, current_price, correct)
            resolved += 1

        self._pending.clear()
        self._pending.extend(remaining)
        # Flush any remaining buffered rows after resolving
        if resolved > 0:
            self._flush_buffer()
        return resolved

    def _append_training_row(
        self, outcome: PendingOutcome, exit_price: float, correct: bool
    ) -> None:
        """Buffer a resolved outcome. Flushes to parquet when buffer is full."""
        row = {
            "timestamp": outcome.timestamp.isoformat(),
            "instrument": outcome.instrument,
            "direction": outcome.direction.value,
            "confidence": outcome.confidence,
            "entry_price": outcome.entry_price,
            "exit_price": exit_price,
            "atr": outcome.atr,
            "label": int(correct),
        }
        for name in FEATURE_NAMES:
            row[name] = outcome.features.get(name, 0.0)

        self._write_buffer.append(row)
        self._row_count += 1

        if len(self._write_buffer) >= self._flush_threshold:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered rows to parquet file."""
        if not self._write_buffer:
            return
        try:
            new_rows = pd.DataFrame(self._write_buffer)
            if self._parquet_path.exists():
                existing = pd.read_parquet(self._parquet_path)
                combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows
            combined.to_parquet(self._parquet_path, index=False)
            self._write_buffer.clear()
        except Exception:
            log.exception("parquet_flush_failed")

    def _count_existing_rows(self) -> int:
        """Count rows in existing parquet file (called once at startup)."""
        if not self._parquet_path.exists():
            return 0
        try:
            return len(pd.read_parquet(self._parquet_path))
        except Exception:
            return 0

    @property
    def training_data_count(self) -> int:
        return self._row_count

    @property
    def rolling_accuracy(self) -> float | None:
        if not self._recent_outcomes:
            return None
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    @property
    def is_degraded(self) -> bool:
        acc = self.rolling_accuracy
        if acc is None:
            return False
        return acc < self._settings.degradation_threshold

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def retrain_model(settings: Settings, data_dir: Path, models_dir: Path) -> bool:
    """Retrain the LightGBM model from accumulated training data.

    Returns True if training succeeded, False otherwise.
    """
    parquet_path = data_dir / "training_data.parquet"
    if not parquet_path.exists():
        log.info("retrain_skipped", reason="no training data")
        return False

    df = pd.read_parquet(parquet_path)
    if len(df) < MIN_TRAINING_SAMPLES:
        log.info("retrain_skipped", reason="insufficient data", samples=len(df))
        return False

    try:
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        features = df[FEATURE_NAMES].fillna(0).values
        labels = df["label"].values

        tscv = TimeSeriesSplit(n_splits=settings.n_splits, gap=settings.gap_bars)
        accuracies = []

        model = None
        for train_idx, test_idx in tscv.split(features):
            x_train, x_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            train_data = lgb.Dataset(
                x_train, label=y_train, feature_name=FEATURE_NAMES
            )
            valid_data = lgb.Dataset(
                x_test, label=y_test, feature_name=FEATURE_NAMES
            )

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "num_threads": 2,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )

            preds = (model.predict(x_test) > 0.5).astype(int)
            acc = (preds == y_test).mean()
            accuracies.append(acc)

        # Train final model on all data
        full_data = lgb.Dataset(
            features, label=labels, feature_name=FEATURE_NAMES
        )
        final_model = lgb.train(params, full_data, num_boost_round=200)

        # Save model
        model_path = models_dir / "lightgbm_model.txt"
        final_model.save_model(str(model_path))

        # Save metadata
        meta = {
            "train_date": datetime.now(UTC).isoformat(),
            "samples": len(df),
            "cv_accuracies": accuracies,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "features": FEATURE_NAMES,
        }
        meta_path = models_dir / "model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info(
            "retrain_complete",
            samples=len(df),
            mean_accuracy=round(meta["mean_accuracy"], 3),
            cv_accuracies=[round(a, 3) for a in accuracies],
        )
        return True

    except Exception:
        log.exception("retrain_failed")
        return False


# ---------------------------------------------------------------------------
# Instrument scorer
# ---------------------------------------------------------------------------


class InstrumentScorer:
    """Scores instruments to decide trade direction and confidence."""

    def __init__(self, settings: Settings, models_dir: Path) -> None:
        self._settings = settings
        self._models_dir = models_dir
        self._model = None
        self._use_rules_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Try to load a trained LightGBM model. Fall back to rules-based if missing."""
        model_path = self._models_dir / "lightgbm_model.txt"
        if not model_path.exists():
            log.info("no_ml_model_found", mode="rules_based")
            self._model = None
            return

        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(model_path))
            self._use_rules_fallback = False
            log.info("ml_model_loaded", path=str(model_path))
        except Exception:
            log.exception("ml_model_load_failed", mode="rules_based")
            self._model = None

    def score_instrument(
        self,
        instrument: str,
        features: dict[str, float],
        sentiment: float = 0.0,
    ) -> Signal:
        """Score a single instrument. Returns a Signal with direction + confidence."""
        features_with_sentiment = {**features, "sentiment": sentiment}

        if self._model is not None and not self._use_rules_fallback:
            return self._score_ml(instrument, features_with_sentiment)
        return self._score_rules(instrument, features_with_sentiment)

    def _score_ml(self, instrument: str, features: dict[str, float]) -> Signal:
        """Score using the trained LightGBM model."""
        feature_values = [features.get(name, 0.0) for name in FEATURE_NAMES]
        prediction = self._model.predict([feature_values])[0]

        if prediction > 0.5:
            direction = Direction.BUY
            confidence = float(prediction)
        elif prediction < 0.5:
            direction = Direction.SELL
            confidence = float(1.0 - prediction)
        else:
            direction = Direction.HOLD
            confidence = 0.0

        return Signal(
            instrument=instrument,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            features=features,
        )

    def _score_rules(self, instrument: str, features: dict[str, float]) -> Signal:
        """Rules-based scoring fallback when no ML model is available."""
        score = 0.0
        max_score = 0.0

        # RSI — oversold/overbought
        rsi_14 = features.get("rsi_14", 50)
        if rsi_14 < 30:
            score += 1.5
        elif rsi_14 < 40:
            score += 0.5
        elif rsi_14 > 70:
            score -= 1.5
        elif rsi_14 > 60:
            score -= 0.5
        max_score += 1.5

        # MACD histogram direction
        macd_hist = features.get("macd_histogram", 0)
        if macd_hist > 0:
            score += 1.0
        elif macd_hist < 0:
            score -= 1.0
        max_score += 1.0

        # EMA cross — short above mid is bullish
        ema_cross = features.get("ema_cross_short_mid", 0)
        if ema_cross > 0:
            score += 1.0
        elif ema_cross < 0:
            score -= 1.0
        max_score += 1.0

        # Price vs long EMA — trend following
        close_vs_ema = features.get("close_vs_ema_long", 0)
        if close_vs_ema > 0.001:
            score += 0.5
        elif close_vs_ema < -0.001:
            score -= 0.5
        max_score += 0.5

        # ADX — only trade when trending
        adx = features.get("adx", 0)
        if adx < 20:
            score *= 0.5

        # Bollinger %B — extremes
        bb_pct = features.get("bb_percent_b", 0.5)
        if bb_pct < 0.1:
            score += 0.5
        elif bb_pct > 0.9:
            score -= 0.5
        max_score += 0.5

        # Sentiment bias
        sentiment = features.get("sentiment", 0.0)
        score += sentiment * 1.0
        max_score += 1.0

        # Session context — signals are more reliable during active sessions
        if features.get("session_overlap", 0) == 1.0:
            score *= 1.2  # London/NY overlap = strongest signals
        elif features.get("session_asia", 0) == 1.0 and adx < 25:
            score *= 0.6  # Asian ranging = weak signals

        # Friday late: reduce confidence (position squaring)
        if features.get("day_of_week", 0) > 0.9 and features.get("hour_cos", 0) < -0.5:
            score *= 0.7

        # Economic calendar — don't trade during high-impact event windows
        if features.get("is_event_window", 0) == 1.0:
            return Signal(
                instrument=instrument,
                direction=Direction.HOLD,
                confidence=0.0,
                timestamp=datetime.now(UTC),
                features=features,
            )

        # Reduce confidence when event is imminent
        hours_until = features.get("hours_until_high_event", 1.0)
        if hours_until < 1.0 / 168:  # Less than 1 hour (normalized)
            score *= 0.5

        if max_score == 0:
            return Signal(
                instrument=instrument,
                direction=Direction.HOLD,
                confidence=0.0,
                timestamp=datetime.now(UTC),
                features=features,
            )

        normalized = score / max_score
        confidence = min(abs(normalized), 1.0)

        if normalized > 0.1:
            direction = Direction.BUY
        elif normalized < -0.1:
            direction = Direction.SELL
        else:
            direction = Direction.HOLD
            confidence = 0.0

        return Signal(
            instrument=instrument,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            features=features,
        )

    def rank_instruments(self, signals: dict[str, Signal]) -> list[Signal]:
        """Rank signals by confidence, filtering out HOLD and low-confidence."""
        # Check DB override for min_confidence
        try:
            from stockio import db

            saved = db.get_setting("min_confidence")
            threshold = float(saved) if saved else self._settings.min_confidence
        except Exception:
            threshold = self._settings.min_confidence

        tradeable = [
            s
            for s in signals.values()
            if s.direction != Direction.HOLD and s.confidence >= threshold
        ]
        tradeable.sort(key=lambda s: s.confidence, reverse=True)
        return tradeable

    def reload_model(self) -> None:
        """Re-load the ML model from disk (after retrain)."""
        self._load_model()

    def set_rules_fallback(self, enabled: bool) -> None:
        """Force rules-based mode (used when model degradation detected)."""
        self._use_rules_fallback = enabled
        if enabled:
            log.warning("scorer_degradation_fallback", mode="rules_based")
