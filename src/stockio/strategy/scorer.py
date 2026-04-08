"""Instrument scoring — rules-based fallback and LightGBM ML mode."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog

from stockio.broker.models import Direction, Signal
from stockio.config import Settings

log = structlog.get_logger()

# Feature names expected by the LightGBM model (order matters for prediction)
FEATURE_NAMES = [
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
]


class InstrumentScorer:
    """Scores instruments to decide trade direction and confidence."""

    def __init__(self, settings: Settings, models_dir: Path) -> None:
        self._settings = settings
        self._models_dir = models_dir
        self._model = None
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

        if self._model is not None:
            return self._score_ml(instrument, features_with_sentiment)
        return self._score_rules(instrument, features_with_sentiment)

    def _score_ml(self, instrument: str, features: dict[str, float]) -> Signal:
        """Score using the trained LightGBM model."""
        import lightgbm as lgb

        feature_values = [features.get(name, 0.0) for name in FEATURE_NAMES]
        prediction = self._model.predict([feature_values])[0]

        # Model outputs probability of upward move
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
            timestamp=datetime.now(timezone.utc),
            features=features,
        )

    def _score_rules(self, instrument: str, features: dict[str, float]) -> Signal:
        """Rules-based scoring fallback when no ML model is available."""
        score = 0.0
        max_score = 0.0

        # RSI — oversold/overbought
        rsi_14 = features.get("rsi_14", 50)
        if rsi_14 < 30:
            score += 1.5  # Strong buy signal
        elif rsi_14 < 40:
            score += 0.5
        elif rsi_14 > 70:
            score -= 1.5  # Strong sell signal
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
            # Weak trend — reduce confidence
            score *= 0.5
        max_score += 0.0  # ADX is a multiplier, not additive

        # Bollinger %B — extremes
        bb_pct = features.get("bb_percent_b", 0.5)
        if bb_pct < 0.1:
            score += 0.5  # Near lower band — buy
        elif bb_pct > 0.9:
            score -= 0.5  # Near upper band — sell
        max_score += 0.5

        # Sentiment bias
        sentiment = features.get("sentiment", 0.0)
        score += sentiment * 1.0
        max_score += 1.0

        # Normalize to direction + confidence
        if max_score == 0:
            return Signal(
                instrument=instrument,
                direction=Direction.HOLD,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                features=features,
            )

        normalized = score / max_score  # Range roughly -1 to +1
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
            timestamp=datetime.now(timezone.utc),
            features=features,
        )

    def rank_instruments(self, signals: dict[str, Signal]) -> list[Signal]:
        """Rank signals by confidence, filtering out HOLD and low-confidence."""
        tradeable = [
            s
            for s in signals.values()
            if s.direction != Direction.HOLD
            and s.confidence >= self._settings.min_confidence
        ]
        tradeable.sort(key=lambda s: s.confidence, reverse=True)
        return tradeable

    def reload_model(self) -> None:
        """Re-load the ML model from disk (after retrain)."""
        self._load_model()
