"""Competing trading strategies — each scores instruments differently.

Four strategies compete on separate OANDA accounts:
- trend:    Only trades WITH the trend (EMA alignment)
- meanrev:  Buys oversold in uptrends, sells overbought in downtrends
- momentum: Only trades strong trends (ADX > 25 + MACD acceleration)
- llm:      Claude makes all trading decisions directly
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Callable

import structlog

from stockio.broker.models import Direction, Signal
from stockio.config import Settings

log = structlog.get_logger()

StrategyFn = Callable[[str, dict[str, float], float], Signal]


def _make_signal(instrument: str, direction: Direction, confidence: float,
                 features: dict) -> Signal:
    return Signal(
        instrument=instrument,
        direction=direction,
        confidence=min(abs(confidence), 1.0),
        timestamp=datetime.now(UTC),
        features=features,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Trend Following
# ---------------------------------------------------------------------------

def score_trend(instrument: str, features: dict[str, float],
                sentiment: float) -> Signal:
    """Only trade WITH the trend. No counter-trend trades.

    BUY only when price is above all EMAs.
    SELL only when price is below all EMAs.
    HOLD when EMAs are mixed (no clear trend).
    """
    ema_cross_short_mid = features.get("ema_cross_short_mid", 0)
    ema_cross_mid_long = features.get("ema_cross_mid_long", 0)
    close_vs_ema = features.get("close_vs_ema_long", 0)
    adx = features.get("adx", 0)
    macd = features.get("macd_histogram", 0)
    rsi = features.get("rsi_14", 50)

    # All EMAs must agree
    all_bullish = (ema_cross_short_mid > 0 and ema_cross_mid_long > 0
                   and close_vs_ema > 0.0005)
    all_bearish = (ema_cross_short_mid < 0 and ema_cross_mid_long < 0
                   and close_vs_ema < -0.0005)

    if not all_bullish and not all_bearish:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    direction = Direction.BUY if all_bullish else Direction.SELL

    # RSI guard: don't buy into overbought or sell into oversold
    if direction == Direction.BUY and rsi > 70:
        return _make_signal(instrument, Direction.HOLD, 0, features)
    if direction == Direction.SELL and rsi < 30:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    # Confidence built from multiple independent signals
    conf = 0.0

    # ADX contribution — scaled continuously (0-0.30)
    conf += min(adx / 80, 0.30)

    # MACD confirmation — scaled by magnitude (0-0.25)
    macd_abs = abs(macd)
    if (direction == Direction.BUY and macd > 0) or \
       (direction == Direction.SELL and macd < 0):
        if macd_abs > 0.002:
            conf += 0.25
        elif macd_abs > 0.001:
            conf += 0.18
        elif macd_abs > 0.0003:
            conf += 0.10
        else:
            conf += 0.05

    # EMA separation strength — how far price is from long EMA (0-0.20)
    ema_dist = abs(close_vs_ema)
    if ema_dist > 0.005:
        conf += 0.20
    elif ema_dist > 0.002:
        conf += 0.12
    elif ema_dist > 0.001:
        conf += 0.06

    # RSI momentum alignment (0-0.15)
    if direction == Direction.BUY and rsi > 55:
        conf += min((rsi - 55) / 100, 0.15)
    elif direction == Direction.SELL and rsi < 45:
        conf += min((45 - rsi) / 100, 0.15)

    # Sentiment alignment (0-0.10)
    if (direction == Direction.BUY and sentiment > 0.1) or \
       (direction == Direction.SELL and sentiment < -0.1):
        conf += 0.10

    return _make_signal(instrument, direction, conf, features)


# ---------------------------------------------------------------------------
# Strategy 2: Mean Reversion
# ---------------------------------------------------------------------------

def score_meanrev(instrument: str, features: dict[str, float],
                  sentiment: float) -> Signal:
    """Buy oversold, sell overbought — with trend and indicator confirmation.

    Uses RSI extremes with Bollinger Band and Stochastic confirmation.
    Blocks trades only when the trend is strongly against the trade
    (don't buy oversold in a strong downtrend — that's a falling knife).
    """
    rsi = features.get("rsi_14", 50)
    bb = features.get("bb_percent_b", 0.5)
    stoch_k = features.get("stoch_k", 50)
    close_vs_ema = features.get("close_vs_ema_long", 0)
    adx = features.get("adx", 0)

    # RSI must be at a genuine extreme — tightened from 35/65
    if 30 <= rsi <= 70:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    if rsi < 30:
        # Oversold — block if strongly trending down (falling knife)
        if close_vs_ema < -0.002 and adx > 30:
            return _make_signal(instrument, Direction.HOLD, 0, features)

        direction = Direction.BUY
        # Continuous RSI scaling — more extreme = more confidence
        # RSI 30 → 0.10, RSI 20 → 0.30, RSI 10 → 0.50
        conf = min((30 - rsi) / 50, 0.50)

        # BB confirmation — continuous scaling (0-0.20)
        if bb < 0.05:
            conf += 0.20
        elif bb < 0.15:
            conf += 0.15
        elif bb < 0.30:
            conf += 0.08

        # Stochastic confirmation — continuous scaling (0-0.15)
        if stoch_k < 15:
            conf += 0.15
        elif stoch_k < 25:
            conf += 0.10
        elif stoch_k < 35:
            conf += 0.05

        # Trend alignment (0-0.10)
        if close_vs_ema > 0:
            conf += 0.10
        elif close_vs_ema > -0.001:
            conf += 0.05

    else:  # rsi > 70
        # Overbought — block if strongly trending up
        if close_vs_ema > 0.002 and adx > 30:
            return _make_signal(instrument, Direction.HOLD, 0, features)

        direction = Direction.SELL
        conf = min((rsi - 70) / 50, 0.50)

        if bb > 0.95:
            conf += 0.20
        elif bb > 0.85:
            conf += 0.15
        elif bb > 0.70:
            conf += 0.08

        if stoch_k > 85:
            conf += 0.15
        elif stoch_k > 75:
            conf += 0.10
        elif stoch_k > 65:
            conf += 0.05

        if close_vs_ema < 0:
            conf += 0.10
        elif close_vs_ema < 0.001:
            conf += 0.05

    return _make_signal(instrument, direction, conf, features)


# ---------------------------------------------------------------------------
# Strategy 3: Momentum
# ---------------------------------------------------------------------------

def score_momentum(instrument: str, features: dict[str, float],
                   sentiment: float) -> Signal:
    """Only trade when there's strong momentum. Sit out weak markets.

    Gate: ADX must be > 30 (strong trend required).
    Uses MACD direction + EMA cross for signal. RSI must confirm direction.
    """
    adx = features.get("adx", 0)
    macd = features.get("macd_histogram", 0)
    ema_cross = features.get("ema_cross_short_mid", 0)
    close_vs_ema = features.get("close_vs_ema_long", 0)
    rsi = features.get("rsi_14", 50)

    # Hard gate: raised from 25 to 30 — only strong trends
    if adx < 30:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    # Direction from MACD + EMA cross agreement
    macd_bullish = macd > 0
    ema_bullish = ema_cross > 0

    if macd_bullish and ema_bullish:
        direction = Direction.BUY
    elif not macd_bullish and not ema_bullish:
        direction = Direction.SELL
    else:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    # RSI must confirm — don't buy if RSI is weak, don't sell if RSI is strong
    if direction == Direction.BUY and rsi < 45:
        return _make_signal(instrument, Direction.HOLD, 0, features)
    if direction == Direction.SELL and rsi > 55:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    # Confidence from ADX strength — continuous (0-0.30)
    conf = min((adx - 30) / 50, 0.30)

    # MACD magnitude — continuous scaling (0-0.25)
    macd_abs = abs(macd)
    if macd_abs > 0.002:
        conf += 0.25
    elif macd_abs > 0.001:
        conf += 0.18
    elif macd_abs > 0.0005:
        conf += 0.10
    else:
        conf += 0.04

    # EMA separation — how far into the trend (0-0.20)
    ema_dist = abs(close_vs_ema)
    if ema_dist > 0.003:
        conf += 0.20
    elif ema_dist > 0.001:
        conf += 0.12
    elif ema_dist > 0.0005:
        conf += 0.06

    # RSI strength bonus (0-0.15)
    if direction == Direction.BUY:
        conf += min(max(rsi - 50, 0) / 133, 0.15)
    else:
        conf += min(max(50 - rsi, 0) / 133, 0.15)

    return _make_signal(instrument, direction, conf, features)


# ---------------------------------------------------------------------------
# Strategy 4: Consensus (2+ strategies must agree)
# ---------------------------------------------------------------------------

def score_consensus(instrument: str, features: dict[str, float],
                    sentiment: float) -> Signal:
    """Only trade when 2+ strategies independently agree on direction.

    Runs trend, meanrev, and momentum scorers on the same data.
    BUY/SELL only if 2+ agree with confidence above a minimum threshold.
    Returns the average confidence of agreeing strategies.
    """
    signals = [
        score_trend(instrument, features, sentiment),
        score_meanrev(instrument, features, sentiment),
        score_momentum(instrument, features, sentiment),
    ]

    buys = [s for s in signals if s.direction == Direction.BUY and s.confidence > 0.20]
    sells = [s for s in signals if s.direction == Direction.SELL and s.confidence > 0.20]

    if len(buys) >= 2:
        avg_conf = sum(s.confidence for s in buys) / len(buys)
        return _make_signal(instrument, Direction.BUY, avg_conf, features)
    elif len(sells) >= 2:
        avg_conf = sum(s.confidence for s in sells) / len(sells)
        return _make_signal(instrument, Direction.SELL, avg_conf, features)
    else:
        return _make_signal(instrument, Direction.HOLD, 0, features)


# ---------------------------------------------------------------------------
# Strategy 5: Contrarian (bet against LLM)
# ---------------------------------------------------------------------------

def score_best_signal(instrument: str, features: dict[str, float],
                      sentiment: float) -> Signal:
    """Pick the single highest-confidence signal from all rule-based strategies.

    Runs trend, meanrev, and momentum independently. Returns whichever
    has the strongest conviction. More aggressive than consensus (which
    requires 2+ to agree) but still benefits from comparing multiple views.
    """
    signals = [
        score_trend(instrument, features, sentiment),
        score_meanrev(instrument, features, sentiment),
        score_momentum(instrument, features, sentiment),
    ]

    # Filter to non-HOLD signals with meaningful confidence
    active = [s for s in signals if s.direction != Direction.HOLD and s.confidence > 0.2]

    if not active:
        return _make_signal(instrument, Direction.HOLD, 0, features)

    # Return the highest confidence signal
    best = max(active, key=lambda s: s.confidence)
    return _make_signal(instrument, best.direction, best.confidence, features)


# ---------------------------------------------------------------------------
# Strategy 6: LLM-Only (Claude decides everything)
# ---------------------------------------------------------------------------

_LLM_SCORING_PROMPT = """\
You are a disciplined forex trader managing a £1,000 account. Your goal is \
capital preservation first, profits second. You have been losing money by \
trading too often with weak signals. From now on, be extremely selective.

## Indicator Guide
- RSI < 30 = oversold (potential BUY), RSI > 70 = overbought (potential SELL)
- RSI 40-60 = neutral — no trade unless other signals are very strong
- ADX > 25 = strong trend (trade WITH it), ADX < 20 = weak/ranging (avoid)
- MACD > 0 = bullish momentum, MACD < 0 = bearish momentum
- BB%B < 0.2 = near lower band (oversold), BB%B > 0.8 = near upper band (overbought)
- EMA_cross > 0 = short-term uptrend, < 0 = short-term downtrend
- trend_vs_ema > 0 = price above long EMA (uptrend), < 0 = below (downtrend)
- Sentiment: positive = bullish news, negative = bearish news

## Current Market Data
{instrument_data}

## Rules (MUST follow)
1. HOLD everything unless you see 3+ indicators aligned in the same direction
2. Never trade against a strong trend (ADX > 25 + clear EMA direction)
3. Maximum 1-2 BUY/SELL recommendations per cycle — HOLD everything else
4. Confidence must reflect conviction: 0.7+ = strong setup, 0.5-0.7 = moderate
5. If in doubt, HOLD. Missing a trade costs nothing; a bad trade costs money
6. Avoid instruments with RSI 40-60 AND ADX < 20 (no signal, no trend)

Return a JSON object mapping instrument names to decisions:
```json
{{
  "EUR_USD": {{"direction": "HOLD", "confidence": 0.0, "reason": "no clear setup"}},
  "GBP_USD": {{"direction": "SELL", "confidence": 0.75, "reason": "RSI 72 + bearish MACD + ADX 30"}},
  ...
}}
```

Return ONLY the JSON, no other text."""


class LLMScorer:
    """Claude makes all trading decisions. No rules, no ML model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._last_decisions: dict[str, dict] = {}

        if settings.anthropic_api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def score_all(
        self,
        features_by_instrument: dict[str, dict[str, float]],
        sentiment_by_instrument: dict[str, float],
    ) -> dict[str, Signal]:
        """Score all instruments in one LLM call. Returns {instrument: Signal}."""
        if not self._client:
            return {}

        # Build instrument data for the prompt
        lines = []
        for inst, feat in features_by_instrument.items():
            sent = sentiment_by_instrument.get(inst, 0)
            rsi = feat.get("rsi_14", 50)
            adx = feat.get("adx", 0)
            macd = feat.get("macd_histogram", 0)
            bb = feat.get("bb_percent_b", 0.5)
            trend = feat.get("close_vs_ema_long", 0)
            ema_cross = feat.get("ema_cross_short_mid", 0)
            stoch = feat.get("stoch_k", 50)

            lines.append(
                f"{inst}: RSI={rsi:.0f} ADX={adx:.0f} MACD={macd:.5f} "
                f"BB%B={bb:.2f} EMA_cross={ema_cross:.5f} "
                f"trend_vs_ema={trend:.4f} Stoch={stoch:.0f} "
                f"sentiment={sent:.2f}"
            )

        prompt = _LLM_SCORING_PROMPT.format(
            instrument_data="\n".join(lines)
        )

        try:
            response = self._client.messages.create(
                model=self._settings.llm_model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            decisions = json.loads(text)
            self._last_decisions = decisions
        except Exception:
            log.exception("llm_scorer_failed")
            return {}

        # Convert to Signal objects
        signals = {}
        for inst, decision in decisions.items():
            dir_str = decision.get("direction", "HOLD").upper()
            if dir_str == "BUY":
                direction = Direction.BUY
            elif dir_str == "SELL":
                direction = Direction.SELL
            else:
                direction = Direction.HOLD

            conf = float(decision.get("confidence", 0))
            features = features_by_instrument.get(inst, {})
            signals[inst] = _make_signal(inst, direction, conf, features)

            if direction != Direction.HOLD:
                log.info(
                    "llm_signal",
                    instrument=inst,
                    direction=dir_str,
                    confidence=round(conf, 3),
                    reason=decision.get("reason", "")[:80],
                )

        return signals

    @property
    def last_decisions(self) -> dict:
        return self._last_decisions


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_SLOTS = ("trend", "meanrev", "momentum", "llm")

STRATEGIES: dict[str, StrategyFn] = {
    "trend": score_trend,
    "meanrev": score_meanrev,
    "momentum": score_momentum,
    "consensus": score_consensus,
    "best_signal": score_best_signal,
    # "llm" is handled separately via LLMScorer (batch scoring)
}
