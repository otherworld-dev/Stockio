"""LLM-powered trade advisor — uses a single Claude call per cycle.

Combines regime detection, trade advice, parameter recommendations,
and loss analysis into one comprehensive prompt per cycle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog

from stockio.config import Settings

log = structlog.get_logger()

_CYCLE_PROMPT = """\
You are a forex trading advisor. Analyze the current market and advise on trades.

## Current Market Data
{instrument_data}

## Recent Trade History (last 10 closed)
{recent_trades}

## Recent Losses to Learn From
{recent_losses}

## Pending Signals Above Threshold
{pending_signals}

## Task
Analyze everything above and return a JSON object with:

1. **regime** — current market assessment
2. **trade_decisions** — for each pending signal, whether to take it and with what parameters
3. **lessons** — what we should learn from recent losses

```json
{{
  "regime": {{
    "type": "trending" or "ranging" or "volatile" or "quiet",
    "direction": "risk_on" or "risk_off" or "mixed",
    "summary": "one sentence",
    "avoid_trading": false
  }},
  "trade_decisions": {{
    "INSTRUMENT_NAME": {{
      "take_trade": true or false,
      "reason": "brief explanation",
      "sl_mult": 1.5,
      "tp_mult": 2.0
    }}
  }},
  "lessons": "one paragraph of key takeaways from recent losses, or empty string if no losses"
}}
```

Important:
- Only recommend trades where indicators genuinely support the direction
- If an instrument has lost 3+ times recently, be very cautious
- Adjust sl_mult (1.0-3.0) and tp_mult (1.0-4.0) based on current volatility
- In ranging markets, use tighter TP. In trending, let winners run
- If market is too uncertain, set avoid_trading to true

Return ONLY the JSON, no other text."""


class LLMAdvisor:
    """Claude-powered trade reasoning — one call per cycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._enabled = bool(settings.anthropic_api_key)
        self._last_advice: dict | None = None
        self._last_advice_time: datetime | None = None

        if self._enabled:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_advice(self) -> dict | None:
        return self._last_advice

    @property
    def regime(self) -> dict | None:
        if self._last_advice:
            return self._last_advice.get("regime")
        return None

    def get_trade_decision(self, instrument: str) -> dict | None:
        """Get the LLM's decision for a specific instrument from the last cycle."""
        if not self._last_advice:
            return None
        decisions = self._last_advice.get("trade_decisions", {})
        return decisions.get(instrument)

    def advise_cycle(
        self,
        features_by_instrument: dict[str, dict],
        sentiment_by_instrument: dict[str, float],
        pending_signals: list[dict],
        recent_trades: list[dict],
    ) -> dict | None:
        """Run the full advisory for this cycle. One LLM call.

        Args:
            features_by_instrument: {instrument: {rsi_14, adx, macd_histogram, ...}}
            sentiment_by_instrument: {instrument: float}
            pending_signals: [{instrument, direction, confidence}, ...] above threshold
            recent_trades: last 10 closed trades from DB
        """
        if not self._enabled:
            return None

        # Build instrument data summary
        inst_lines = []
        for inst, feat in features_by_instrument.items():
            sent = sentiment_by_instrument.get(inst, 0)
            rsi = feat.get("rsi_14", 50)
            adx = feat.get("adx", 0)
            macd = feat.get("macd_histogram", 0)
            trend = feat.get("close_vs_ema_long", 0)
            bb = feat.get("bb_percent_b", 0.5)
            atr = feat.get("atr", 0)

            rsi_label = "(oversold)" if rsi < 30 else "(overbought)" if rsi > 70 else ""
            adx_label = "(strong trend)" if adx > 25 else "(weak)" if adx < 20 else ""
            macd_label = "(bullish)" if macd > 0 else "(bearish)"

            inst_lines.append(
                f"{inst}: RSI={rsi:.0f}{rsi_label} ADX={adx:.0f}{adx_label} "
                f"MACD={macd:.5f}{macd_label} BB%B={bb:.2f} "
                f"trend_vs_ema={trend:.4f} ATR={atr:.5f} sentiment={sent:.2f}"
            )

        # Build recent trades summary
        trade_lines = []
        for t in recent_trades[-10:]:
            pnl = t.get("pnl") or 0
            result = "WIN" if pnl > 0 else "LOSS"
            trade_lines.append(
                f"{t.get('instrument', '?')} {t.get('direction', '?')} "
                f"→ {result} {pnl:.0f} ({t.get('close_reason', '')})"
            )

        # Build recent losses for analysis
        losses = [t for t in recent_trades[-10:] if (t.get("pnl") or 0) < 0]
        loss_lines = []
        for t in losses[-5:]:
            loss_lines.append(
                f"{t.get('instrument', '?')} {t.get('direction', '?')}: "
                f"entry={t.get('price', 0):.5f} exit={t.get('exit_price', 0):.5f} "
                f"P&L={t.get('pnl', 0):.0f} reason={t.get('close_reason', '')}"
            )

        # Build pending signals
        signal_lines = []
        for s in pending_signals:
            signal_lines.append(
                f"{s['instrument']} {s['direction']} conf={s['confidence']:.1%}"
            )

        prompt = _CYCLE_PROMPT.format(
            instrument_data="\n".join(inst_lines) or "No data available",
            recent_trades="\n".join(trade_lines) or "No recent trades",
            recent_losses="\n".join(loss_lines) or "No recent losses",
            pending_signals="\n".join(signal_lines) or "No signals above threshold",
        )

        result = self._call_llm_json(prompt)
        if result:
            self._last_advice = result
            self._last_advice_time = datetime.now(UTC)

            regime = result.get("regime", {})
            decisions = result.get("trade_decisions", {})
            approved = sum(1 for d in decisions.values() if d.get("take_trade"))
            vetoed = sum(1 for d in decisions.values() if not d.get("take_trade"))

            log.info(
                "llm_cycle_advice",
                regime=regime.get("type"),
                regime_direction=regime.get("direction"),
                avoid_trading=regime.get("avoid_trading", False),
                signals=len(pending_signals),
                approved=approved,
                vetoed=vetoed,
                lessons=result.get("lessons", "")[:100],
            )
        return result

    def _call_llm_json(self, prompt: str) -> dict | None:
        """Call Claude and parse a JSON response."""
        text = ""
        try:
            response = self._client.messages.create(
                model=self._settings.llm_model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("llm_json_parse_failed", raw=text[:200])
            return None
        except Exception:
            log.exception("llm_advisor_call_failed")
            return None
