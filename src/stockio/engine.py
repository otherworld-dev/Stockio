"""Trading engine — orchestrates the cycle: data → indicators → scoring → trading."""

from __future__ import annotations

import collections
import math
from datetime import datetime, timezone

import structlog

from stockio.broker.base import BrokerBase
from stockio.broker.models import (
    AccountSummary,
    Candle,
    Direction,
    OrderRequest,
    OrderType,
    Signal,
)
from stockio.config import InstrumentConfig, Settings
from stockio.strategy.indicators import build_feature_vector, compute_indicators
from stockio.strategy.notifier import TelegramNotifier
from stockio.strategy.scorer import InstrumentScorer

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------


class RiskManager:
    """Tracks P&L and enforces risk limits."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._peak_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._last_reset_day: int = -1
        self._last_reset_week: int = -1
        self._halted: bool = False
        self._halt_reason: str = ""

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def update(self, account: AccountSummary) -> None:
        """Update equity tracking and reset daily/weekly counters."""
        now = datetime.now(timezone.utc)

        # Track peak equity
        if account.equity > self._peak_equity:
            self._peak_equity = account.equity

        # Reset daily P&L at midnight UTC
        if now.day != self._last_reset_day:
            self._daily_pnl = 0.0
            self._last_reset_day = now.day

        # Reset weekly P&L on Monday
        if now.isocalendar().week != self._last_reset_week:
            self._weekly_pnl = 0.0
            self._last_reset_week = now.isocalendar().week

    def record_pnl(self, pnl: float) -> None:
        self._daily_pnl += pnl
        self._weekly_pnl += pnl

    def check(self, account: AccountSummary) -> tuple[bool, str]:
        """Run all risk checks. Returns (ok, reason)."""
        if self._halted:
            return False, f"Trading halted: {self._halt_reason}"

        s = self._settings

        # Max positions
        if account.open_position_count >= s.max_positions:
            return False, f"Max positions ({s.max_positions}) reached"

        # Daily loss limit
        if self._peak_equity > 0:
            daily_limit = s.daily_loss_limit * self._peak_equity
            if self._daily_pnl < -daily_limit:
                return False, f"Daily loss limit ({s.daily_loss_limit:.0%}) hit"

            # Weekly loss limit
            weekly_limit = s.weekly_loss_limit * self._peak_equity
            if self._weekly_pnl < -weekly_limit:
                return False, f"Weekly loss limit ({s.weekly_loss_limit:.0%}) hit"

            # Max drawdown kill switch
            drawdown = (self._peak_equity - account.equity) / self._peak_equity
            if drawdown >= s.max_drawdown:
                self._halted = True
                self._halt_reason = (
                    f"Max drawdown ({s.max_drawdown:.0%}) breached — "
                    f"peak={self._peak_equity:.2f}, current={account.equity:.2f}"
                )
                return False, self._halt_reason

        return True, ""


def calculate_position_size(
    account: AccountSummary,
    atr: float,
    instrument: InstrumentConfig,
    settings: Settings,
) -> int:
    """Calculate position size based on risk per trade and ATR-based stop distance."""
    risk_amount = account.equity * settings.risk_per_trade
    stop_distance = atr * settings.stop_loss_atr_mult

    if stop_distance <= 0:
        return 0

    units = risk_amount / stop_distance
    # Round down to nearest min_units
    units = max(instrument.min_units, int(math.floor(units / instrument.min_units) * instrument.min_units))
    return units


def calculate_stop_take_profit(
    price: float,
    direction: Direction,
    atr: float,
    settings: Settings,
) -> tuple[float, float]:
    """Calculate stop-loss and take-profit prices."""
    sl_dist = atr * settings.stop_loss_atr_mult
    tp_dist = atr * settings.take_profit_atr_mult

    if direction == Direction.BUY:
        stop_loss = round(price - sl_dist, 5)
        take_profit = round(price + tp_dist, 5)
    else:
        stop_loss = round(price + sl_dist, 5)
        take_profit = round(price - tp_dist, 5)

    return stop_loss, take_profit


# ---------------------------------------------------------------------------
# Trading engine
# ---------------------------------------------------------------------------


class TradingEngine:
    """Runs the periodic trading cycle."""

    def __init__(
        self,
        broker: BrokerBase,
        instruments: dict[str, InstrumentConfig],
        settings: Settings,
        notifier: TelegramNotifier | None = None,
    ) -> None:
        self._broker = broker
        self._instruments = instruments
        self._settings = settings
        self._cycle_count = 0
        self._scorer = InstrumentScorer(settings, settings.models_dir)
        self._risk = RiskManager(settings)
        self._notifier = notifier

        # Bounded candle cache — one deque per instrument, maxlen prevents leaks
        self._candle_cache: dict[str, collections.deque[Candle]] = {
            name: collections.deque(maxlen=settings.lookback_bars)
            for name in instruments
        }
        self._warmed_up: set[str] = set()

        # Sentiment scores (populated by Phase 4)
        self._sentiment: dict[str, float] = {}
        # Latest features per instrument (used for outcome tracking in Phase 5)
        self._latest_features: dict[str, dict[str, float]] = {}

    def run_cycle(self) -> None:
        """Execute one trading cycle."""
        self._cycle_count += 1
        cycle_log = log.bind(cycle=self._cycle_count)
        cycle_log.info("cycle_start")

        # Step 1: Update candle data
        for name in self._instruments:
            try:
                self._update_candles(name)
            except Exception:
                cycle_log.exception("candle_fetch_failed", instrument=name)

        # Step 2: Compute indicators and score each instrument
        signals: dict[str, Signal] = {}
        for name in self._warmed_up:
            try:
                candles = list(self._candle_cache[name])
                df = compute_indicators(candles, self._settings)
                if df.empty:
                    continue
                features = build_feature_vector(df, self._settings)
                self._latest_features[name] = features
                sentiment = self._sentiment.get(name, 0.0)
                signal = self._scorer.score_instrument(name, features, sentiment)
                signals[name] = signal
            except Exception:
                cycle_log.exception("scoring_failed", instrument=name)

        # Step 3: Rank instruments
        ranked = self._scorer.rank_instruments(signals)

        if ranked:
            cycle_log.info(
                "instrument_ranking",
                top=[
                    {
                        "instrument": s.instrument,
                        "direction": s.direction.value,
                        "confidence": round(s.confidence, 3),
                    }
                    for s in ranked[:5]
                ],
            )
        else:
            cycle_log.info("no_tradeable_signals")

        # Step 4: Execute trades
        self._execute_trades(ranked, cycle_log)

        cycle_log.info(
            "cycle_complete",
            instruments_ready=len(self._warmed_up),
            signals_generated=len(signals),
            tradeable=len(ranked),
        )

    def _execute_trades(self, ranked: list[Signal], cycle_log: structlog.BoundLogger) -> None:
        """Attempt to trade the top-ranked instruments."""
        if not ranked:
            return

        try:
            account = self._broker.get_account()
        except Exception:
            cycle_log.exception("account_fetch_failed")
            return

        self._risk.update(account)

        if self._risk.is_halted:
            cycle_log.critical("trading_halted", reason=self._risk.halt_reason)
            if self._notifier:
                self._notifier.notify_risk_halt(self._risk.halt_reason)
            return

        for signal in ranked:
            # Risk check before each trade
            risk_ok, reason = self._risk.check(account)
            if not risk_ok:
                cycle_log.warning("risk_check_failed", reason=reason, instrument=signal.instrument)
                break  # Stop trying if we hit a portfolio-level limit

            features = self._latest_features.get(signal.instrument, {})
            atr = features.get("atr", 0)
            if atr <= 0:
                cycle_log.warning("invalid_atr", instrument=signal.instrument)
                continue

            instrument_cfg = self._instruments[signal.instrument]

            # Get current price
            try:
                quote = self._broker.get_price(signal.instrument)
            except Exception:
                cycle_log.exception("price_fetch_failed", instrument=signal.instrument)
                continue

            entry_price = quote.ask if signal.direction == Direction.BUY else quote.bid
            units = calculate_position_size(account, atr, instrument_cfg, self._settings)
            if units <= 0:
                continue

            stop_loss, take_profit = calculate_stop_take_profit(
                entry_price, signal.direction, atr, self._settings
            )

            order = OrderRequest(
                instrument=signal.instrument,
                direction=signal.direction,
                units=units,
                order_type=OrderType.MARKET,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                signal_confidence=signal.confidence,
            )

            try:
                trade_id = self._broker.submit_order(order)
                cycle_log.info(
                    "trade_executed",
                    instrument=signal.instrument,
                    direction=signal.direction.value,
                    units=units,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=round(signal.confidence, 3),
                    trade_id=trade_id,
                )
                if self._notifier:
                    self._notifier.notify_trade(order, trade_id)

                # Refresh account after trade for next risk check
                account = self._broker.get_account()
            except Exception:
                cycle_log.exception("order_failed", instrument=signal.instrument)
                if self._notifier:
                    self._notifier.notify_error(
                        f"Order failed for {signal.instrument}: check logs"
                    )

    def _update_candles(self, instrument: str) -> None:
        """Fetch and cache candles for a single instrument."""
        cache = self._candle_cache[instrument]

        if instrument not in self._warmed_up:
            candles = self._broker.get_candles(
                instrument=instrument,
                granularity=self._settings.granularity,
                count=self._settings.lookback_bars,
            )
            cache.clear()
            cache.extend(candles)
            if candles:
                self._warmed_up.add(instrument)
                log.info("warmup_complete", instrument=instrument, bars=len(candles))
        else:
            candles = self._broker.get_candles(
                instrument=instrument,
                granularity=self._settings.granularity,
                count=5,
            )
            if cache:
                last_ts = cache[-1].timestamp
                new = [c for c in candles if c.timestamp > last_ts]
            else:
                new = candles
            cache.extend(new)

    def update_sentiment(self, sentiment: dict[str, float]) -> None:
        """Update cached sentiment scores (called by sentiment analyzer)."""
        self._sentiment = sentiment

    @property
    def candle_cache(self) -> dict[str, collections.deque[Candle]]:
        return self._candle_cache

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def scorer(self) -> InstrumentScorer:
        return self._scorer

    @property
    def risk(self) -> RiskManager:
        return self._risk
