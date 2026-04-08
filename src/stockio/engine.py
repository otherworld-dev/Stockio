"""Trading engine — orchestrates the cycle: data → indicators → scoring → trading."""

from __future__ import annotations

import collections
import math
from datetime import UTC, datetime

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
from stockio.strategy.scorer import InstrumentScorer, OutcomeTracker, retrain_model

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Tracks consecutive failures for an external service and disables it temporarily."""

    def __init__(self, name: str, max_failures: int = 5, cooldown_seconds: int = 900) -> None:
        self.name = name
        self._max_failures = max_failures
        self._cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._open_until: datetime | None = None

    @property
    def is_open(self) -> bool:
        """True if circuit is open (service disabled)."""
        if self._open_until is None:
            return False
        if datetime.now(UTC) >= self._open_until:
            self._reset()
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures:
            from datetime import timedelta

            self._open_until = datetime.now(UTC) + timedelta(
                seconds=self._cooldown_seconds
            )
            log.warning(
                "circuit_breaker_opened",
                service=self.name,
                cooldown_seconds=self._cooldown_seconds,
            )

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._open_until = None
        log.info("circuit_breaker_closed", service=self.name)


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
        now = datetime.now(UTC)

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
    min_u = instrument.min_units
    units = max(min_u, int(math.floor(units / min_u) * min_u))
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
        self._outcome_tracker = OutcomeTracker(settings, settings.data_dir)

        # Circuit breakers for external services
        self._cb_oanda = CircuitBreaker("oanda")
        self._cb_newsapi = CircuitBreaker("newsapi")
        self._cb_anthropic = CircuitBreaker("anthropic")

        # Bounded candle cache — one deque per instrument, maxlen prevents leaks
        self._candle_cache: dict[str, collections.deque[Candle]] = {
            name: collections.deque(maxlen=settings.lookback_bars)
            for name in instruments
        }
        self._warmed_up: set[str] = set()

        # Sentiment scores
        self._sentiment: dict[str, float] = {}
        # Latest features per instrument (used for outcome tracking)
        self._latest_features: dict[str, dict[str, float]] = {}
        # Track when last retrain happened
        self._last_retrain_count: int = 0
        # Daily summary tracking
        self._trades_today: list[dict] = []
        self._last_summary_day: int = -1
        # Heartbeat tracking
        self._last_heartbeat: datetime | None = None

    def run_cycle(self) -> None:
        """Execute one trading cycle."""
        self._cycle_count += 1
        cycle_log = log.bind(cycle=self._cycle_count)
        cycle_log.info("cycle_start")

        # Heartbeat
        self._maybe_heartbeat(cycle_log)

        # Circuit breaker: skip cycle if OANDA is down
        if self._cb_oanda.is_open:
            cycle_log.warning("cycle_skipped", reason="oanda_circuit_breaker_open")
            return

        # Step 1: Update candle data
        oanda_failed = 0
        for name in self._instruments:
            try:
                self._update_candles(name)
                self._cb_oanda.record_success()
            except Exception:
                cycle_log.exception("candle_fetch_failed", instrument=name)
                oanda_failed += 1
                self._cb_oanda.record_failure()

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

        # Step 5: Resolve pending outcomes for model training
        self._resolve_and_maybe_retrain(cycle_log)

        cycle_log.info(
            "cycle_complete",
            instruments_ready=len(self._warmed_up),
            signals_generated=len(signals),
            tradeable=len(ranked),
            pending_outcomes=self._outcome_tracker.pending_count,
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
                self._trades_today.append({
                    "instrument": signal.instrument,
                    "direction": signal.direction.value,
                    "units": units,
                    "confidence": signal.confidence,
                    "trade_id": trade_id,
                })
                if self._notifier:
                    self._notifier.notify_trade(order, trade_id)

                # Record prediction for outcome tracking
                self._outcome_tracker.record_pending(
                    instrument=signal.instrument,
                    features=features,
                    direction=signal.direction,
                    confidence=signal.confidence,
                    entry_price=entry_price,
                    atr=atr,
                    current_cycle=self._cycle_count,
                )

                # Refresh account after trade for next risk check
                account = self._broker.get_account()
            except Exception:
                cycle_log.exception("order_failed", instrument=signal.instrument)
                if self._notifier:
                    self._notifier.notify_error(
                        f"Order failed for {signal.instrument}: check logs"
                    )

    def _resolve_and_maybe_retrain(self, cycle_log: structlog.BoundLogger) -> None:
        """Resolve pending outcomes and retrain model if enough new data."""

        def _get_mid_price(instrument: str) -> float:
            quote = self._broker.get_price(instrument)
            return (quote.bid + quote.ask) / 2

        resolved = self._outcome_tracker.resolve_outcomes(
            self._cycle_count, _get_mid_price
        )
        if resolved > 0:
            acc = self._outcome_tracker.rolling_accuracy
            cycle_log.info(
                "outcomes_resolved",
                count=resolved,
                rolling_accuracy=round(acc, 3) if acc is not None else None,
                total_training_samples=self._outcome_tracker.training_data_count,
            )

        # Check for model degradation
        if self._outcome_tracker.is_degraded:
            self._scorer.set_rules_fallback(True)
            if self._notifier:
                self._notifier.notify_error(
                    f"Model degraded (accuracy={self._outcome_tracker.rolling_accuracy:.1%}), "
                    f"falling back to rules-based scoring"
                )

        # Retrain if enough new data (every 200 new samples since last retrain)
        current_count = self._outcome_tracker.training_data_count
        if current_count - self._last_retrain_count >= 200:
            cycle_log.info("retrain_triggered", samples=current_count)
            success = retrain_model(
                self._settings, self._settings.data_dir, self._settings.models_dir
            )
            if success:
                self._scorer.reload_model()
                self._scorer.set_rules_fallback(False)
                self._last_retrain_count = current_count

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

    def _maybe_heartbeat(self, cycle_log: structlog.BoundLogger) -> None:
        """Log a heartbeat if enough time has passed."""
        import resource
        import sys

        now = datetime.now(UTC)
        if self._last_heartbeat is not None:
            elapsed = (now - self._last_heartbeat).total_seconds()
            if elapsed < self._settings.heartbeat_seconds:
                return

        # Memory usage (platform-dependent)
        try:
            if sys.platform != "win32":
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            else:
                import os

                import psutil  # type: ignore[import-untyped]

                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            mem_mb = 0

        cycle_log.info(
            "heartbeat",
            cycles_run=self._cycle_count,
            instruments_warmed=len(self._warmed_up),
            pending_outcomes=self._outcome_tracker.pending_count,
            training_samples=self._outcome_tracker.training_data_count,
            trades_today=len(self._trades_today),
            memory_mb=round(mem_mb, 1),
            halted=self._risk.is_halted,
        )
        self._last_heartbeat = now

    def maybe_daily_summary(self) -> None:
        """Send daily summary via Telegram if it's the right hour."""
        now = datetime.now(UTC)
        if now.hour != self._settings.daily_summary_hour:
            return
        if now.day == self._last_summary_day:
            return

        self._last_summary_day = now.day

        try:
            account = self._broker.get_account()
        except Exception:
            return

        stats = {
            "Date": now.strftime("%Y-%m-%d"),
            "Trades Today": len(self._trades_today),
            "Balance": account.balance,
            "Equity": account.equity,
            "Unrealized P&L": account.unrealized_pnl,
            "Open Positions": account.open_position_count,
            "Model Accuracy": (
                f"{self._outcome_tracker.rolling_accuracy:.1%}"
                if self._outcome_tracker.rolling_accuracy is not None
                else "N/A"
            ),
            "Training Samples": self._outcome_tracker.training_data_count,
            "Cycles Run": self._cycle_count,
        }

        log.info("daily_summary", **stats)
        if self._notifier:
            self._notifier.notify_daily_summary(stats)

        # Reset daily counters
        self._trades_today = []

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
