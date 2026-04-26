"""Trading engine — orchestrates the cycle: data → indicators → scoring → trading."""

from __future__ import annotations

import collections
import math
import threading
from datetime import UTC, datetime

import structlog

from stockio import db
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
        self._last_reset_date: object = None  # date object, not just day number
        self._last_reset_week: tuple = ()  # (year, week) tuple
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
        today = now.date()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._last_reset_date = today

        # Reset weekly P&L on Monday
        iso = now.isocalendar()
        year_week = (iso.year, iso.week)
        if year_week != self._last_reset_week:
            self._weekly_pnl = 0.0
            self._last_reset_week = year_week

    def record_pnl(self, pnl: float) -> None:
        self._daily_pnl += pnl
        self._weekly_pnl += pnl

    def check(self, account: AccountSummary) -> tuple[bool, str]:
        """Run all risk checks. Returns (ok, reason). Reads DB overrides."""
        if self._halted:
            return False, f"Trading halted: {self._halt_reason}"

        s = self._settings
        max_pos = db.get_int_setting("max_positions", s.max_positions)
        daily_lim = db.get_float_setting("daily_loss_limit", s.daily_loss_limit)
        weekly_lim = db.get_float_setting("weekly_loss_limit", s.weekly_loss_limit)
        max_dd = db.get_float_setting("max_drawdown", s.max_drawdown)

        # Max positions
        if account.open_position_count >= max_pos:
            return False, f"Max positions ({max_pos}) reached"

        # Margin check — don't trade if margin used exceeds 50% of equity
        max_margin_pct = db.get_float_setting("max_margin_pct", 0.50)
        if account.equity > 0 and account.margin_used > 0:
            margin_pct = account.margin_used / account.equity
            if margin_pct >= max_margin_pct:
                return False, f"Margin too high ({margin_pct:.0%}, limit {max_margin_pct:.0%})"

        # Daily loss limit (can be disabled via settings for paper trading)
        disable_daily = db.get_setting("disable_daily_limit")
        if self._peak_equity > 0:
            if not disable_daily and self._daily_pnl < -(daily_lim * self._peak_equity):
                return False, f"Daily loss limit ({daily_lim:.0%}) hit"

            # Weekly loss limit
            if not disable_daily and self._weekly_pnl < -(weekly_lim * self._peak_equity):
                return False, f"Weekly loss limit ({weekly_lim:.0%}) hit"

            # Max drawdown kill switch
            drawdown = (self._peak_equity - account.equity) / self._peak_equity
            if drawdown >= max_dd:
                self._halted = True
                self._halt_reason = (
                    f"Max drawdown ({max_dd:.0%}) breached — "
                    f"peak={self._peak_equity:.2f}, current={account.equity:.2f}"
                )
                return False, self._halt_reason

        return True, ""


def calculate_position_size(
    account: AccountSummary,
    atr: float,
    instrument: InstrumentConfig,
    settings: Settings,
    current_price: float = 0,
) -> int:
    """Calculate position size based on risk per trade and ATR-based stop distance.

    Converts risk to account currency (GBP) using pip value conversion.
    """
    from stockio.broker.yahoo import _pip_value_in_gbp

    risk_pct = db.get_float_setting("risk_per_trade", settings.risk_per_trade)
    sl_mult = db.get_float_setting("stop_loss_atr_mult", settings.stop_loss_atr_mult)

    risk_amount = account.equity * risk_pct
    stop_distance = atr * sl_mult

    if stop_distance <= 0:
        return 0

    # Convert stop distance to account currency per unit
    conversion = _pip_value_in_gbp(instrument.name, 1, current_price or 1.0)
    stop_in_gbp_per_unit = stop_distance * conversion

    if stop_in_gbp_per_unit <= 0:
        return 0

    units = risk_amount / stop_in_gbp_per_unit
    min_u = instrument.min_units
    units = max(min_u, int(math.floor(units / min_u) * min_u))
    return units


def calculate_stop_take_profit(
    price: float,
    direction: Direction,
    atr: float,
    settings: Settings,
    sl_mult_override: float | None = None,
    tp_mult_override: float | None = None,
) -> tuple[float, float]:
    """Calculate stop-loss and take-profit prices."""
    sl_mult = sl_mult_override or db.get_float_setting("stop_loss_atr_mult", settings.stop_loss_atr_mult)
    tp_mult = tp_mult_override or db.get_float_setting("take_profit_atr_mult", settings.take_profit_atr_mult)
    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

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
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self._broker = broker
        self._instruments = instruments
        self._settings = settings
        self._shutdown = shutdown_event
        self._cycle_count = 0

        # LLM trade advisor
        from stockio.strategy.llm_advisor import LLMAdvisor
        self._advisor = LLMAdvisor(settings)
        self._last_signals: dict[str, Signal] = {}  # Cached from last cycle
        self._last_cycle_time: datetime | None = None
        self._scorer = InstrumentScorer(settings, settings.models_dir)
        self._risk = RiskManager(settings)
        self._notifier = notifier
        self._outcome_tracker = OutcomeTracker(settings, settings.data_dir)

        # Economic calendar
        from stockio.strategy.calendar import EconomicCalendar

        self._calendar = EconomicCalendar(settings)

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
        # Stop-loss cooldown — don't re-enter instruments that just stopped out
        self._sl_cooldown: dict[str, int] = {}  # instrument → cycle when cooldown expires
        self._sl_cooldown_cycles = 8  # Wait 8 cycles (~2 hours at M15) after SL hit
        # Trailing stop state — tracks the best price seen for each open trade
        self._trailing_best: dict[str, float] = {}  # trade_id → best price seen
        # Daily loss counter — block instrument after repeated losses in a day
        self._daily_losses: dict[str, int] = {}  # instrument → loss count today
        self._daily_losses_date: object = None  # date of last reset
        self._max_daily_losses_per_instrument = 2  # Block after 2 losses/day
        # Cached account data — updated each cycle by the bot thread.
        # The web API reads this instead of calling the broker directly,
        # avoiding thread-safety issues with the shared requests.Session.
        self._last_account: AccountSummary | None = None

        # Sync with broker on startup
        self._sync_on_startup()

    def _sync_on_startup(self) -> None:
        """Sync DB trades with actual broker positions on startup."""
        # 1) Close stale DB trades that no longer exist at broker
        try:
            open_db_trades = db.get_open_trades()
            if open_db_trades:
                broker_positions = {p.trade_id: p for p in self._broker.get_positions()}

                closed_count = 0
                for trade in open_db_trades:
                    trade_id = trade["trade_id"]
                    instrument = trade["instrument"]
                    if trade_id not in broker_positions:
                        pnl = 0.0
                        exit_price = trade["price"]
                        close_reason = "Closed while bot offline"

                        details = self._broker.get_closed_trade_details(trade_id)
                        if details:
                            exit_price = details.get("close_price", exit_price)
                            pnl = details.get("realized_pnl", 0.0)
                            state = details.get("state", "")
                            if state:
                                close_reason = f"Closed while offline ({state})"

                        db.close_trade(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            pnl=pnl,
                            close_reason=close_reason,
                        )

                        # Always cooldown — PnL from OANDA may be 0 for
                        # recently closed trades, so don't rely on pnl < 0
                        self._sl_cooldown[instrument] = self._sl_cooldown_cycles
                        closed_count += 1

                if closed_count:
                    log.info("startup_sync", closed_stale_trades=closed_count)
        except Exception:
            log.debug("startup_sync_skipped")

        # 2) Restore cooldowns from recent losing trades already in DB.
        #    This covers trades closed by the *previous* session — they're
        #    already CLOSED in DB so the stale-trade sync above won't see them.
        try:
            recent_losses = db.get_recently_closed_losses(hours=2)
            for row in recent_losses:
                inst = row["instrument"]
                if inst not in self._sl_cooldown:
                    self._sl_cooldown[inst] = self._sl_cooldown_cycles
            if recent_losses:
                log.info(
                    "cooldowns_restored",
                    instruments=[r["instrument"] for r in recent_losses],
                )
        except Exception:
            log.debug("cooldown_restore_skipped")

    def run_cycle(self) -> None:
        """Execute one trading cycle."""
        self._cycle_count += 1
        self._last_cycle_time = datetime.now(UTC)
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
            if self._shutdown and self._shutdown.is_set():
                cycle_log.info("cycle_aborted_shutdown")
                return
            try:
                self._update_candles(name)
                self._cb_oanda.record_success()
            except Exception:
                cycle_log.exception("candle_fetch_failed", instrument=name)
                oanda_failed += 1
                self._cb_oanda.record_failure()

        # Refresh economic calendar if needed
        if self._calendar.needs_refresh():
            self._calendar.refresh()

        # Step 2: Compute indicators and score each instrument
        signals: dict[str, Signal] = {}
        for name in self._warmed_up:
            try:
                candles = list(self._candle_cache[name])
                df = compute_indicators(candles, self._settings)
                if df.empty:
                    continue
                features = build_feature_vector(df, self._settings)
                # Merge calendar features
                features.update(self._calendar.get_features(name))
                self._latest_features[name] = features
                sentiment = self._sentiment.get(name, 0.0)
                signal = self._scorer.score_instrument(name, features, sentiment)
                signals[name] = signal
            except Exception:
                cycle_log.exception("scoring_failed", instrument=name)

        # Cache scored signals for the dashboard API
        self._last_signals = signals

        # Step 3: Rank instruments and filter correlated pairs
        from stockio.strategy.correlation import filter_correlated_signals

        ranked = self._scorer.rank_instruments(signals)
        try:
            existing_positions = self._broker.get_positions()
            open_instruments = {p.instrument for p in existing_positions}
        except Exception:
            open_instruments = set()
        ranked = filter_correlated_signals(ranked, open_instruments)

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

        # Step 3b: Get LLM advice for this cycle
        if self._advisor.enabled and ranked:
            try:
                recent_trades = db.get_trade_history(limit=10)
                pending = [
                    {"instrument": s.instrument, "direction": s.direction.value,
                     "confidence": s.confidence}
                    for s in ranked
                ]
                self._advisor.advise_cycle(
                    self._latest_features,
                    self._sentiment,
                    pending,
                    recent_trades,
                )
            except Exception:
                cycle_log.exception("llm_advisor_failed")

        # Step 4: Check open positions for SL/TP exits
        self._check_position_exits(cycle_log)

        # Step 5: Execute new trades
        self._execute_trades(ranked, cycle_log)

        # Step 6: Resolve pending outcomes for model training
        self._resolve_and_maybe_retrain(cycle_log)

        # Step 7: Cache account data and persist snapshot to SQLite
        try:
            self._last_account = self._broker.get_account()
        except Exception:
            cycle_log.exception("account_cache_failed")
        self._persist_cycle(cycle_log, signals, ranked)
        self._outcome_tracker.persist_pending()

        cycle_log.info(
            "cycle_complete",
            instruments_ready=len(self._warmed_up),
            signals_generated=len(signals),
            tradeable=len(ranked),
            pending_outcomes=self._outcome_tracker.pending_count,
        )

    def _persist_cycle(
        self,
        cycle_log: structlog.BoundLogger,
        signals: dict[str, Signal],
        ranked: list[Signal],
    ) -> None:
        """Save snapshot and bot log to SQLite."""
        try:
            account = self._last_account
            if not account:
                cycle_log.warning("persist_skipped_no_account")
                return
            db.record_snapshot(
                balance=account.balance,
                equity=account.equity,
                unrealized_pnl=account.unrealized_pnl,
                open_positions=account.open_position_count,
                cycle=self._cycle_count,
            )
            db.record_bot_log(
                cycle=self._cycle_count,
                summary={
                    "instruments_scored": len(signals),
                    "tradeable": len(ranked),
                    "top_signals": [
                        {
                            "instrument": s.instrument,
                            "direction": s.direction.value,
                            "confidence": round(s.confidence, 3),
                        }
                        for s in ranked[:3]
                    ],
                    "sentiment": {
                        k: round(v, 3) for k, v in self._sentiment.items() if v != 0
                    },
                    "model_accuracy": (
                        round(self._outcome_tracker.rolling_accuracy, 3)
                        if self._outcome_tracker.rolling_accuracy is not None
                        else None
                    ),
                },
            )
        except Exception:
            cycle_log.exception("db_persist_cycle_failed")

    def _check_position_exits(self, cycle_log: structlog.BoundLogger) -> None:
        """Check open positions for stop-loss / take-profit hits.

        For paper mode (YahooBroker): we simulate SL/TP exits here.
        For live mode (OANDA): SL/TP are handled server-side, but we sync
        the exit to our DB by checking which positions have closed.
        """
        open_trades = db.get_open_trades()
        if not open_trades:
            return

        # Get current open positions from broker
        try:
            broker_positions = {p.trade_id: p for p in self._broker.get_positions()}
        except Exception:
            return

        for trade in open_trades:
            trade_id = trade["trade_id"]
            instrument = trade["instrument"]
            direction = trade["direction"]
            entry_price = trade["price"]
            sl = trade["stop_loss"]
            tp = trade["take_profit"]
            units = trade["units"]

            # Check if position still exists at broker
            if trade_id in broker_positions:
                # Trailing stop — update SL if price has moved favourably
                self._maybe_trail_stop(
                    trade_id, instrument, direction, entry_price,
                    sl, tp, broker_positions[trade_id], cycle_log
                )

                # OANDA handles SL/TP server-side — skip client-side check
                from stockio.broker.oanda import OandaBroker

                if isinstance(self._broker, OandaBroker):
                    continue

                # Position still open — check SL/TP (paper mode only)
                try:
                    quote = self._broker.get_price(instrument)
                    mid = (quote.bid + quote.ask) / 2

                    close_reason = None
                    exit_price = mid

                    if direction == "BUY":
                        if sl and mid <= sl:
                            close_reason = "Stop loss hit"
                        elif tp and mid >= tp:
                            close_reason = "Take profit hit"
                    elif direction == "SELL":
                        if sl and mid >= sl:
                            close_reason = "Stop loss hit"
                        elif tp and mid <= tp:
                            close_reason = "Take profit hit"

                    if close_reason:
                        # Set cooldown and record daily loss if stop-loss hit
                        if "Stop loss" in close_reason:
                            self._sl_cooldown[instrument] = (
                                self._cycle_count + self._sl_cooldown_cycles
                            )
                            self._record_instrument_loss(instrument)
                            log.info(
                                "sl_cooldown_set",
                                instrument=instrument,
                                until_cycle=self._sl_cooldown[instrument],
                            )

                        # Calculate P&L (convert to account currency)
                        from stockio.broker.yahoo import _pip_value_in_gbp

                        conversion = _pip_value_in_gbp(instrument, units, exit_price)
                        if direction == "BUY":
                            pnl = (exit_price - entry_price) * units * conversion
                        else:
                            pnl = (entry_price - exit_price) * units * conversion

                        # Close at broker
                        try:
                            self._broker.close_position(trade_id)
                        except Exception:
                            cycle_log.exception(
                                "position_close_failed", trade_id=trade_id
                            )
                            continue

                        # Record exit in DB and track P&L for risk limits
                        db.close_trade(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            pnl=pnl,
                            close_reason=close_reason,
                        )
                        self._risk.record_pnl(pnl)
                        self._trailing_best.pop(trade_id, None)
                        cycle_log.info(
                            "position_exited",
                            trade_id=trade_id,
                            instrument=instrument,
                            direction=direction,
                            reason=close_reason,
                            pnl=round(pnl, 4),
                        )
                        if self._notifier:
                            self._notifier.notify_error(
                                f"Position closed: {instrument} {direction}\n"
                                f"Reason: {close_reason}\n"
                                f"P&L: {pnl:+.4f}"
                            )
                except Exception:
                    cycle_log.exception(
                        "exit_check_failed", trade_id=trade_id
                    )
            else:
                # Position no longer at broker (closed by SL/TP on broker side)
                # Try to get actual close details from broker
                pnl = 0.0
                exit_price = entry_price
                close_reason = "Closed by broker (SL/TP)"

                details = self._broker.get_closed_trade_details(trade_id)
                if details:
                    exit_price = details.get("close_price", entry_price)
                    pnl = details.get("realized_pnl", 0.0)
                    state = details.get("state", "")
                    if state:
                        close_reason = f"Closed by broker ({state})"
                    cycle_log.info(
                        "trade_closed_details",
                        trade_id=trade_id,
                        instrument=instrument,
                        exit_price=exit_price,
                        pnl=round(pnl, 2),
                    )
                else:
                    # Fallback: estimate from current price
                    try:
                        from stockio.broker.yahoo import _pip_value_in_gbp

                        quote = self._broker.get_price(instrument)
                        exit_price = (quote.bid + quote.ask) / 2
                        conversion = _pip_value_in_gbp(
                            instrument, units, exit_price
                        )
                        if direction == "BUY":
                            pnl = (
                                (exit_price - entry_price) * units * conversion
                            )
                        else:
                            pnl = (
                                (entry_price - exit_price) * units * conversion
                            )
                    except Exception:
                        pass

                # Set cooldown and record daily loss if loss
                if pnl < 0:
                    self._sl_cooldown[instrument] = (
                        self._cycle_count + self._sl_cooldown_cycles
                    )
                    self._record_instrument_loss(instrument)

                db.close_trade(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    pnl=pnl,
                    close_reason=close_reason,
                )
                self._risk.record_pnl(pnl)
                self._trailing_best.pop(trade_id, None)
                cycle_log.info(
                    "position_closed_externally",
                    trade_id=trade_id,
                    instrument=instrument,
                    pnl=round(pnl, 4),
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

        # Get existing positions to avoid duplicate trades on same instrument
        try:
            open_positions = self._broker.get_positions()
            open_instruments = {p.instrument for p in open_positions}
        except Exception:
            cycle_log.exception("positions_fetch_failed")
            open_instruments = set()

        for signal in ranked:
            # Skip instruments with existing open positions
            if signal.instrument in open_instruments:
                cycle_log.debug(
                    "skipped_duplicate", instrument=signal.instrument
                )
                continue

            # Skip instruments in stop-loss cooldown
            cooldown_until = self._sl_cooldown.get(signal.instrument, 0)
            if self._cycle_count < cooldown_until:
                cycle_log.debug(
                    "skipped_sl_cooldown",
                    instrument=signal.instrument,
                    cycles_remaining=cooldown_until - self._cycle_count,
                )
                continue

            # Skip instruments that have lost too many times today
            self._reset_daily_losses_if_new_day()
            daily_count = self._daily_losses.get(signal.instrument, 0)
            if daily_count >= self._max_daily_losses_per_instrument:
                cycle_log.debug(
                    "skipped_daily_loss_limit",
                    instrument=signal.instrument,
                    losses_today=daily_count,
                )
                continue

            # Risk check before each trade
            risk_ok, reason = self._risk.check(account)
            if not risk_ok:
                cycle_log.warning(
                    "risk_check_failed",
                    reason=reason,
                    instrument=signal.instrument,
                )
                break  # Stop trying if we hit a portfolio-level limit

            # Check LLM advisor decision
            llm_decision = self._advisor.get_trade_decision(signal.instrument)
            if llm_decision and not llm_decision.get("take_trade", True):
                cycle_log.info(
                    "llm_vetoed",
                    instrument=signal.instrument,
                    reason=llm_decision.get("reason", "")[:100],
                )
                continue

            # Check if LLM regime says avoid trading entirely
            regime = self._advisor.regime
            if regime and regime.get("avoid_trading"):
                cycle_log.info(
                    "llm_regime_avoid",
                    reason=regime.get("reason", "")[:100],
                )
                break

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
            units = calculate_position_size(
                account, atr, instrument_cfg, self._settings, current_price=entry_price
            )
            if units <= 0:
                continue

            # Scale position size when model is unproven or underperforming
            acc = self._outcome_tracker.rolling_accuracy
            live_samples = len(self._outcome_tracker._recent_outcomes)
            if live_samples < 50:
                # New model: half size until we have enough live data
                units = max(instrument_cfg.min_units, units // 2)
                cycle_log.info(
                    "position_scaled_new_model",
                    instrument=signal.instrument,
                    live_samples=live_samples,
                    scale="50%",
                )
            elif acc is not None and acc < 0.50:
                # Underperforming: scale proportionally
                scale = max(0.25, acc / 0.50)
                units = max(instrument_cfg.min_units, int(units * scale))
                cycle_log.info(
                    "position_scaled_low_accuracy",
                    instrument=signal.instrument,
                    accuracy=round(acc, 3),
                    scale=f"{scale:.0%}",
                )

            # Get optimized params: LLM advice > Level 2/3 optimizer > defaults
            sl_override = None
            tp_override = None
            param_source = "default"

            # Check LLM-suggested params first
            if llm_decision:
                llm_sl = llm_decision.get("sl_mult")
                llm_tp = llm_decision.get("tp_mult")
                if llm_sl and 0.8 <= llm_sl <= 3.5:
                    sl_override = llm_sl
                if llm_tp and 0.8 <= llm_tp <= 5.0:
                    tp_override = llm_tp
                if sl_override or tp_override:
                    param_source = "llm"

            # Fall back to Level 2/3 optimizer
            if not sl_override and not tp_override:
                from stockio.strategy.optimizer import get_instrument_params
                inst_params = get_instrument_params(
                    signal.instrument, self._settings, self._settings.data_dir
                )
                if inst_params.level > 1:
                    sl_override = inst_params.sl_atr_mult
                    tp_override = inst_params.tp_atr_mult
                    param_source = f"L{inst_params.level}"

            stop_loss, take_profit = calculate_stop_take_profit(
                entry_price, signal.direction, atr, self._settings,
                sl_mult_override=sl_override,
                tp_mult_override=tp_override,
            )
            if param_source != "default":
                cycle_log.info(
                    "using_custom_params",
                    instrument=signal.instrument,
                    source=param_source,
                    sl_mult=sl_override,
                    tp_mult=tp_override,
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

                # Persist trade to SQLite
                try:
                    db.record_trade(
                        order=order,
                        trade_id=trade_id,
                        fill_price=entry_price,
                        reason=f"signal confidence {signal.confidence:.3f}",
                    )
                except Exception:
                    cycle_log.exception("db_record_trade_failed")

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
                try:
                    account = self._broker.get_account()
                except Exception:
                    cycle_log.exception("account_refresh_failed_after_trade")
                    break  # Can't trust risk checks with stale data

                open_instruments.add(signal.instrument)
            except Exception:
                cycle_log.exception("order_failed", instrument=signal.instrument)
                if self._notifier:
                    self._notifier.notify_error(
                        f"Order failed for {signal.instrument}: check logs"
                    )

    def _reset_daily_losses_if_new_day(self) -> None:
        """Reset the per-instrument daily loss counters at midnight UTC."""
        today = datetime.now(UTC).date()
        if today != self._daily_losses_date:
            self._daily_losses.clear()
            self._daily_losses_date = today

    def _record_instrument_loss(self, instrument: str) -> None:
        """Increment the daily loss counter for an instrument."""
        self._reset_daily_losses_if_new_day()
        self._daily_losses[instrument] = self._daily_losses.get(instrument, 0) + 1
        log.info(
            "daily_loss_recorded",
            instrument=instrument,
            losses_today=self._daily_losses[instrument],
            limit=self._max_daily_losses_per_instrument,
        )

    def _maybe_trail_stop(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry_price: float,
        current_sl: float | None,
        position,  # Position dataclass from broker
        cycle_log,
    ) -> None:
        """Trail the stop-loss behind price as it moves favourably.

        Strategy:
        - After price moves 1x ATR in our favour, move SL to breakeven
        - After that, trail SL at 1x ATR behind the best price seen
        """
        if not current_sl:
            return

        features = self._latest_features.get(instrument, {})
        atr = features.get("atr", 0)
        if atr <= 0:
            return

        # Get current price from position's unrealized P&L direction
        try:
            quote = self._broker.get_price(instrument)
            mid = (quote.bid + quote.ask) / 2
        except Exception:
            return

        # Track best price seen for this trade
        best = self._trailing_best.get(trade_id)
        if direction == "BUY":
            if best is None or mid > best:
                self._trailing_best[trade_id] = mid
                best = mid
            # Only trail if price has moved at least 1x ATR in our favour
            if best - entry_price < atr:
                return
            # New SL: trail 1x ATR behind best price
            new_sl = round(best - atr, 5)
        else:  # SELL
            if best is None or mid < best:
                self._trailing_best[trade_id] = mid
                best = mid
            if entry_price - best < atr:
                return
            new_sl = round(best + atr, 5)

        # Only update if the new SL is better (tighter) than current
        if direction == "BUY" and new_sl <= current_sl:
            return
        if direction == "SELL" and new_sl >= current_sl:
            return

        # Update at broker and in DB
        try:
            self._broker.modify_trade_sl(trade_id, new_sl)
            db.update_trade_sl(trade_id, new_sl)
            cycle_log.info(
                "trailing_stop_updated",
                trade_id=trade_id,
                instrument=instrument,
                old_sl=current_sl,
                new_sl=new_sl,
                best_price=best,
            )
        except Exception:
            cycle_log.exception("trailing_stop_failed", trade_id=trade_id)

    def _resolve_and_maybe_retrain(self, cycle_log: structlog.BoundLogger) -> None:
        """Resolve pending outcomes and retrain model if enough new data."""

        def _get_price_range(instrument: str) -> tuple[float, float, float]:
            """Return (mid, low, high) from recent candles for outcome checking."""
            cache = self._candle_cache.get(instrument)
            if cache and len(cache) >= self._settings.label_horizon_bars:
                recent = list(cache)[-self._settings.label_horizon_bars:]
                low = min(c.low for c in recent)
                high = max(c.high for c in recent)
                mid = (recent[-1].close)
                return mid, low, high
            # Fallback to current price
            quote = self._broker.get_price(instrument)
            mid = (quote.bid + quote.ask) / 2
            return mid, mid, mid

        resolved = self._outcome_tracker.resolve_outcomes(
            self._cycle_count, _get_price_range
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

        # Run parameter optimization every 10 cycles
        if self._cycle_count % 10 == 0:
            from stockio.strategy.optimizer import maybe_optimize

            optimized = maybe_optimize(self._settings, self._settings.data_dir)
            if optimized:
                cycle_log.info(
                    "params_optimized",
                    instruments=list(optimized.keys()),
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

    def _maybe_heartbeat(self, cycle_log: structlog.BoundLogger) -> None:
        """Log a heartbeat if enough time has passed."""
        import sys

        now = datetime.now(UTC)
        if self._last_heartbeat is not None:
            elapsed = (now - self._last_heartbeat).total_seconds()
            if elapsed < self._settings.heartbeat_seconds:
                return

        # Memory usage (platform-dependent)
        mem_mb = 0.0
        try:
            if sys.platform != "win32":
                import resource

                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            else:
                import os

                import psutil  # type: ignore[import-untyped]

                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass

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

        account = self._last_account
        if not account:
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
    def last_account(self) -> AccountSummary | None:
        return self._last_account

    @property
    def risk(self) -> RiskManager:
        return self._risk
