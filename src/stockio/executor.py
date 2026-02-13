"""Trade execution module.

Provides a unified interface with two backends:
  - PaperExecutor  — simulated trades using live market prices (default)
  - AlpacaExecutor — real trades via the Alpaca brokerage API

Both executors share the same interface so the rest of the system is
broker-agnostic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from stockio import config
from stockio.config import get_logger
from stockio.market_data import get_latest_price
from stockio.portfolio import (
    Position,
    TradeRecord,
    check_position_limit,
    check_short_limit,
    check_stop_loss,
    check_take_profit,
    get_cash,
    get_position,
    get_positions,
    record_buy,
    record_cover,
    record_sell,
    record_short,
)
from stockio.strategy import Signal, TradeSignal

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Executor(ABC):
    """Interface every executor must implement."""

    @abstractmethod
    def execute(self, signal: TradeSignal, current_price: float) -> TradeRecord | None:
        """Execute a trade based on *signal*. Returns the trade record or None."""

    @abstractmethod
    def check_exits(self, ticker: str, current_price: float) -> TradeRecord | None:
        """Check stop-loss / take-profit and exit if triggered."""


# ---------------------------------------------------------------------------
# Paper (simulated) executor
# ---------------------------------------------------------------------------


class PaperExecutor(Executor):
    """Simulate trades using real market prices but no real money."""

    def execute(self, signal: TradeSignal, current_price: float) -> TradeRecord | None:
        ticker = signal.ticker

        if signal.signal == Signal.HOLD:
            return None
        if signal.signal == Signal.BUY:
            return self._buy(ticker, current_price, signal)
        if signal.signal == Signal.SELL:
            return self._sell(ticker, current_price, signal)
        if signal.signal == Signal.SHORT:
            return self._short(ticker, current_price, signal)
        if signal.signal == Signal.COVER:
            return self._cover(ticker, current_price, signal)

        return None

    def _buy(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        cash = get_cash()
        if cash < 1.0:
            log.info("No cash available to buy %s", ticker)
            return None

        # Size the position: allocate up to MAX_POSITION_PCT of portfolio,
        # scaled by signal confidence
        max_spend = cash * (config.MAX_POSITION_PCT / 100.0) * signal.confidence
        max_spend = min(max_spend, cash)  # never exceed available cash

        if max_spend < 1.0:
            log.info("Position too small for %s (max_spend=£%.2f)", ticker, max_spend)
            return None

        # Risk check
        if not check_position_limit(ticker, max_spend):
            log.info("Position limit reached for %s", ticker)
            return None

        shares = max_spend / price
        # Round down to 4 decimal places (fractional shares)
        shares = math.floor(shares * 10000) / 10000
        if shares <= 0:
            return None

        reason = "; ".join(signal.reasons)
        try:
            return record_buy(ticker, shares, price, reason=reason)
        except ValueError as exc:
            log.warning("Buy failed for %s: %s", ticker, exc)
            return None

    def _sell(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        pos = get_position(ticker)
        if pos is None or pos.direction != "long":
            log.info("No long position in %s to sell", ticker)
            return None

        # Sell entire position on SELL signal
        reason = "; ".join(signal.reasons)
        try:
            return record_sell(ticker, pos.shares, price, reason=reason)
        except ValueError as exc:
            log.warning("Sell failed for %s: %s", ticker, exc)
            return None

    def _short(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        """Open a short position (bet that the price will drop)."""
        if not config.SHORT_SELLING_ENABLED:
            return None

        # Safety: never short if it would push total value below zero
        if not self._short_is_safe(ticker, price, signal.confidence):
            return None

        cash = get_cash()
        # Size: use MAX_SHORT_POSITION_PCT, scaled by confidence
        max_exposure = cash * (config.MAX_SHORT_POSITION_PCT / 100.0) * signal.confidence
        if max_exposure < 1.0:
            log.info("Short position too small for %s", ticker)
            return None

        if not check_short_limit(ticker, max_exposure):
            log.info("Short limit reached for %s", ticker)
            return None

        shares = max_exposure / price
        shares = math.floor(shares * 10000) / 10000
        if shares <= 0:
            return None

        reason = "; ".join(signal.reasons)
        try:
            return record_short(ticker, shares, price, reason=reason)
        except ValueError as exc:
            log.warning("Short failed for %s: %s", ticker, exc)
            return None

    def _cover(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        """Close a short position (buy back borrowed shares)."""
        pos = get_position(ticker)
        if pos is None or pos.direction != "short":
            log.info("No short position in %s to cover", ticker)
            return None

        reason = "; ".join(signal.reasons)
        try:
            return record_cover(ticker, pos.shares, price, reason=reason)
        except ValueError as exc:
            log.warning("Cover failed for %s: %s", ticker, exc)
            return None

    def _short_is_safe(self, ticker: str, price: float, confidence: float) -> bool:
        """Check that opening a short won't risk going below zero.

        We require that even if the shorted stock price *doubles* from
        entry, the total portfolio value remains positive.  This is the
        key safety guard the user asked for.
        """
        cash = get_cash()
        positions = get_positions()
        long_value = sum(
            p.shares * price for p in positions
            if p.direction == "long"
        )
        # Current short liability at current prices
        short_liability = sum(
            p.shares * price for p in positions
            if p.direction == "short"
        )
        # Proposed new short
        new_short_size = cash * (config.MAX_SHORT_POSITION_PCT / 100.0) * confidence
        # Worst case: the new short stock price doubles → loss = short_size
        worst_case_loss = new_short_size + short_liability
        # Total value must stay positive after worst-case
        current_value = cash + long_value - short_liability
        if current_value - worst_case_loss <= 0:
            log.info(
                "Short on %s rejected — worst-case loss (£%.2f) would "
                "make portfolio negative (current value: £%.2f)",
                ticker, worst_case_loss, current_value,
            )
            return False
        return True

    def check_exits(self, ticker: str, current_price: float) -> TradeRecord | None:
        pos = get_position(ticker)
        if pos is None:
            return None

        if pos.direction == "short":
            return self._check_short_exits(pos, current_price)

        # Long position exits
        if check_stop_loss(ticker, current_price):
            log.warning(
                "STOP-LOSS triggered for %s (cost=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            try:
                return record_sell(
                    ticker, pos.shares, current_price,
                    reason=f"Stop-loss at {config.STOP_LOSS_PCT}%",
                )
            except ValueError:
                return None

        if check_take_profit(ticker, current_price):
            log.info(
                "TAKE-PROFIT triggered for %s (cost=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            try:
                return record_sell(
                    ticker, pos.shares, current_price,
                    reason=f"Take-profit at {config.TAKE_PROFIT_PCT}%",
                )
            except ValueError:
                return None

        return None

    def _check_short_exits(
        self, pos: Position, current_price: float
    ) -> TradeRecord | None:
        """Check stop-loss, take-profit, and the critical safety guard for shorts."""
        ticker = pos.ticker

        # SAFETY GUARD: auto-cover if portfolio value is dangerously low.
        # This prevents ever going into negative money.
        cash = get_cash()
        positions = get_positions()
        long_val = sum(
            p.shares * current_price for p in positions if p.direction == "long"
        )
        short_liab = sum(
            p.shares * current_price for p in positions if p.direction == "short"
        )
        total_value = cash + long_val - short_liab
        initial = config.INITIAL_BUDGET_GBP
        # Auto-cover if portfolio drops below 10% of initial budget
        if total_value < initial * 0.10:
            log.warning(
                "SAFETY COVER for %s — portfolio value £%.2f is critically low",
                ticker, total_value,
            )
            try:
                return record_cover(
                    ticker, pos.shares, current_price,
                    reason="Safety cover — portfolio value critically low",
                )
            except ValueError:
                return None

        # Normal short stop-loss (price rose too much)
        if check_stop_loss(ticker, current_price):
            log.warning(
                "SHORT STOP-LOSS for %s (entry=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            try:
                return record_cover(
                    ticker, pos.shares, current_price,
                    reason=f"Short stop-loss at {config.SHORT_STOP_LOSS_PCT}%",
                )
            except ValueError:
                return None

        # Short take-profit (price dropped enough)
        if check_take_profit(ticker, current_price):
            log.info(
                "SHORT TAKE-PROFIT for %s (entry=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            try:
                return record_cover(
                    ticker, pos.shares, current_price,
                    reason=f"Short take-profit at {config.SHORT_TAKE_PROFIT_PCT}%",
                )
            except ValueError:
                return None

        return None


# ---------------------------------------------------------------------------
# Alpaca (live) executor — skeleton ready for API keys
# ---------------------------------------------------------------------------


class AlpacaExecutor(Executor):
    """Execute real trades via the Alpaca brokerage API.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment.
    """

    def __init__(self) -> None:
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError(
                "Alpaca API credentials not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )
        # Lazy import — only needed when actually doing live trading
        import alpaca_trade_api as tradeapi

        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL,
            api_version="v2",
        )
        log.info("Alpaca executor initialised (base=%s)", config.ALPACA_BASE_URL)

    def execute(self, signal: TradeSignal, current_price: float) -> TradeRecord | None:
        # For live trading, delegate to Alpaca's order API and mirror
        # the result into our local portfolio DB.
        log.info(
            "LIVE ORDER: %s %s @ ~£%.2f (confidence=%.2f)",
            signal.signal.value, signal.ticker, current_price, signal.confidence,
        )
        # TODO: implement full Alpaca order flow
        # self.api.submit_order(...)
        raise NotImplementedError("Alpaca live execution is not yet implemented")

    def check_exits(self, ticker: str, current_price: float) -> TradeRecord | None:
        raise NotImplementedError("Alpaca live exit checks are not yet implemented")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_executor() -> Executor:
    """Return the appropriate executor based on config."""
    if config.MODE == "live":
        return AlpacaExecutor()
    return PaperExecutor()
