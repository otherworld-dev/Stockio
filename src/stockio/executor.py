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
    TradeRecord,
    check_position_limit,
    check_stop_loss,
    check_take_profit,
    get_cash,
    get_position,
    record_buy,
    record_sell,
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
        if pos is None:
            log.info("No position in %s to sell", ticker)
            return None

        # Sell entire position on SELL signal
        reason = "; ".join(signal.reasons)
        try:
            return record_sell(ticker, pos.shares, price, reason=reason)
        except ValueError as exc:
            log.warning("Sell failed for %s: %s", ticker, exc)
            return None

    def check_exits(self, ticker: str, current_price: float) -> TradeRecord | None:
        pos = get_position(ticker)
        if pos is None:
            return None

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
