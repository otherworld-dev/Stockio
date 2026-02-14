"""Trade execution module.

Provides a unified interface with two backends:
  - PaperExecutor  — simulated trades using live market prices (default)
  - AlpacaExecutor — real trades via the Alpaca brokerage API

Both executors share the same interface so the rest of the system is
broker-agnostic.  Supports all asset types with per-asset position sizing.
"""

from __future__ import annotations

import datetime as dt
import math
import time
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
    remove_position,
    set_cash,
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

        # Size the position: allocate up to per-asset MAX_POSITION_PCT of portfolio,
        # scaled by signal confidence
        asset_type = config.get_asset_type(ticker)
        risk = config.get_risk_params(asset_type)
        max_spend = cash * (risk["max_position_pct"] / 100.0) * signal.confidence
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
        # Size: use per-asset max position pct, scaled by confidence
        asset_type = config.get_asset_type(ticker)
        risk = config.get_risk_params(asset_type)
        max_exposure = cash * (risk["max_position_pct"] / 100.0) * signal.confidence
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

        asset_type = config.get_asset_type(ticker)
        risk = config.get_risk_params(asset_type)

        if pos.direction == "short":
            return self._check_short_exits(pos, current_price, risk)

        # Long position exits
        if check_stop_loss(ticker, current_price):
            log.warning(
                "STOP-LOSS triggered for %s (cost=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            try:
                return record_sell(
                    ticker, pos.shares, current_price,
                    reason=f"Stop-loss at {risk['stop_loss_pct']}%",
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
                    reason=f"Take-profit at {risk['take_profit_pct']}%",
                )
            except ValueError:
                return None

        return None

    def _check_short_exits(
        self, pos: Position, current_price: float, risk: dict | None = None,
    ) -> TradeRecord | None:
        """Check stop-loss, take-profit, and the critical safety guard for shorts."""
        ticker = pos.ticker
        if risk is None:
            asset_type = config.get_asset_type(ticker)
            risk = config.get_risk_params(asset_type)

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
                    reason=f"Short stop-loss at {risk['stop_loss_pct']}%",
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
                    reason=f"Short take-profit at {risk['take_profit_pct']}%",
                )
            except ValueError:
                return None

        return None


# ---------------------------------------------------------------------------
# Alpaca (live) executor — real trades via Alpaca brokerage
# ---------------------------------------------------------------------------

# Maximum seconds to wait for an order to fill before giving up.
_ORDER_FILL_TIMEOUT = 30


class AlpacaExecutor(Executor):
    """Execute real trades via the Alpaca brokerage API.

    Uses the ``alpaca-py`` SDK (the official, actively maintained SDK).
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment.

    Every trade submitted to Alpaca is also mirrored into the local
    SQLite portfolio DB so the dashboard and risk checks stay in sync.
    """

    def __init__(self) -> None:
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError(
                "Alpaca API credentials not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        from alpaca.trading.client import TradingClient

        is_paper = "paper" in config.ALPACA_BASE_URL
        self.client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=is_paper,
        )
        # Verify credentials by fetching account info
        acct = self.client.get_account()
        log.info(
            "Alpaca executor initialised (paper=%s, equity=$%s, buying_power=$%s)",
            is_paper, acct.equity, acct.buying_power,
        )

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _submit_market_order(
        self, ticker: str, qty: float, side: str
    ) -> object | None:
        """Submit a market order and wait for it to fill.

        *side* is ``"buy"`` or ``"sell"``.  Returns the filled Order
        object or ``None`` if the order failed / timed out.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=round(qty, 4),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        log.info("ALPACA ORDER: %s %s x%.4f (market)", side.upper(), ticker, qty)

        try:
            order = self.client.submit_order(req)
        except Exception as exc:
            log.error("Alpaca order failed for %s: %s", ticker, exc)
            return None

        # Poll until filled or timeout
        filled = self._wait_for_fill(order.id)
        if filled is None:
            log.warning(
                "Order %s for %s did not fill within %ds — cancelling",
                order.id, ticker, _ORDER_FILL_TIMEOUT,
            )
            try:
                self.client.cancel_order_by_id(order.id)
            except Exception:
                pass
            return None

        return filled

    def _wait_for_fill(self, order_id: str) -> object | None:
        """Poll Alpaca until the order fills or times out."""
        deadline = time.monotonic() + _ORDER_FILL_TIMEOUT
        while time.monotonic() < deadline:
            order = self.client.get_order_by_id(order_id)
            if order.status == "filled":
                return order
            if order.status in ("canceled", "expired", "rejected"):
                log.warning("Order %s ended with status: %s", order_id, order.status)
                return None
            time.sleep(1)
        return None

    # ------------------------------------------------------------------
    # Execute trades
    # ------------------------------------------------------------------

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

        max_spend = cash * (config.MAX_POSITION_PCT / 100.0) * signal.confidence
        max_spend = min(max_spend, cash)
        if max_spend < 1.0:
            return None
        if not check_position_limit(ticker, max_spend):
            log.info("Position limit reached for %s", ticker)
            return None

        shares = math.floor((max_spend / price) * 10000) / 10000
        if shares <= 0:
            return None

        filled = self._submit_market_order(ticker, shares, "buy")
        if filled is None:
            return None

        fill_price = float(filled.filled_avg_price)
        fill_qty = float(filled.filled_qty)
        reason = "; ".join(signal.reasons)

        try:
            return record_buy(ticker, fill_qty, fill_price, reason=reason)
        except ValueError as exc:
            log.warning("Local record_buy failed for %s: %s", ticker, exc)
            return None

    def _sell(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        pos = get_position(ticker)
        if pos is None or pos.direction != "long":
            log.info("No long position in %s to sell", ticker)
            return None

        filled = self._submit_market_order(ticker, pos.shares, "sell")
        if filled is None:
            return None

        fill_price = float(filled.filled_avg_price)
        fill_qty = float(filled.filled_qty)
        reason = "; ".join(signal.reasons)

        try:
            return record_sell(ticker, fill_qty, fill_price, reason=reason)
        except ValueError as exc:
            log.warning("Local record_sell failed for %s: %s", ticker, exc)
            return None

    def _short(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        """Open a short position via Alpaca.

        On Alpaca, shorting is done by submitting a ``sell`` order when
        you have no long position.  The broker handles the share borrow.
        """
        if not config.SHORT_SELLING_ENABLED:
            return None

        # Run the same safety checks as paper mode
        cash = get_cash()
        positions = get_positions()
        long_value = sum(p.shares * price for p in positions if p.direction == "long")
        short_liability = sum(p.shares * price for p in positions if p.direction == "short")
        new_short_size = cash * (config.MAX_SHORT_POSITION_PCT / 100.0) * signal.confidence
        worst_case_loss = new_short_size + short_liability
        current_value = cash + long_value - short_liability
        if current_value - worst_case_loss <= 0:
            log.info("Short on %s rejected — would risk going negative", ticker)
            return None

        max_exposure = cash * (config.MAX_SHORT_POSITION_PCT / 100.0) * signal.confidence
        if max_exposure < 1.0:
            return None
        if not check_short_limit(ticker, max_exposure):
            log.info("Short limit reached for %s", ticker)
            return None

        shares = math.floor((max_exposure / price) * 10000) / 10000
        if shares <= 0:
            return None

        # Alpaca: sell without holding = short
        filled = self._submit_market_order(ticker, shares, "sell")
        if filled is None:
            return None

        fill_price = float(filled.filled_avg_price)
        fill_qty = float(filled.filled_qty)
        reason = "; ".join(signal.reasons)

        try:
            return record_short(ticker, fill_qty, fill_price, reason=reason)
        except ValueError as exc:
            log.warning("Local record_short failed for %s: %s", ticker, exc)
            return None

    def _cover(
        self, ticker: str, price: float, signal: TradeSignal
    ) -> TradeRecord | None:
        """Close a short position — buy back the borrowed shares."""
        pos = get_position(ticker)
        if pos is None or pos.direction != "short":
            log.info("No short position in %s to cover", ticker)
            return None

        # Alpaca: buy to cover
        filled = self._submit_market_order(ticker, pos.shares, "buy")
        if filled is None:
            return None

        fill_price = float(filled.filled_avg_price)
        fill_qty = float(filled.filled_qty)
        reason = "; ".join(signal.reasons)

        try:
            return record_cover(ticker, fill_qty, fill_price, reason=reason)
        except ValueError as exc:
            log.warning("Local record_cover failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Exit checks (stop-loss, take-profit, safety cover)
    # ------------------------------------------------------------------

    def check_exits(self, ticker: str, current_price: float) -> TradeRecord | None:
        pos = get_position(ticker)
        if pos is None:
            return None

        if pos.direction == "short":
            return self._check_short_exits(pos, current_price)

        # Long position exits
        if check_stop_loss(ticker, current_price):
            log.warning(
                "LIVE STOP-LOSS for %s (cost=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            filled = self._submit_market_order(ticker, pos.shares, "sell")
            if filled is None:
                return None
            fill_price = float(filled.filled_avg_price)
            fill_qty = float(filled.filled_qty)
            try:
                return record_sell(
                    ticker, fill_qty, fill_price,
                    reason=f"Stop-loss at {config.STOP_LOSS_PCT}%",
                )
            except ValueError:
                return None

        if check_take_profit(ticker, current_price):
            log.info(
                "LIVE TAKE-PROFIT for %s (cost=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            filled = self._submit_market_order(ticker, pos.shares, "sell")
            if filled is None:
                return None
            fill_price = float(filled.filled_avg_price)
            fill_qty = float(filled.filled_qty)
            try:
                return record_sell(
                    ticker, fill_qty, fill_price,
                    reason=f"Take-profit at {config.TAKE_PROFIT_PCT}%",
                )
            except ValueError:
                return None

        return None

    def _check_short_exits(
        self, pos: Position, current_price: float
    ) -> TradeRecord | None:
        """Check stop-loss, take-profit, and critical safety guard for live shorts."""
        ticker = pos.ticker

        # SAFETY GUARD: auto-cover if portfolio value is dangerously low
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
        if total_value < initial * 0.10:
            log.warning(
                "LIVE SAFETY COVER for %s — portfolio value £%.2f critically low",
                ticker, total_value,
            )
            filled = self._submit_market_order(ticker, pos.shares, "buy")
            if filled is None:
                return None
            fill_price = float(filled.filled_avg_price)
            fill_qty = float(filled.filled_qty)
            try:
                return record_cover(
                    ticker, fill_qty, fill_price,
                    reason="Safety cover — portfolio value critically low",
                )
            except ValueError:
                return None

        if check_stop_loss(ticker, current_price):
            log.warning(
                "LIVE SHORT STOP-LOSS for %s (entry=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            filled = self._submit_market_order(ticker, pos.shares, "buy")
            if filled is None:
                return None
            fill_price = float(filled.filled_avg_price)
            fill_qty = float(filled.filled_qty)
            try:
                return record_cover(
                    ticker, fill_qty, fill_price,
                    reason=f"Short stop-loss at {config.SHORT_STOP_LOSS_PCT}%",
                )
            except ValueError:
                return None

        if check_take_profit(ticker, current_price):
            log.info(
                "LIVE SHORT TAKE-PROFIT for %s (entry=%.2f, price=%.2f)",
                ticker, pos.avg_cost, current_price,
            )
            filled = self._submit_market_order(ticker, pos.shares, "buy")
            if filled is None:
                return None
            fill_price = float(filled.filled_avg_price)
            fill_qty = float(filled.filled_qty)
            try:
                return record_cover(
                    ticker, fill_qty, fill_price,
                    reason=f"Short take-profit at {config.SHORT_TAKE_PROFIT_PCT}%",
                )
            except ValueError:
                return None

        return None

    # ------------------------------------------------------------------
    # Account sync — reconcile local DB with Alpaca's actual state
    # ------------------------------------------------------------------

    def sync_account(self) -> None:
        """Sync local portfolio DB with Alpaca's actual account state.

        Call this at bot startup (or periodically) to ensure our local
        records match the broker's reality — e.g. after a manual trade
        on Alpaca's dashboard, or if the bot restarted mid-cycle.
        """
        acct = self.client.get_account()
        broker_cash = float(acct.cash)
        local_cash = get_cash()
        if abs(broker_cash - local_cash) > 0.01:
            log.info(
                "Syncing cash: local £%.2f → broker $%.2f",
                local_cash, broker_cash,
            )
            set_cash(broker_cash)

        alpaca_positions = self.client.get_all_positions()
        alpaca_tickers = set()
        log.info("Alpaca has %d open positions", len(alpaca_positions))
        for ap in alpaca_positions:
            ticker = ap.symbol
            alpaca_tickers.add(ticker)
            qty = abs(float(ap.qty))
            avg_entry = float(ap.avg_entry_price)
            # Alpaca: negative qty = short position
            direction = "short" if float(ap.qty) < 0 else "long"

            local_pos = get_position(ticker)
            if local_pos is None:
                log.info(
                    "Syncing missing position: %s %s x%.4f @ $%.2f",
                    direction.upper(), ticker, qty, avg_entry,
                )
                if direction == "long":
                    record_buy(ticker, qty, avg_entry, reason="Synced from Alpaca")
                else:
                    record_short(ticker, qty, avg_entry, reason="Synced from Alpaca")

        # Remove local positions that no longer exist on Alpaca
        # (e.g. stale paper-trading positions from before switching to live)
        for pos in get_positions():
            if pos.ticker not in alpaca_tickers:
                log.info(
                    "Removing stale local position: %s %s x%.4f (not on Alpaca)",
                    pos.direction.upper(), pos.ticker, pos.shares,
                )
                remove_position(pos.ticker)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_executor() -> Executor:
    """Return the appropriate executor based on config."""
    if config.MODE == "live":
        return AlpacaExecutor()
    return PaperExecutor()
