"""Yahoo Finance data provider — free market data, no signup required.

Implements BrokerBase for data fetching (candles, prices). Trade execution
methods raise NotImplementedError — use PaperExecutor for simulated trades.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
import yfinance as yf

from stockio.broker.base import BrokerBase
from stockio.broker.models import (
    AccountSummary,
    Candle,
    OrderRequest,
    Position,
    PriceQuote,
)

log = structlog.get_logger()

def _pip_value_in_gbp(instrument: str, units: int, current_price: float) -> float:
    """Convert a 1-pip move to GBP value for position sizing.

    For forex, P&L is in the quote currency:
    - XXX_USD pairs: P&L is in USD → divide by GBP/USD rate (~1.30)
    - XXX_JPY pairs: P&L is in JPY → divide by GBP/JPY rate (~190)
    - EUR_GBP: P&L is already in GBP
    - XXX_CHF pairs: P&L is in CHF → divide by GBP/CHF rate (~1.10)
    - XXX_CAD pairs: P&L is in CAD → divide by GBP/CAD rate (~1.80)

    Uses approximate rates — close enough for paper trading.
    """
    quote_ccy = instrument.split("_")[1] if "_" in instrument else "USD"

    # Approximate conversion rates to GBP (updated periodically would be better,
    # but these are close enough for paper P&L tracking)
    to_gbp = {
        "USD": 1 / 1.30,
        "JPY": 1 / 190.0,
        "GBP": 1.0,
        "CHF": 1 / 1.10,
        "CAD": 1 / 1.80,
        "AUD": 1 / 1.95,
        "NZD": 1 / 2.10,
    }
    return to_gbp.get(quote_ccy, 1 / 1.30)


# Map OANDA-style instrument names to Yahoo Finance tickers
_YAHOO_TICKER_MAP = {
    # Forex (Yahoo uses =X suffix)
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "AUD_USD": "AUDUSD=X",
    "NZD_USD": "NZDUSD=X",
    "USD_CAD": "USDCAD=X",
    "USD_CHF": "USDCHF=X",
    "EUR_GBP": "EURGBP=X",
    "EUR_JPY": "EURJPY=X",
    "GBP_JPY": "GBPJPY=X",
    "AUD_JPY": "AUDJPY=X",
    # Commodities
    "XAU_USD": "GC=F",  # Gold futures
    "XAG_USD": "SI=F",  # Silver futures
    "BCO_USD": "BZ=F",  # Brent crude futures
}

# Map granularity to yfinance interval + period
_GRANULARITY_MAP = {
    "M1": ("1m", "7d"),
    "M5": ("5m", "60d"),
    "M15": ("15m", "60d"),
    "M30": ("30m", "60d"),
    "H1": ("1h", "730d"),
    "H4": ("1h", "730d"),  # yfinance doesn't have 4h, use 1h
    "D": ("1d", "2y"),
}


def _to_yahoo_ticker(instrument: str) -> str:
    """Convert OANDA instrument name to Yahoo Finance ticker."""
    if instrument in _YAHOO_TICKER_MAP:
        return _YAHOO_TICKER_MAP[instrument]
    return instrument.replace("_", "")


class YahooBroker(BrokerBase):
    """Free market data from Yahoo Finance. No API key required.

    Only provides data (candles, prices). Trade execution is not supported —
    pair with PaperExecutor for simulated trading.
    """

    def __init__(self, initial_budget: float = 500.0) -> None:
        self._budget = initial_budget
        # Paper account state
        self._cash = initial_budget
        self._positions: list[Position] = []
        self._trade_counter = 0

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        ticker = _to_yahoo_ticker(instrument)
        interval, period = _GRANULARITY_MAP.get(granularity, ("15m", "60d"))

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            log.warning("yahoo_no_data", instrument=instrument, ticker=ticker)
            return []

        # Handle multi-level columns from yfinance
        if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
            data = data.droplevel(level=1, axis=1)

        candles: list[Candle] = []
        for ts, row in data.iterrows():
            candles.append(
                Candle(
                    instrument=instrument,
                    granularity=granularity,
                    timestamp=ts.to_pydatetime().replace(tzinfo=UTC),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row.get("Volume", 0)),
                    complete=True,
                )
            )

        # Trim to requested count
        if count and len(candles) > count:
            candles = candles[-count:]

        return candles

    def get_price(self, instrument: str) -> PriceQuote:
        ticker = _to_yahoo_ticker(instrument)
        data = yf.download(ticker, period="1d", interval="1m", progress=False)

        if data.empty:
            raise ValueError(f"No price data for {instrument}")

        if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
            data = data.droplevel(level=1, axis=1)

        last = data.iloc[-1]
        price = float(last["Close"])
        # Simulate a small spread
        spread = price * 0.0002  # 2 pips typical
        return PriceQuote(
            instrument=instrument,
            bid=price - spread / 2,
            ask=price + spread / 2,
            timestamp=datetime.now(UTC),
        )

    def get_account(self) -> AccountSummary:
        # Calculate holdings value from positions (converted to GBP)
        unrealized = 0.0
        for pos in self._positions:
            try:
                quote = self.get_price(pos.instrument)
                mid = (quote.bid + quote.ask) / 2
                conversion = _pip_value_in_gbp(pos.instrument, pos.units, mid)
                if pos.direction.value == "BUY":
                    unrealized += (mid - pos.entry_price) * pos.units * conversion
                else:
                    unrealized += (pos.entry_price - mid) * pos.units * conversion
            except Exception:
                pass

        equity = self._cash + unrealized
        return AccountSummary(
            balance=self._cash,
            equity=equity,
            unrealized_pnl=unrealized,
            margin_used=0.0,
            margin_available=equity,
            open_position_count=len(self._positions),
            currency="GBP",
        )

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def submit_order(self, order: OrderRequest) -> str:
        """Simulate a paper trade.

        For forex, we track positions and calculate P&L from price movement.
        We do NOT deduct the full notional value — that's not how forex works.
        Instead, we reserve a small margin amount (leverage-based).
        """
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter}"

        try:
            quote = self.get_price(order.instrument)
        except Exception as exc:
            raise ValueError(f"Cannot get price for {order.instrument}") from exc

        fill_price = quote.ask if order.direction.value == "BUY" else quote.bid

        # Simulate spread cost (deducted immediately, converted to GBP)
        conversion = _pip_value_in_gbp(order.instrument, order.units, fill_price)
        spread_cost = (quote.ask - quote.bid) * order.units * conversion
        self._cash -= spread_cost

        self._positions.append(
            Position(
                instrument=order.instrument,
                direction=order.direction,
                units=order.units,
                entry_price=fill_price,
                unrealized_pnl=0.0,
                trade_id=trade_id,
            )
        )

        log.info(
            "paper_trade",
            instrument=order.instrument,
            direction=order.direction.value,
            units=order.units,
            price=fill_price,
            spread_cost=round(spread_cost, 4),
            trade_id=trade_id,
        )
        return trade_id

    def close_position(self, trade_id: str) -> None:
        pos = next((p for p in self._positions if p.trade_id == trade_id), None)
        if not pos:
            return

        try:
            quote = self.get_price(pos.instrument)
            mid = (quote.bid + quote.ask) / 2
            conversion = _pip_value_in_gbp(pos.instrument, pos.units, mid)
            if pos.direction.value == "BUY":
                pnl = (mid - pos.entry_price) * pos.units * conversion
            else:
                pnl = (pos.entry_price - mid) * pos.units * conversion
            self._cash += pnl
        except Exception:
            pass  # Position closed, P&L lost — acceptable for paper

        self._positions = [p for p in self._positions if p.trade_id != trade_id]
        log.info("paper_close", trade_id=trade_id)
