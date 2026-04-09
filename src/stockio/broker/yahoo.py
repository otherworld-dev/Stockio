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
        # Calculate holdings value from positions
        unrealized = 0.0
        for pos in self._positions:
            try:
                quote = self.get_price(pos.instrument)
                mid = (quote.bid + quote.ask) / 2
                if pos.direction.value == "BUY":
                    unrealized += (mid - pos.entry_price) * pos.units
                else:
                    unrealized += (pos.entry_price - mid) * pos.units
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
        """Simulate a paper trade."""
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter}"

        try:
            quote = self.get_price(order.instrument)
        except Exception as exc:
            raise ValueError(f"Cannot get price for {order.instrument}") from exc

        fill_price = quote.ask if order.direction.value == "BUY" else quote.bid

        cost = fill_price * order.units
        self._cash -= cost if order.direction.value == "BUY" else -cost

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
            if pos.direction.value == "BUY":
                pnl = (mid - pos.entry_price) * pos.units
            else:
                pnl = (pos.entry_price - mid) * pos.units
            self._cash += pos.entry_price * pos.units + pnl
        except Exception:
            self._cash += pos.entry_price * pos.units

        self._positions = [p for p in self._positions if p.trade_id != trade_id]
        log.info("paper_close", trade_id=trade_id)
