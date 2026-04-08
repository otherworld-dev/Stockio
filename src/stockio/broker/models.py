"""Data models for the broker layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class Candle:
    """Single OHLCV candle."""

    instrument: str
    granularity: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool = True


@dataclass(frozen=True, slots=True)
class Signal:
    """Trade signal from the ML model."""

    instrument: str
    direction: Direction
    confidence: float
    timestamp: datetime
    features: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Order to be submitted to the broker."""

    instrument: str
    direction: Direction
    units: int
    order_type: OrderType = OrderType.MARKET
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    signal_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Position:
    """An open position at the broker."""

    instrument: str
    direction: Direction
    units: int
    entry_price: float
    unrealized_pnl: float
    trade_id: str
    open_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class AccountSummary:
    """Broker account summary."""

    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    margin_available: float
    open_position_count: int
    currency: str = "GBP"


@dataclass(frozen=True, slots=True)
class PriceQuote:
    """Current bid/ask price for an instrument."""

    instrument: str
    bid: float
    ask: float
    timestamp: datetime
    spread: float = 0.0

    def __post_init__(self) -> None:
        if self.spread == 0.0:
            object.__setattr__(self, "spread", self.ask - self.bid)
