"""Abstract broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from stockio.broker.models import (
    AccountSummary,
    Candle,
    OrderRequest,
    Position,
    PriceQuote,
)


class BrokerBase(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        """Fetch historical candles for an instrument.

        Either `count` (most recent N candles) or `from_time`/`to_time` must be provided.
        """

    @abstractmethod
    def get_price(self, instrument: str) -> PriceQuote:
        """Get current bid/ask price for an instrument."""

    @abstractmethod
    def get_account(self) -> AccountSummary:
        """Get account balance, equity, margin info."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> str:
        """Submit an order. Returns the trade/order ID from the broker."""

    @abstractmethod
    def close_position(self, trade_id: str) -> None:
        """Close a specific position by trade ID."""
