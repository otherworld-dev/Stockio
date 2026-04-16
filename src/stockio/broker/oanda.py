"""Concrete OANDA v20 broker implementation."""

from __future__ import annotations

import threading
from datetime import datetime

import oandapyV20
import oandapyV20.endpoints.accounts as ep_accounts
import oandapyV20.endpoints.instruments as ep_instruments
import oandapyV20.endpoints.orders as ep_orders
import oandapyV20.endpoints.pricing as ep_pricing
import oandapyV20.endpoints.trades as ep_trades
import structlog
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    StopLossDetails,
    TakeProfitDetails,
)
from oandapyV20.exceptions import V20Error
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stockio.broker.base import BrokerBase
from stockio.broker.models import (
    AccountSummary,
    Candle,
    Direction,
    OrderRequest,
    Position,
    PriceQuote,
)
from stockio.config import Settings

log = structlog.get_logger()


class OandaBroker(BrokerBase):
    """OANDA v20 REST API broker."""

    def __init__(self, settings: Settings) -> None:
        self._account_id = settings.oanda_account_id
        self._client = oandapyV20.API(
            access_token=settings.oanda_api_token,
            environment=settings.oanda_environment,
        )
        self._lock = threading.Lock()

    def _request(self, req):
        """Execute an oandapyV20 request under a lock (thread-safe)."""
        with self._lock:
            return self._client.request(req)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(V20Error),
        reraise=True,
    )
    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: int | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        params: dict = {"granularity": granularity}
        if count is not None:
            params["count"] = count
        if from_time is not None:
            params["from"] = from_time.isoformat()
        if to_time is not None:
            params["to"] = to_time.isoformat()

        req = ep_instruments.InstrumentsCandles(instrument=instrument, params=params)
        resp = self._request(req)

        candles: list[Candle] = []
        for c in resp.get("candles", []):
            if not c.get("complete", False):
                continue
            mid = c["mid"]
            candles.append(
                Candle(
                    instrument=instrument,
                    granularity=granularity,
                    timestamp=datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=int(c.get("volume", 0)),
                    complete=True,
                )
            )
        return candles

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(V20Error),
        reraise=True,
    )
    def get_price(self, instrument: str) -> PriceQuote:
        req = ep_pricing.PricingInfo(
            accountID=self._account_id,
            params={"instruments": instrument},
        )
        resp = self._request(req)
        price_data = resp["prices"][0]
        return PriceQuote(
            instrument=instrument,
            bid=float(price_data["bids"][0]["price"]),
            ask=float(price_data["asks"][0]["price"]),
            timestamp=datetime.fromisoformat(
                price_data["time"].replace("Z", "+00:00")
            ),
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(V20Error),
        reraise=True,
    )
    def get_account(self) -> AccountSummary:
        req = ep_accounts.AccountSummary(accountID=self._account_id)
        resp = self._request(req)
        acct = resp["account"]
        return AccountSummary(
            balance=float(acct["balance"]),
            equity=float(acct["NAV"]),
            unrealized_pnl=float(acct["unrealizedPL"]),
            margin_used=float(acct["marginUsed"]),
            margin_available=float(acct["marginAvailable"]),
            open_position_count=int(acct["openPositionCount"]),
            currency=acct.get("currency", "GBP"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(V20Error),
        reraise=True,
    )
    def get_positions(self) -> list[Position]:
        """Get all open trades (individual trade-level, not aggregated positions)."""
        req = ep_trades.TradesList(
            accountID=self._account_id,
            params={"state": "OPEN"},
        )
        resp = self._request(req)

        positions: list[Position] = []
        for trade in resp.get("trades", []):
            units = int(trade["currentUnits"])
            direction = Direction.BUY if units > 0 else Direction.SELL
            positions.append(
                Position(
                    instrument=trade["instrument"],
                    direction=direction,
                    units=abs(units),
                    entry_price=float(trade.get("price", 0)),
                    unrealized_pnl=float(trade.get("unrealizedPL", 0)),
                    trade_id=trade["id"],
                )
            )
        return positions

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    # No retry on order submission — it's not idempotent.
    # A timeout after the order was filled would cause a duplicate position.
    def submit_order(self, order: OrderRequest) -> str:
        units = order.units if order.direction == Direction.BUY else -order.units

        kwargs: dict = {
            "instrument": order.instrument,
            "units": units,
        }
        if order.take_profit_price is not None:
            kwargs["takeProfitOnFill"] = TakeProfitDetails(
                price=order.take_profit_price
            ).data
        if order.stop_loss_price is not None:
            kwargs["stopLossOnFill"] = StopLossDetails(
                price=order.stop_loss_price
            ).data

        mkt_order = MarketOrderRequest(**kwargs)
        req = ep_orders.OrderCreate(
            accountID=self._account_id, data=mkt_order.data
        )
        resp = self._request(req)

        trade_id = (
            resp.get("orderFillTransaction", {})
            .get("tradeOpened", {})
            .get("tradeID", "unknown")
        )
        log.info(
            "order_submitted",
            instrument=order.instrument,
            direction=order.direction.value,
            units=order.units,
            trade_id=trade_id,
        )
        return trade_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(V20Error),
        reraise=True,
    )
    def close_position(self, trade_id: str) -> None:
        req = ep_trades.TradeClose(
            accountID=self._account_id,
            tradeID=trade_id,
            data={"units": "ALL"},
        )
        self._request(req)
        log.info("position_closed", trade_id=trade_id)

    def modify_trade_sl(self, trade_id: str, stop_loss_price: float) -> None:
        """Update the stop-loss on an existing trade (for trailing stops)."""
        data = {"stopLoss": {"price": str(stop_loss_price)}}
        req = ep_trades.TradeCRCDO(
            accountID=self._account_id,
            tradeID=trade_id,
            data=data,
        )
        self._request(req)
        log.info("sl_modified", trade_id=trade_id, new_sl=stop_loss_price)

    def get_closed_trade_details(self, trade_id: str) -> dict | None:
        """Get details of a closed trade from OANDA (actual fill price + P&L)."""
        try:
            req = ep_trades.TradeDetails(
                accountID=self._account_id, tradeID=trade_id
            )
            resp = self._request(req)
            trade = resp.get("trade", {})
            return {
                "close_price": float(trade.get("averageClosePrice", 0)),
                "realized_pnl": float(trade.get("realizedPL", 0)),
                "close_time": trade.get("closeTime", ""),
                "state": trade.get("state", ""),
            }
        except Exception:
            return None
