"""Telegram notification — fire-and-forget alerts for trades, errors, summaries."""

from __future__ import annotations

import httpx
import structlog

from stockio.broker.models import OrderRequest
from stockio.config import Settings

log = structlog.get_logger()


class TelegramNotifier:
    """Sends messages via Telegram Bot API. No-op if credentials are missing."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        if self._enabled:
            log.info("telegram_notifier_enabled")

    def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            httpx.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception:
            log.exception("telegram_send_failed")

    def notify_trade(self, order: OrderRequest, trade_id: str) -> None:
        conf = f"{order.signal_confidence:.1%}" if order.signal_confidence is not None else "N/A"
        self._send(
            f"*Trade Executed*\n"
            f"Instrument: `{order.instrument}`\n"
            f"Direction: {order.direction.value}\n"
            f"Units: {order.units}\n"
            f"SL: {order.stop_loss_price}\n"
            f"TP: {order.take_profit_price}\n"
            f"Confidence: {conf}\n"
            f"Trade ID: `{trade_id}`"
        )

    def notify_error(self, error: str) -> None:
        self._send(f"*Error*\n{error}")

    def notify_risk_halt(self, reason: str) -> None:
        self._send(f"*TRADING HALTED*\n{reason}")

    def notify_daily_summary(self, stats: dict) -> None:
        lines = ["*Daily Summary*"]
        for key, val in stats.items():
            if isinstance(val, float):
                lines.append(f"{key}: {val:.4f}")
            else:
                lines.append(f"{key}: {val}")
        self._send("\n".join(lines))
