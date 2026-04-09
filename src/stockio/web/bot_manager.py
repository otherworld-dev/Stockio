"""Thread-based bot instance management for the web dashboard.

Supports running paper + live bot instances simultaneously, each with
an isolated SQLite database.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from stockio import db
from stockio.broker import OandaBroker
from stockio.config import Settings, load_instruments
from stockio.engine import TradingEngine
from stockio.strategy.notifier import TelegramNotifier
from stockio.strategy.sentiment import SentimentAnalyzer

log = structlog.get_logger()


@dataclass
class BotSlot:
    """One independently-running bot instance."""

    name: str  # "paper" or "live"
    db_path: Path = field(default_factory=lambda: Path("data/stockio.db"))
    thread: threading.Thread | None = None
    engine: TradingEngine | None = None
    running: bool = False
    generation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown_event: threading.Event = field(default_factory=threading.Event)
    last_error: str = ""

    # Latest cycle data for the dashboard (updated by the bot thread)
    last_signals: list[dict] = field(default_factory=list)
    last_sentiment: dict[str, float] = field(default_factory=dict)


# Fixed slots — both can run simultaneously
_settings: Settings | None = None
_slots: dict[str, BotSlot] = {}


def init_slots(settings: Settings) -> None:
    """Initialize bot slots from settings."""
    global _settings, _slots
    _settings = settings
    _slots = {
        "paper": BotSlot(
            name="paper",
            db_path=settings.get_db_path("paper"),
        ),
        "live": BotSlot(
            name="live",
            db_path=settings.get_db_path("live"),
        ),
    }


def get_slots() -> dict[str, BotSlot]:
    return _slots


def get_slot(name: str) -> BotSlot | None:
    return _slots.get(name)


def start_bot(name: str) -> bool:
    """Start a bot instance in a background thread."""
    slot = _slots.get(name)
    if not slot or not _settings:
        return False

    with slot.lock:
        if slot.running:
            return False

        slot.generation += 1
        slot.shutdown_event.clear()
        slot.running = True
        slot.last_error = ""

        gen = slot.generation
        slot.thread = threading.Thread(
            target=_run_bot,
            args=(slot, gen),
            name=f"stockio-{name}",
            daemon=True,
        )
        slot.thread.start()
        log.info("bot_started", instance=name, generation=gen)
        return True


def stop_bot(name: str) -> bool:
    """Signal a bot instance to stop."""
    slot = _slots.get(name)
    if not slot:
        return False

    with slot.lock:
        if not slot.running:
            return False
        slot.shutdown_event.set()
        log.info("bot_stop_requested", instance=name)
        return True


def _run_bot(slot: BotSlot, generation: int) -> None:
    """Bot thread main loop."""
    try:
        db.set_active_db(slot.db_path)
        settings = _settings
        instruments = load_instruments()

        if settings.oanda_api_token and settings.oanda_account_id:
            broker = OandaBroker(settings)
        else:
            from stockio.broker import YahooBroker

            broker = YahooBroker(initial_budget=settings.initial_budget)

        notifier = TelegramNotifier(settings)
        sentiment = SentimentAnalyzer(settings)
        engine = TradingEngine(
            broker=broker,
            instruments=instruments,
            settings=settings,
            notifier=notifier,
        )
        slot.engine = engine

        while not slot.shutdown_event.is_set():
            if slot.generation != generation:
                break  # Stale thread — a new one was started
            try:
                if sentiment.needs_refresh():
                    scores = sentiment.refresh_all(instruments)
                    engine.update_sentiment(scores)
                    slot.last_sentiment = scores

                engine.run_cycle()
                engine.maybe_daily_summary()

                # Expose latest signals for the dashboard
                # (read from the engine's last scoring pass)
            except Exception as exc:
                slot.last_error = str(exc)
                log.exception("bot_cycle_error", instance=slot.name)

            slot.shutdown_event.wait(timeout=settings.cycle_seconds)

    except Exception as exc:
        slot.last_error = str(exc)
        log.exception("bot_thread_crashed", instance=slot.name)
    finally:
        with slot.lock:
            slot.running = False
            slot.engine = None
        log.info("bot_stopped", instance=slot.name)
