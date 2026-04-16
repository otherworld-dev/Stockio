"""Thread-based bot instance management for the web dashboard.

Supports running paper + live bot instances simultaneously, each with
an isolated SQLite database.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from stockio import db
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
    last_trump_sentiment: dict[str, float] = field(default_factory=dict)
    sentiment_analyzer: Any = None  # SentimentAnalyzer instance


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
    """Signal a bot instance to stop and wait briefly for it to finish."""
    slot = _slots.get(name)
    if not slot:
        return False

    with slot.lock:
        if not slot.running:
            return False
        slot.shutdown_event.set()
        # Bump generation so even if the thread is stuck in init,
        # it will exit on the next loop iteration check
        slot.generation += 1
        log.info("bot_stop_requested", instance=name)

    # Wait up to 10 seconds for the thread to finish
    if slot.thread and slot.thread.is_alive():
        slot.thread.join(timeout=10)

    # If thread is still alive after timeout, force-mark as stopped
    # (the daemon thread will eventually die on its own)
    if slot.thread and slot.thread.is_alive():
        log.warning("bot_stop_timeout", instance=name)
        with slot.lock:
            slot.running = False
            slot.engine = None

    return True


def _create_broker_for_slot(slot_name: str, settings: Settings):
    """Create the appropriate broker for a bot slot.

    Paper slot → OANDA practice account (or Yahoo fallback)
    Live slot  → OANDA live account (or Yahoo fallback)
    """
    from stockio.broker import YahooBroker

    if slot_name == "paper":
        account_id = (
            settings.oanda_practice_account_id or settings.oanda_account_id
        )
        api_token = (
            settings.oanda_practice_api_token or settings.oanda_api_token
        )
        environment = "practice"
    elif slot_name == "live":
        account_id = settings.oanda_live_account_id
        api_token = settings.oanda_live_api_token
        environment = "live"
    else:
        account_id = ""
        api_token = ""
        environment = "practice"

    if account_id and api_token:
        # Create a temporary settings-like object for OandaBroker
        from stockio.broker.oanda import OandaBroker

        class _SlotSettings:
            pass

        s = _SlotSettings()
        s.oanda_account_id = account_id
        s.oanda_api_token = api_token
        s.oanda_environment = environment
        broker = OandaBroker(s)
        log.info(
            "broker_created",
            slot=slot_name,
            broker="oanda",
            environment=environment,
        )
        return broker

    log.info("broker_created", slot=slot_name, broker="yahoo_paper")
    return YahooBroker(initial_budget=settings.initial_budget)


def _run_bot(slot: BotSlot, generation: int) -> None:
    """Bot thread main loop."""
    try:
        db.set_active_db(slot.db_path)
        settings = _settings
        instruments = load_instruments()

        broker = _create_broker_for_slot(slot.name, settings)

        notifier = TelegramNotifier(settings)
        sentiment = SentimentAnalyzer(settings)
        slot.sentiment_analyzer = sentiment
        engine = TradingEngine(
            broker=broker,
            instruments=instruments,
            settings=settings,
            notifier=notifier,
            shutdown_event=slot.shutdown_event,
        )
        slot.engine = engine

        while not slot.shutdown_event.is_set():
            if slot.generation != generation:
                break  # Stale thread — a new one was started
            try:
                if sentiment.needs_refresh():
                    scores = sentiment.refresh_all(instruments)
                    if slot.shutdown_event.is_set():
                        break
                    engine.update_sentiment(scores)
                    slot.last_sentiment = scores
                    slot.last_trump_sentiment = {
                        k: sentiment.get_trump_sentiment(k)
                        for k in instruments
                    }

                if slot.shutdown_event.is_set():
                    break

                engine.run_cycle()
                engine.maybe_daily_summary()

                # Expose latest signals for the dashboard
                # (read from the engine's last scoring pass)
            except Exception as exc:
                slot.last_error = str(exc)
                log.exception("bot_cycle_error", instance=slot.name)

            # Check for updated cycle_seconds from dashboard settings
            try:
                saved_cycle = db.get_setting("cycle_seconds")
                cycle_wait = int(saved_cycle) if saved_cycle else settings.cycle_seconds
            except (ValueError, TypeError):
                cycle_wait = settings.cycle_seconds
            slot.shutdown_event.wait(timeout=cycle_wait)

    except Exception as exc:
        slot.last_error = str(exc)
        log.exception("bot_thread_crashed", instance=slot.name)
    finally:
        with slot.lock:
            slot.running = False
            slot.engine = None
        log.info("bot_stopped", instance=slot.name)
