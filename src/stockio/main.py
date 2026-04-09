"""Entry point and bot runner for Stockio."""

from __future__ import annotations

import logging
import signal
import sys
import threading

import structlog


def configure_logging(level: str) -> None:
    """Set up structlog with JSON output."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def run_bot(cycle_seconds_override: int | None = None) -> None:
    """Run the headless trading bot loop."""
    from stockio import db
    from stockio.config import load_instruments, load_settings
    from stockio.engine import TradingEngine
    from stockio.strategy.notifier import TelegramNotifier
    from stockio.strategy.sentiment import SentimentAnalyzer

    settings = load_settings()
    configure_logging(settings.log_level)

    cycle_seconds = cycle_seconds_override or settings.cycle_seconds

    log = structlog.get_logger()

    db.set_default_db(settings.get_db_path())

    instruments = load_instruments()
    log.info("instruments_loaded", instruments=list(instruments.keys()))

    # Auto-select broker: OANDA if credentials exist, Yahoo Finance otherwise
    if settings.oanda_api_token and settings.oanda_account_id:
        from stockio.broker import OandaBroker

        broker = OandaBroker(settings)
        broker_name = "oanda"
    else:
        from stockio.broker import YahooBroker

        broker = YahooBroker(initial_budget=settings.initial_budget)
        broker_name = "yahoo (paper mode — no OANDA credentials)"

    log.info(
        "starting",
        broker=broker_name,
        granularity=settings.granularity,
        cycle_seconds=cycle_seconds,
    )
    notifier = TelegramNotifier(settings)
    sentiment = SentimentAnalyzer(settings)
    engine = TradingEngine(
        broker=broker, instruments=instruments, settings=settings, notifier=notifier
    )

    # Graceful shutdown via SIGTERM (systemd) or SIGINT (Ctrl+C)
    shutdown = threading.Event()

    def _handle_shutdown(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        log.info("shutdown_signal_received", signal=sig_name)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    while not shutdown.is_set():
        try:
            if sentiment.needs_refresh():
                scores = sentiment.refresh_all(instruments)
                engine.update_sentiment(scores)

            engine.run_cycle()
            engine.maybe_daily_summary()
        except Exception:
            log.exception("cycle_failed")

        shutdown.wait(timeout=cycle_seconds)

    log.info("shutdown_complete", cycles_run=engine.cycle_count)


def main() -> None:
    """Entry point — delegates to CLI."""
    from stockio.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
