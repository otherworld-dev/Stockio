"""Entry point for the Stockio trading bot."""

from __future__ import annotations

import signal
import sys
import threading

import structlog


def _configure_logging(level: str) -> None:
    """Set up structlog with JSON output."""
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
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main() -> None:
    """Run the trading bot."""
    from stockio.broker import OandaBroker
    from stockio.config import load_instruments, load_settings
    from stockio.engine import TradingEngine
    from stockio.strategy.notifier import TelegramNotifier

    settings = load_settings()
    _configure_logging(settings.log_level)

    log = structlog.get_logger()
    log.info(
        "starting",
        environment=settings.oanda_environment,
        granularity=settings.granularity,
        cycle_seconds=settings.cycle_seconds,
    )

    instruments = load_instruments()
    log.info("instruments_loaded", instruments=list(instruments.keys()))

    broker = OandaBroker(settings)
    notifier = TelegramNotifier(settings)
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

    # Main loop
    while not shutdown.is_set():
        try:
            engine.run_cycle()
        except Exception:
            log.exception("cycle_failed")

        shutdown.wait(timeout=settings.cycle_seconds)

    log.info("shutdown_complete", cycles_run=engine.cycle_count)


if __name__ == "__main__":
    main()
