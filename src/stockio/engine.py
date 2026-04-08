"""Trading engine — orchestrates the cycle: data → indicators → scoring → trading."""

from __future__ import annotations

import collections
from datetime import datetime, timezone

import structlog

from stockio.broker.base import BrokerBase
from stockio.broker.models import Candle
from stockio.config import InstrumentConfig, Settings

log = structlog.get_logger()


class TradingEngine:
    """Runs the periodic trading cycle."""

    def __init__(
        self,
        broker: BrokerBase,
        instruments: dict[str, InstrumentConfig],
        settings: Settings,
    ) -> None:
        self._broker = broker
        self._instruments = instruments
        self._settings = settings
        self._cycle_count = 0

        # Bounded candle cache — one deque per instrument, maxlen prevents leaks
        self._candle_cache: dict[str, collections.deque[Candle]] = {
            name: collections.deque(maxlen=settings.lookback_bars)
            for name in instruments
        }
        self._warmed_up: set[str] = set()

    def run_cycle(self) -> None:
        """Execute one trading cycle."""
        self._cycle_count += 1
        cycle_log = log.bind(cycle=self._cycle_count)
        cycle_log.info("cycle_start")

        for name in self._instruments:
            try:
                self._update_candles(name)
            except Exception:
                cycle_log.exception("candle_fetch_failed", instrument=name)

        cycle_log.info(
            "cycle_complete",
            instruments_ready=len(self._warmed_up),
            instruments_total=len(self._instruments),
        )

    def _update_candles(self, instrument: str) -> None:
        """Fetch and cache candles for a single instrument."""
        cache = self._candle_cache[instrument]

        if instrument not in self._warmed_up:
            # First run — backfill full lookback
            candles = self._broker.get_candles(
                instrument=instrument,
                granularity=self._settings.granularity,
                count=self._settings.lookback_bars,
            )
            cache.clear()
            cache.extend(candles)
            if candles:
                self._warmed_up.add(instrument)
                log.info(
                    "warmup_complete",
                    instrument=instrument,
                    bars=len(candles),
                )
        else:
            # Subsequent runs — fetch only recent candles
            candles = self._broker.get_candles(
                instrument=instrument,
                granularity=self._settings.granularity,
                count=5,
            )
            # Append only candles newer than what we have
            if cache:
                last_ts = cache[-1].timestamp
                new = [c for c in candles if c.timestamp > last_ts]
            else:
                new = candles
            cache.extend(new)

    @property
    def candle_cache(self) -> dict[str, collections.deque[Candle]]:
        return self._candle_cache

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
