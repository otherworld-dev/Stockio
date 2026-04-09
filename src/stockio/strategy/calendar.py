"""Economic calendar — fetches scheduled events and provides ML features.

Uses the ForexFactory calendar JSON mirror (free, no auth).
Events are cached and refreshed every 6 hours.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import structlog

from stockio.config import Settings

log = structlog.get_logger()

# ForexFactory calendar JSON mirror (unofficial, widely used)
_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map instrument names to their constituent currencies
_INSTRUMENT_CURRENCIES: dict[str, tuple[str, ...]] = {
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "USD_JPY": ("USD", "JPY"),
    "AUD_USD": ("AUD", "USD"),
    "NZD_USD": ("NZD", "USD"),
    "USD_CAD": ("USD", "CAD"),
    "USD_CHF": ("USD", "CHF"),
    "EUR_GBP": ("EUR", "GBP"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
    # Commodities — affected by USD events
    "XAU_USD": ("XAU", "USD"),
    "XAG_USD": ("XAG", "USD"),
    "BCO_USD": ("OIL", "USD"),
}

# Impact level mapping
_IMPACT_MAP = {"High": 3, "Medium": 2, "Low": 1}


class EconomicEvent:
    """A single scheduled economic event."""

    __slots__ = ("timestamp", "currency", "impact", "title", "forecast", "previous")

    def __init__(
        self,
        timestamp: datetime,
        currency: str,
        impact: str,
        title: str,
        forecast: str = "",
        previous: str = "",
    ) -> None:
        self.timestamp = timestamp
        self.currency = currency
        self.impact = impact  # "High", "Medium", "Low"
        self.title = title
        self.forecast = forecast
        self.previous = previous

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "currency": self.currency,
            "impact": self.impact,
            "title": self.title,
            "forecast": self.forecast,
            "previous": self.previous,
        }


class EconomicCalendar:
    """Fetches and caches economic events from ForexFactory."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._events: list[EconomicEvent] = []
        self._last_refresh: datetime | None = None
        self._refresh_interval = 6 * 3600  # 6 hours
        self._blackout_minutes = 30  # Don't trade within 30 min of high-impact event

    def needs_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        age = (datetime.now(UTC) - self._last_refresh).total_seconds()
        return age >= self._refresh_interval

    def refresh(self) -> None:
        """Fetch this week's economic events."""
        try:
            resp = httpx.get(_CALENDAR_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            events: list[EconomicEvent] = []
            for item in data:
                try:
                    # Parse the date string — format varies
                    date_str = item.get("date", "")
                    ts = _parse_event_date(date_str)
                    if ts is None:
                        continue

                    events.append(
                        EconomicEvent(
                            timestamp=ts,
                            currency=item.get("country", "").upper(),
                            impact=item.get("impact", "Low"),
                            title=item.get("title", ""),
                            forecast=str(item.get("forecast", "")),
                            previous=str(item.get("previous", "")),
                        )
                    )
                except Exception:
                    continue

            self._events = sorted(events, key=lambda e: e.timestamp)
            self._last_refresh = datetime.now(UTC)
            log.info(
                "calendar_refreshed",
                events=len(self._events),
                high_impact=sum(1 for e in self._events if e.impact == "High"),
            )
        except Exception:
            log.exception("calendar_refresh_failed")

    def get_upcoming(
        self, currency: str | None = None, hours_ahead: int = 48
    ) -> list[EconomicEvent]:
        """Return upcoming events, optionally filtered by currency."""
        now = datetime.now(UTC)
        cutoff = now + timedelta(hours=hours_ahead)
        events = [
            e
            for e in self._events
            if now <= e.timestamp <= cutoff
        ]
        if currency:
            events = [e for e in events if e.currency == currency]
        return events

    def hours_until_next_high_impact(self, instrument: str) -> float | None:
        """Hours until next high-impact event for either currency in the pair."""
        currencies = _INSTRUMENT_CURRENCIES.get(instrument, ())
        now = datetime.now(UTC)

        for event in self._events:
            if event.timestamp <= now:
                continue
            if event.impact == "High" and event.currency in currencies:
                delta = (event.timestamp - now).total_seconds() / 3600
                return delta
        return None

    def is_event_window(self, instrument: str) -> bool:
        """True if within blackout window of a high-impact event."""
        currencies = _INSTRUMENT_CURRENCIES.get(instrument, ())
        now = datetime.now(UTC)
        window = timedelta(minutes=self._blackout_minutes)

        for event in self._events:
            if event.impact != "High":
                continue
            if event.currency not in currencies:
                continue
            if abs(event.timestamp - now) <= window:
                return True
        return False

    def get_features(self, instrument: str) -> dict[str, float]:
        """Return ML features for a given instrument."""
        hours_high = self.hours_until_next_high_impact(instrument)
        currencies = _INSTRUMENT_CURRENCIES.get(instrument, ())
        now = datetime.now(UTC)

        # Hours until next medium+ impact event
        hours_medium = None
        for event in self._events:
            if event.timestamp <= now:
                continue
            if event.currency in currencies and event.impact in ("High", "Medium"):
                hours_medium = (event.timestamp - now).total_seconds() / 3600
                break

        # Count events in next 4 hours
        cutoff_4h = now + timedelta(hours=4)
        events_4h = sum(
            1
            for e in self._events
            if now <= e.timestamp <= cutoff_4h
            and e.currency in currencies
            and e.impact in ("High", "Medium")
        )

        return {
            "hours_until_high_event": min(hours_high / 168, 1.0) if hours_high is not None else 1.0,
            "hours_until_medium_event": (
                min(hours_medium / 168, 1.0) if hours_medium is not None else 1.0
            ),
            "is_event_window": 1.0 if self.is_event_window(instrument) else 0.0,
            "events_next_4h": min(events_4h / 5, 1.0),
        }

    def get_all_upcoming_dict(self) -> list[dict]:
        """Return all upcoming events as dicts (for dashboard API)."""
        return [e.to_dict() for e in self.get_upcoming(hours_ahead=72)]


def _parse_event_date(date_str: str) -> datetime | None:
    """Parse ForexFactory date string into datetime."""
    if not date_str:
        return None
    # Try multiple formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%b %d, %Y %I:%M%p",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    return None
