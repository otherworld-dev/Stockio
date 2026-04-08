"""LLM-based sentiment analysis — NewsAPI + RSS headlines → Claude Haiku scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import feedparser
import httpx
import structlog

from stockio.config import InstrumentConfig, Settings

log = structlog.get_logger()

_SENTIMENT_PROMPT = """\
You are a financial sentiment analyst. Given these recent news headlines \
related to {instrument} ({display_name}), rate the overall market sentiment \
on a scale from -1.0 (very bearish) to +1.0 (very bullish).

Consider the likely impact on the price direction over the next 1-4 hours.
Return ONLY a single decimal number between -1.0 and 1.0, nothing else.

Headlines:
{headlines}"""


class SentimentAnalyzer:
    """Fetches news headlines and scores them via Claude Haiku."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._news_api_key = settings.news_api_key
        self._cache: dict[str, float] = {}
        self._cache_time: datetime | None = None
        self._enabled = bool(settings.anthropic_api_key)
        self._client = None

        if self._enabled:
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            log.info("sentiment_disabled", reason="no ANTHROPIC_API_KEY")

    def needs_refresh(self) -> bool:
        if not self._enabled:
            return False
        if self._cache_time is None:
            return True
        age = (datetime.now(UTC) - self._cache_time).total_seconds()
        return age >= self._settings.sentiment_refresh_seconds

    def get_sentiment(self, instrument: str) -> float:
        return self._cache.get(instrument, 0.0)

    def refresh_all(self, instruments: dict[str, InstrumentConfig]) -> dict[str, float]:
        """Fetch headlines and score sentiment for all instruments."""
        if not self._enabled:
            return {}

        results: dict[str, float] = {}
        for name, cfg in instruments.items():
            try:
                headlines = self._fetch_headlines(cfg.news_keywords)
                score = (
                    self._analyze(name, cfg.display_name, headlines)
                    if headlines
                    else 0.0
                )
                results[name] = score
                log.info(
                    "sentiment_scored",
                    instrument=name,
                    score=round(score, 3),
                    headlines=len(headlines),
                )
            except Exception:
                log.exception("sentiment_failed", instrument=name)
                results[name] = 0.0

        self._cache = results
        self._cache_time = datetime.now(UTC)
        return results

    def _fetch_headlines(self, keywords: list[str]) -> list[str]:
        """Fetch headlines from NewsAPI, fall back to RSS."""
        headlines = self._fetch_newsapi(keywords)
        if not headlines:
            headlines = self._fetch_rss(keywords)
        return headlines[: self._settings.max_headlines]

    def _fetch_newsapi(self, keywords: list[str]) -> list[str]:
        """Fetch from NewsAPI.org /v2/everything endpoint."""
        if not self._news_api_key:
            return []

        query = " OR ".join(keywords[:5])  # Batch keywords to reduce API calls
        cutoff = datetime.now(UTC) - timedelta(hours=self._settings.news_lookback_hours)

        try:
            resp = httpx.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sortBy": "publishedAt",
                    "pageSize": self._settings.max_headlines,
                    "apiKey": self._news_api_key,
                    "language": "en",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                article["title"]
                for article in data.get("articles", [])
                if article.get("title")
            ]
        except Exception:
            log.exception("newsapi_fetch_failed")
            return []

    def _fetch_rss(self, keywords: list[str]) -> list[str]:
        """Fallback: fetch from configured RSS feeds and filter by keywords."""
        headlines: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords]

        for feed_url in self._settings.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.get("entries", []):
                    title = entry.get("title", "")
                    # Filter by keyword relevance
                    title_lower = title.lower()
                    if any(kw in title_lower for kw in keywords_lower):
                        headlines.append(title)
            except Exception:
                log.exception("rss_fetch_failed", feed=feed_url)

        return headlines

    def _analyze(self, instrument: str, display_name: str, headlines: list[str]) -> float:
        """Call Claude Haiku to score headline sentiment."""
        prompt = _SENTIMENT_PROMPT.format(
            instrument=instrument,
            display_name=display_name,
            headlines="\n".join(f"- {h}" for h in headlines),
        )

        text = ""
        try:
            response = self._client.messages.create(
                model=self._settings.llm_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            score = float(text)
            return max(-1.0, min(1.0, score))
        except (ValueError, IndexError):
            log.warning("sentiment_parse_failed", instrument=instrument, raw=text)
            return 0.0
        except Exception:
            log.exception("anthropic_call_failed", instrument=instrument)
            return 0.0
