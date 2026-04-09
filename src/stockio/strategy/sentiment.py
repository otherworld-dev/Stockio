"""LLM-based sentiment analysis — NewsAPI + RSS + Trump Watch → Claude Haiku scoring."""

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

_TRUMP_PROMPT = """\
You are a financial analyst specialising in political risk. Given these recent \
headlines about the Trump administration, family, and inner circle, assess the \
likely SHORT-TERM impact on {instrument} ({display_name}).

Consider:
- Tariff announcements (negative for affected currencies/commodities)
- Trade deals or tariff pauses (positive/relief rally)
- Executive orders affecting specific sectors
- Sanctions or geopolitical escalation (risk-off → gold up, equities down)
- USD strength/weakness implications
- Elon Musk/DOGE government efficiency moves (affects USD, tech sentiment)
- Trump family business deals (potential conflicts, sector impacts)
- Regulatory changes (SEC, crypto, energy policy)
- Truth Social posts that signal upcoming policy

Rate the impact from -1.0 (very bearish for this instrument) to +1.0 (very bullish).
Return ONLY a single decimal number between -1.0 and 1.0, nothing else.

Headlines:
{headlines}"""

# ---------------------------------------------------------------------------
# Trump / political monitoring feeds (no API key needed)
# ---------------------------------------------------------------------------

_TRUMP_FEEDS = [
    # White House presidential actions (executive orders, proclamations)
    "https://www.whitehouse.gov/presidential-actions/feed/",
    # Google News RSS — Trump + market-moving topics
    "https://news.google.com/rss/search?q=trump+tariff+OR+trade+OR+executive+order&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=trump+economy+OR+sanctions+OR+market&hl=en-US&gl=US&ceid=US:en",
    # Trump family + key figures with market influence
    "https://news.google.com/rss/search?q=elon+musk+DOGE+OR+government+OR+regulation&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=jared+kushner+OR+ivanka+trump+business+OR+deal&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=donald+trump+jr+OR+eric+trump+business&hl=en-US&gl=US&ceid=US:en",
    # Political/economic news
    "https://www.theguardian.com/us-news/donaldtrump/rss",
]

_TRUMP_KEYWORDS = [
    # Trump direct
    "trump", "potus", "white house", "truth social", "mar-a-lago",
    # Tariffs & trade (primary market mover)
    "tariff", "trade war", "trade deal", "trade ban", "reciprocal",
    "import duty", "china trade", "eu tariff", "canada tariff",
    "mexico tariff", "steel tariff", "auto tariff",
    # Executive actions
    "executive order", "sanctions", "presidential action",
    # Trump family & inner circle (market-moving figures)
    "ivanka trump", "jared kushner", "donald trump jr", "eric trump",
    "trump organization", "trump media",
    # Key allies with market influence
    "elon musk", "doge", "vivek ramaswamy",
    # Policy areas that move markets
    "government shutdown", "debt ceiling", "federal reserve",
    "deregulation", "crypto regulation", "sec chairman",
]

# ---------------------------------------------------------------------------
# Built-in financial news feeds (always scanned, no API key)
# ---------------------------------------------------------------------------

_BUILTIN_FEEDS = [
    # UK
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/uk/business/rss",
    "https://www.theguardian.com/business/stock-markets/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^FTSE&region=GB&lang=en-GB",
    # US / Global
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US",
    "https://www.theguardian.com/business/useconomy/rss",
    # Europe
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^STOXX50E&region=EU&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GDAXI&region=DE&lang=en-US",
    # Asia
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^N225&region=JP&lang=en-US",
]

# ---------------------------------------------------------------------------
# Reddit subreddits (free JSON API, no auth needed)
# ---------------------------------------------------------------------------

_REDDIT_SUBREDDITS = [
    "forex",
    "wallstreetbets",
    "stocks",
    "investing",
    "commodities",
    "Gold",
    "economy",
]

_REDDIT_USER_AGENT = "Stockio/1.0 (trading bot; market sentiment)"


class SentimentAnalyzer:
    """Fetches news headlines and scores them via Claude Haiku.

    Includes Trump/political monitoring with higher weight (1.5x).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._news_api_key = settings.news_api_key
        self._cache: dict[str, float] = {}
        self._trump_cache: dict[str, float] = {}
        self._news_cache: dict[str, float] = {}  # Per-instrument news-only scores
        self._cache_time: datetime | None = None
        self._enabled = bool(settings.anthropic_api_key)
        self._client = None
        self._trump_weight = 1.5  # Political news gets 1.5x weight

        # Cached headlines for dashboard display
        self._last_trump_headlines: list[str] = []
        self._last_news_headlines: dict[str, list[str]] = {}  # Per-instrument

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
        # Check DB override for refresh interval
        try:
            from stockio import db

            saved = db.get_setting("sentiment_refresh_seconds")
            interval = int(saved) if saved else self._settings.sentiment_refresh_seconds
        except Exception:
            interval = self._settings.sentiment_refresh_seconds
        age = (datetime.now(UTC) - self._cache_time).total_seconds()
        return age >= interval

    def get_sentiment(self, instrument: str) -> float:
        return self._cache.get(instrument, 0.0)

    def get_trump_sentiment(self, instrument: str) -> float:
        return self._trump_cache.get(instrument, 0.0)

    def refresh_all(self, instruments: dict[str, InstrumentConfig]) -> dict[str, float]:
        """Fetch headlines and score sentiment for all instruments."""
        if not self._enabled:
            return {}

        # Fetch Trump headlines once (shared across all instruments)
        trump_headlines = self._fetch_trump_headlines()
        if trump_headlines:
            log.info("trump_headlines_fetched", count=len(trump_headlines))

        results: dict[str, float] = {}
        for name, cfg in instruments.items():
            try:
                # Regular sentiment
                headlines = self._fetch_headlines(cfg.news_keywords)
                news_score = (
                    self._analyze(name, cfg.display_name, headlines)
                    if headlines
                    else 0.0
                )

                # Trump/political sentiment (weighted higher)
                trump_score = 0.0
                if trump_headlines:
                    trump_score = self._analyze_trump(
                        name, cfg.display_name, trump_headlines
                    )

                # Weighted combination
                combined = news_score + (trump_score * self._trump_weight)
                # Clamp to [-1, 1]
                combined = max(-1.0, min(1.0, combined))

                results[name] = combined
                self._trump_cache[name] = trump_score
                self._news_cache[name] = news_score
                self._last_news_headlines[name] = headlines
                log.info(
                    "sentiment_scored",
                    instrument=name,
                    news_score=round(news_score, 3),
                    trump_score=round(trump_score, 3),
                    combined=round(combined, 3),
                    news_headlines=len(headlines),
                    trump_headlines=len(trump_headlines),
                )
            except Exception:
                log.exception("sentiment_failed", instrument=name)
                results[name] = 0.0

        self._last_trump_headlines = trump_headlines
        self._cache = results
        self._cache_time = datetime.now(UTC)
        return results

    # ------------------------------------------------------------------
    # Dashboard data accessors
    # ------------------------------------------------------------------

    def get_breakdown(self) -> dict:
        """Return full sentiment breakdown for the dashboard."""
        return {
            name: {
                "news_score": round(self._news_cache.get(name, 0.0), 3),
                "trump_score": round(self._trump_cache.get(name, 0.0), 3),
                "combined": round(self._cache.get(name, 0.0), 3),
                "news_headlines": self._last_news_headlines.get(name, []),
                "reddit_count": sum(
                    1
                    for h in self._last_news_headlines.get(name, [])
                    if h.startswith("[r/")
                ),
            }
            for name in self._cache
        }

    def get_trump_headlines(self) -> list[str]:
        """Return cached Trump headlines for the dashboard."""
        return self._last_trump_headlines

    # ------------------------------------------------------------------
    # Headline fetching
    # ------------------------------------------------------------------

    def _fetch_headlines(self, keywords: list[str]) -> list[str]:
        """Fetch headlines from NewsAPI + Reddit + built-in RSS."""
        headlines: list[str] = []

        # NewsAPI (if key available)
        headlines.extend(self._fetch_newsapi(keywords))

        # Reddit (free, no auth)
        headlines.extend(self._fetch_reddit(keywords))

        # Built-in RSS feeds (always available)
        if not headlines:
            headlines.extend(self._fetch_rss(keywords, self._settings.rss_feeds))
        if not headlines:
            headlines.extend(self._fetch_rss(keywords, _BUILTIN_FEEDS))

        # Deduplicate
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower()
            if h_lower not in seen:
                seen.add(h_lower)
                unique.append(h)

        return unique[: self._settings.max_headlines]

    def _fetch_trump_headlines(self) -> list[str]:
        """Fetch recent Trump/political headlines from dedicated feeds."""
        headlines: list[str] = []
        for feed_url in _TRUMP_FEEDS:
            try:
                resp = httpx.get(feed_url, timeout=10, follow_redirects=True)
                feed = feedparser.parse(resp.text)
                for entry in feed.get("entries", [])[:10]:
                    title = entry.get("title", "")
                    if title:
                        headlines.append(title)
            except Exception:
                log.debug("trump_feed_failed", feed=feed_url)

        # Deduplicate
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower()
            if h_lower not in seen:
                seen.add(h_lower)
                unique.append(h)

        return unique[:15]  # Cap at 15 headlines

    def _fetch_newsapi(self, keywords: list[str]) -> list[str]:
        """Fetch from NewsAPI.org /v2/everything endpoint."""
        if not self._news_api_key:
            return []

        query = " OR ".join(keywords[:5])
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

    def _fetch_reddit(self, keywords: list[str]) -> list[str]:
        """Fetch from Reddit's free JSON API (no auth needed)."""
        headlines: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords]

        for subreddit in _REDDIT_SUBREDDITS:
            try:
                resp = httpx.get(
                    f"https://www.reddit.com/r/{subreddit}/hot.json",
                    params={"limit": 15, "raw_json": 1},
                    headers={"User-Agent": _REDDIT_USER_AGENT},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    if post.get("stickied"):
                        continue
                    title = post.get("title", "")
                    title_lower = title.lower()
                    # Match by keywords
                    if any(kw in title_lower for kw in keywords_lower):
                        headlines.append(f"[r/{subreddit}] {title}")
            except Exception:
                log.debug("reddit_fetch_failed", subreddit=subreddit)

        if headlines:
            log.info("reddit_headlines_fetched", count=len(headlines))
        return headlines

    def _fetch_rss(self, keywords: list[str], feeds: list[str]) -> list[str]:
        """Fetch from RSS feeds, filtered by keywords."""
        headlines: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords]

        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.get("entries", []):
                    title = entry.get("title", "")
                    title_lower = title.lower()
                    if any(kw in title_lower for kw in keywords_lower):
                        headlines.append(title)
            except Exception:
                log.debug("rss_fetch_failed", feed=feed_url)

        return headlines

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _analyze(self, instrument: str, display_name: str, headlines: list[str]) -> float:
        """Score regular news headline sentiment via Claude Haiku."""
        return self._call_llm(
            _SENTIMENT_PROMPT.format(
                instrument=instrument,
                display_name=display_name,
                headlines="\n".join(f"- {h}" for h in headlines),
            ),
            instrument,
        )

    def _analyze_trump(
        self, instrument: str, display_name: str, headlines: list[str]
    ) -> float:
        """Score Trump/political headline impact via Claude Haiku."""
        return self._call_llm(
            _TRUMP_PROMPT.format(
                instrument=instrument,
                display_name=display_name,
                headlines="\n".join(f"- {h}" for h in headlines),
            ),
            instrument,
        )

    def _call_llm(self, prompt: str, instrument: str) -> float:
        """Call Claude Haiku and parse a float score."""
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
            log.warning("llm_parse_failed", instrument=instrument, raw=text)
            return 0.0
        except Exception:
            log.exception("llm_call_failed", instrument=instrument)
            return 0.0
