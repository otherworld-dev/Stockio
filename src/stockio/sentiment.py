"""News fetching and financial sentiment analysis.

Scans multiple news sources for stories that could affect stock prices:
  - Per-ticker Yahoo Finance feeds (region-aware)
  - Built-in UK & global financial news feeds (BBC, Guardian, Reuters, etc.)
  - User-configured RSS feeds
  - Reddit posts from finance subreddits (cashtag + company name matching)

Matches headlines to tickers by:
  - Ticker symbol (e.g. "VOD.L", "AAPL")
  - Cashtag (e.g. "$AAPL", "$VOD")
  - Company name (e.g. "Vodafone", "Rolls-Royce")
  - Broad market keywords (e.g. "Bank of England", "interest rate")

Uses a FinBERT transformer model to score sentiment as bullish / bearish /
neutral for each stock.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import feedparser
import requests

from stockio import config
from stockio.config import NEWS_FEEDS, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentiment model (lazy-loaded so startup stays fast when not needed)
# ---------------------------------------------------------------------------

_sentiment_pipeline = None


def _get_pipeline():
    """Lazy-load the FinBERT sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        log.info("Loading FinBERT sentiment model (first call — may take a moment) ...")
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU — set to 0 for GPU
        )
        log.info("Sentiment model loaded.")
    return _sentiment_pipeline


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NewsItem:
    title: str
    link: str
    published: str
    source: str
    match_type: str = "ticker"  # "ticker", "name", "broad_market"


@dataclass
class SentimentScore:
    """Aggregated sentiment for a ticker."""

    ticker: str
    score: float  # -1 (very bearish) to +1 (very bullish)
    num_articles: int
    headlines: list[str]
    market_sentiment: float = 0.0  # broad market mood (-1 to +1)


# ---------------------------------------------------------------------------
# Built-in news feeds (always scanned, in addition to user-configured feeds)
# ---------------------------------------------------------------------------

_BUILTIN_FEEDS = [
    # ---- UK ----
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/uk/business/rss",
    "https://www.theguardian.com/business/stock-markets/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^FTSE&region=GB&lang=en-GB",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^FTMC&region=GB&lang=en-GB",
    # ---- US / Global ----
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US",
    "https://www.theguardian.com/business/useconomy/rss",
    # ---- Europe ----
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^STOXX50E&region=EU&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GDAXI&region=DE&lang=en-US",
    # ---- Asia-Pacific ----
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^N225&region=JP&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^HSI&region=HK&lang=en-US",
]

# ---------------------------------------------------------------------------
# Broad market keyword detection
# ---------------------------------------------------------------------------

# Headlines containing these terms affect ALL tickers on the given market.
# The sentiment is blended in at a lower weight.
_BROAD_MARKET_KEYWORDS: list[tuple[str, str]] = [
    # (keyword_pattern, market_region)
    # --- UK macro ---
    (r"\bbank of england\b", "GB"),
    (r"\bboe\b", "GB"),
    (r"\bbase rate\b", "GB"),
    (r"\bftse\b", "GB"),
    (r"\buk econom", "GB"),
    (r"\bsterling\b", "GB"),
    (r"\bgilt\b", "GB"),
    (r"\bchancellor\b", "GB"),
    # --- US macro ---
    (r"\bfederal reserve\b", "ALL"),
    (r"\bthe fed\b", "ALL"),
    (r"\bfed rate\b", "ALL"),
    (r"\bwall street\b", "ALL"),
    (r"\bs&p 500\b", "ALL"),
    (r"\bdow jones\b", "ALL"),
    (r"\bnasdaq\b", "ALL"),
    (r"\btreasury yield", "ALL"),
    (r"\bus econom", "ALL"),
    # --- EU macro ---
    (r"\becb\b", "ALL"),
    (r"\beuropean central bank\b", "ALL"),
    (r"\beurozone\b", "ALL"),
    (r"\beu econom", "ALL"),
    (r"\bdax\b", "ALL"),
    (r"\beuro stoxx\b", "ALL"),
    # --- Asia macro ---
    (r"\bbank of japan\b", "ALL"),
    (r"\bboj\b", "ALL"),
    (r"\bnikkei\b", "ALL"),
    (r"\bhang seng\b", "ALL"),
    (r"\bchina econom", "ALL"),
    (r"\bpboc\b", "ALL"),
    (r"\byuan\b", "ALL"),
    # --- Global / cross-market ---
    (r"\binterest rate", "ALL"),
    (r"\binflation\b", "ALL"),
    (r"\bcpi\b", "ALL"),
    (r"\bgdp\b", "ALL"),
    (r"\brecession\b", "ALL"),
    (r"\bfiscal\b", "ALL"),
    (r"\bbudget\b", "ALL"),
    (r"\btax cut", "ALL"),
    (r"\btax hike", "ALL"),
    (r"\btax rise", "ALL"),
    (r"\bunemployment\b", "ALL"),
    (r"\bjobs report\b", "ALL"),
    (r"\bnon.?farm payroll", "ALL"),
    (r"\btrade war\b", "ALL"),
    (r"\btariff", "ALL"),
    (r"\bsanction", "ALL"),
    (r"\bmarket crash\b", "ALL"),
    (r"\bmarket rall", "ALL"),
    (r"\bbear market\b", "ALL"),
    (r"\bbull market\b", "ALL"),
    (r"\bstock market\b", "ALL"),
    (r"\bglobal growth\b", "ALL"),
    (r"\bsupply chain\b", "ALL"),
    (r"\bgeopolitic", "ALL"),
    (r"\bcurrency war", "ALL"),
    (r"\bdebt ceiling\b", "ALL"),
    (r"\bcredit rating\b", "ALL"),
    (r"\bdefault\b", "ALL"),
    # --- Commodities ---
    (r"\boil price", "ALL"),
    (r"\bcrude oil\b", "ALL"),
    (r"\bbrent\b", "ALL"),
    (r"\bwti\b", "ALL"),
    (r"\bopec\b", "ALL"),
    (r"\bgold price", "ALL"),
    (r"\bcopper price", "ALL"),
    (r"\bnatural gas price", "ALL"),
    (r"\blithium price", "ALL"),
    (r"\biron ore\b", "ALL"),
    (r"\bcommodit", "ALL"),
    # --- Crypto / fintech (market sentiment proxy) ---
    (r"\bbitcoin\b", "ALL"),
    (r"\bcrypto crash\b", "ALL"),
    (r"\bcrypto rall", "ALL"),
]

# ---------------------------------------------------------------------------
# Per-ticker Yahoo feed template
# ---------------------------------------------------------------------------

_TICKER_FEED_TEMPLATE = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={ticker}&region={region}&lang={lang}"
)

# ---------------------------------------------------------------------------
# Company name matching
# ---------------------------------------------------------------------------

# Suffixes to strip from company names for headline matching
_NAME_STRIP_SUFFIXES = [
    " plc", " p.l.c.", " ltd", " limited", " inc", " inc.",
    " corp", " corp.", " corporation", " holdings", " holding",
    " group", " & co", " co.", " sa", " s.a.", " ag", " se",
    " n.v.", " nv",
]

# Minimum length for a company name search term (avoids matching "BP", "AA" etc.
# in random words). Short names are still matched as whole words via ticker symbol.
_MIN_NAME_LENGTH = 4


def _clean_company_name(raw_name: str) -> str:
    """Clean a company name for headline matching.

    E.g. "Vodafone Group PLC" → "vodafone"
         "Rolls-Royce Holdings plc" → "rolls-royce"
    """
    name = raw_name.strip().lower()
    for suffix in _NAME_STRIP_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    # If the cleaned name still has multiple words, also prepare a shorter version
    return name


def _build_name_index(tickers: list[str]) -> dict[str, list[str]]:
    """Build a mapping of ``{search_term: [ticker, ...]}`` for company name matching.

    Includes:
      - The ticker symbol itself (stripped of exchange suffix)
      - The cleaned company name
      - The first word of the company name (if long enough)
    """
    try:
        from stockio.market_discovery import get_ticker_names
        names = get_ticker_names(tickers)
    except (ImportError, Exception):
        names = {}

    index: dict[str, list[str]] = {}

    for ticker in tickers:
        # Always index by ticker symbol (without exchange suffix)
        base = ticker.split(".")[0].lower()
        index.setdefault(base, []).append(ticker)

        # Index by company name if we have it
        raw_name = names.get(ticker, "")
        if not raw_name:
            continue

        cleaned = _clean_company_name(raw_name)
        if len(cleaned) >= _MIN_NAME_LENGTH:
            index.setdefault(cleaned, []).append(ticker)

        # Also index by just the first significant word if it's distinctive
        # e.g. "vodafone group" → also match just "vodafone"
        parts = cleaned.split()
        if len(parts) > 1 and len(parts[0]) >= _MIN_NAME_LENGTH:
            index.setdefault(parts[0], []).append(ticker)

    return index


def _headline_matches_name(headline: str, search_term: str) -> bool:
    """Check if headline mentions a company name (case-insensitive, word boundary)."""
    pattern = rf"\b{re.escape(search_term)}\b"
    return bool(re.search(pattern, headline, re.IGNORECASE))


def _is_broad_market_story(headline: str) -> bool:
    """Check if a headline is about broad market/macro events."""
    hl_lower = headline.lower()
    for pattern, _ in _BROAD_MARKET_KEYWORDS:
        if re.search(pattern, hl_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------


def _ticker_search_name(ticker: str) -> str:
    """Return the base name to match in headlines (strip exchange suffix)."""
    return ticker.split(".")[0]


def fetch_news(tickers: list[str], max_per_source: int = 20) -> dict[str, list[NewsItem]]:
    """Fetch recent headlines relevant to each ticker.

    Scans:
      1. Per-ticker Yahoo Finance RSS (region-aware)
      2. Built-in UK/global financial news feeds
      3. User-configured feeds from STOCKIO_NEWS_FEEDS
      4. Reddit posts from configured subreddits (cashtag + name matching)

    Matches headlines against:
      - Ticker symbols and cashtags ($AAPL, $VOD)
      - Company names (from market discovery cache)
      - Broad market keywords (applied to all tickers)

    Returns ``{ticker: [NewsItem, ...]}``.
    """
    result: dict[str, list[NewsItem]] = {t: [] for t in tickers}

    # Lazy import to avoid circular dependency
    try:
        from stockio.market_discovery import get_market_region, get_news_lang
    except ImportError:
        get_market_region = None
        get_news_lang = None

    # Build company name → ticker index for matching
    name_index = _build_name_index(tickers)

    # 1. Per-ticker RSS feeds (region-aware, limited to avoid too many requests)
    # Only fetch per-ticker feeds for a subset to keep things fast
    ticker_feed_limit = min(len(tickers), 30)
    for ticker in tickers[:ticker_feed_limit]:
        if get_market_region and get_news_lang:
            region = get_market_region(ticker)
            lang = get_news_lang(ticker)
        else:
            region = "US"
            lang = "en-US"

        url = _TICKER_FEED_TEMPLATE.format(ticker=ticker, region=region, lang=lang)
        items = _parse_feed(url, source="yahoo-ticker", max_items=max_per_source)
        for item in items:
            item.match_type = "ticker"
        result[ticker].extend(items)

    # 2. Fetch all general feeds (builtin + user-configured)
    # Deduplicate feed URLs
    all_feed_urls = list(dict.fromkeys(_BUILTIN_FEEDS + NEWS_FEEDS))
    all_general_items: list[NewsItem] = []
    for feed_url in all_feed_urls:
        items = _parse_feed(feed_url, source="general", max_items=max_per_source)
        all_general_items.extend(items)

    log.info("Fetched %d headlines from %d general/builtin feeds",
             len(all_general_items), len(all_feed_urls))

    # 3. Match general headlines to tickers
    seen: dict[str, set[str]] = {t: set() for t in tickers}  # avoid duplicates

    for item in all_general_items:
        hl_lower = item.title.lower()
        matched_tickers: set[str] = set()

        # 3a. Match by company name / ticker symbol
        for search_term, mapped_tickers in name_index.items():
            if _headline_matches_name(item.title, search_term):
                for t in mapped_tickers:
                    if t in result and item.title not in seen[t]:
                        news_item = NewsItem(
                            title=item.title,
                            link=item.link,
                            published=item.published,
                            source=item.source,
                            match_type="name" if len(search_term) >= _MIN_NAME_LENGTH else "ticker",
                        )
                        result[t].append(news_item)
                        seen[t].add(item.title)
                        matched_tickers.add(t)

        # 3b. Check if this is a broad market story
        if _is_broad_market_story(item.title):
            for ticker in tickers:
                if ticker not in matched_tickers and item.title not in seen[ticker]:
                    broad_item = NewsItem(
                        title=item.title,
                        link=item.link,
                        published=item.published,
                        source=item.source,
                        match_type="broad_market",
                    )
                    result[ticker].append(broad_item)
                    seen[ticker].add(item.title)

    # 4. Reddit posts
    reddit_items = fetch_reddit_posts(tickers, name_index)
    for ticker, items in reddit_items.items():
        if ticker in result:
            # Track already-seen titles to avoid duplicates
            existing_titles = {it.title for it in result[ticker]}
            for item in items:
                if item.title not in existing_titles:
                    result[ticker].append(item)
                    existing_titles.add(item.title)

    # Log summary
    name_matches = sum(
        1 for items in result.values()
        for it in items if it.match_type == "name"
    )
    broad_matches = sum(
        1 for items in result.values()
        for it in items if it.match_type == "broad_market"
    )
    ticker_matches = sum(
        1 for items in result.values()
        for it in items if it.match_type == "ticker"
    )
    reddit_matches = sum(
        1 for items in result.values()
        for it in items if it.source.startswith("reddit/")
    )

    log.info(
        "News matched: %d by ticker, %d by company name, %d broad market, %d from Reddit",
        ticker_matches, name_matches, broad_matches, reddit_matches,
    )

    for t, items in result.items():
        if items:
            log.info("  %s: %d articles", t, len(items))

    return result


def _parse_feed(url: str, source: str, max_items: int) -> list[NewsItem]:
    try:
        feed = feedparser.parse(url)
        items: list[NewsItem] = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            items.append(
                NewsItem(
                    title=title,
                    link=entry.get("link", ""),
                    published=entry.get("published", ""),
                    source=source,
                )
            )
        return items
    except Exception as exc:
        log.warning("Failed to parse feed %s: %s", url, exc)
        return []


# ---------------------------------------------------------------------------
# Reddit / Social media fetching
# ---------------------------------------------------------------------------

_REDDIT_USER_AGENT = "Stockio/1.0 (stock sentiment bot)"

# Cashtag pattern: $AAPL, $VOD, $TSLA etc. (1-6 uppercase letters after $)
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,6})\b")


def _fetch_subreddit_posts(subreddit: str, limit: int = 25) -> list[dict]:
    """Fetch hot posts from a subreddit using the public JSON API (no auth needed)."""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    try:
        resp = requests.get(
            url,
            params={"limit": limit, "raw_json": 1},
            headers={"User-Agent": _REDDIT_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post.get("stickied"):
                continue
            posts.append({
                "title": post.get("title", ""),
                "selftext": post.get("selftext", "")[:500],
                "score": post.get("score", 0),
                "url": f"https://www.reddit.com{post.get('permalink', '')}",
                "subreddit": subreddit,
            })
        return posts
    except Exception as exc:
        log.warning("Failed to fetch r/%s: %s", subreddit, exc)
        return []


def _extract_cashtags(text: str) -> list[str]:
    """Extract cashtags like $AAPL from text. Returns base symbols (no suffix)."""
    return _CASHTAG_RE.findall(text)


def fetch_reddit_posts(
    tickers: list[str],
    name_index: dict[str, list[str]],
) -> dict[str, list[NewsItem]]:
    """Fetch Reddit posts from configured subreddits and match to tickers.

    Matches by:
      - Cashtag ($AAPL, $VOD etc.)
      - Ticker symbol mention
      - Company name mention
      - Broad market keywords

    Posts are weighted by Reddit score (upvotes) — higher-scored posts
    carry more influence.

    Returns ``{ticker: [NewsItem, ...]}``.
    """
    if not config.REDDIT_ENABLED:
        return {}

    result: dict[str, list[NewsItem]] = {t: [] for t in tickers}
    seen: dict[str, set[str]] = {t: set() for t in tickers}

    # Build a lookup of base symbol → ticker(s) for cashtag matching
    # e.g. "AAPL" → ["AAPL"], "VOD" → ["VOD.L"]
    base_to_tickers: dict[str, list[str]] = {}
    for ticker in tickers:
        base = ticker.split(".")[0].upper()
        base_to_tickers.setdefault(base, []).append(ticker)

    all_posts: list[dict] = []
    for subreddit in config.REDDIT_SUBREDDITS:
        posts = _fetch_subreddit_posts(subreddit, limit=config.REDDIT_MAX_POSTS)
        all_posts.extend(posts)
        if posts:
            log.info("  r/%s: fetched %d posts", subreddit, len(posts))
        time.sleep(0.5)  # polite delay between subreddits

    log.info("Fetched %d Reddit posts from %d subreddits",
             len(all_posts), len(config.REDDIT_SUBREDDITS))

    for post in all_posts:
        title = post["title"]
        text = title + " " + post.get("selftext", "")
        matched_tickers: set[str] = set()

        # 1. Cashtag matching (highest confidence for Reddit)
        cashtags = _extract_cashtags(text)
        for tag in cashtags:
            for t in base_to_tickers.get(tag, []):
                if t in result and title not in seen[t]:
                    result[t].append(NewsItem(
                        title=title,
                        link=post["url"],
                        published="",
                        source=f"reddit/r/{post['subreddit']}",
                        match_type="ticker",
                    ))
                    seen[t].add(title)
                    matched_tickers.add(t)

        # 2. Company name / ticker symbol matching (reuse existing name index)
        for search_term, mapped_tickers in name_index.items():
            if _headline_matches_name(title, search_term):
                for t in mapped_tickers:
                    if t in result and t not in matched_tickers and title not in seen[t]:
                        result[t].append(NewsItem(
                            title=title,
                            link=post["url"],
                            published="",
                            source=f"reddit/r/{post['subreddit']}",
                            match_type="name" if len(search_term) >= _MIN_NAME_LENGTH else "ticker",
                        ))
                        seen[t].add(title)
                        matched_tickers.add(t)

        # 3. Broad market keywords
        if _is_broad_market_story(title):
            for ticker in tickers:
                if ticker not in matched_tickers and title not in seen[ticker]:
                    result[ticker].append(NewsItem(
                        title=title,
                        link=post["url"],
                        published="",
                        source=f"reddit/r/{post['subreddit']}",
                        match_type="broad_market",
                    ))
                    seen[ticker].add(title)

    # Log summary
    reddit_matches = sum(len(items) for items in result.values())
    tickers_with_reddit = sum(1 for items in result.values() if items)
    if reddit_matches:
        log.info(
            "Reddit matched: %d posts to %d tickers",
            reddit_matches, tickers_with_reddit,
        )

    return result


def _headline_mentions(headline: str, ticker: str) -> bool:
    """Check whether *headline* plausibly mentions *ticker*."""
    pattern = rf"\b{re.escape(ticker)}\b"
    return bool(re.search(pattern, headline, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

# Map FinBERT labels to numeric scores
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# How much weight broad market stories get vs company-specific ones
_BROAD_MARKET_WEIGHT = 0.3  # 30% of normal weight
_NAME_MATCH_WEIGHT = 0.8  # 80% — slightly less confident than ticker match


def analyse_sentiment(news: dict[str, list[NewsItem]]) -> dict[str, SentimentScore]:
    """Score sentiment for each ticker based on fetched headlines.

    Weights different match types differently:
      - ticker match (news): full weight (1.0)
      - company name match: 0.8 weight
      - broad market story: 0.3 weight
      - Reddit source: scaled by REDDIT_WEIGHT (default 0.3)

    Returns ``{ticker: SentimentScore}``.
    """
    pipe = _get_pipeline()
    scores: dict[str, SentimentScore] = {}

    # First, compute broad market sentiment once (shared across tickers)
    broad_headlines: list[str] = []
    seen_broad: set[str] = set()
    for items in news.values():
        for it in items:
            if it.match_type == "broad_market" and it.title not in seen_broad:
                broad_headlines.append(it.title)
                seen_broad.add(it.title)

    market_sentiment = 0.0
    if broad_headlines:
        truncated = [h[:512] for h in broad_headlines]
        try:
            results = pipe(truncated, batch_size=16, truncation=True)
            total_w = 0.0
            weighted_s = 0.0
            for res in results:
                label = res["label"].lower()
                conf = res["score"]
                weighted_s += _LABEL_MAP.get(label, 0.0) * conf
                total_w += conf
            market_sentiment = weighted_s / total_w if total_w > 0 else 0.0
        except Exception as exc:
            log.warning("Broad market sentiment analysis failed: %s", exc)

        log.info(
            "Broad market sentiment: %+.4f (%d headlines)",
            market_sentiment, len(broad_headlines),
        )

    # Now score each ticker
    for ticker, items in news.items():
        if not items:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[],
                market_sentiment=market_sentiment,
            )
            continue

        # Separate company-specific vs broad market items
        specific_items = [it for it in items if it.match_type != "broad_market"]
        headlines = [it.title for it in specific_items if it.title.strip()]

        if not headlines and not broad_headlines:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[],
                market_sentiment=market_sentiment,
            )
            continue

        # Score company-specific headlines
        specific_score = 0.0
        specific_weight = 0.0

        if headlines:
            truncated = [h[:512] for h in headlines]
            try:
                results = pipe(truncated, batch_size=16, truncation=True)
                for i, res in enumerate(results):
                    label = res["label"].lower()
                    conf = res["score"]
                    # Weight by match type
                    item = specific_items[i] if i < len(specific_items) else None
                    match_type = item.match_type if item else "ticker"
                    type_weight = _NAME_MATCH_WEIGHT if match_type == "name" else 1.0
                    # Scale down Reddit sources
                    if item and item.source.startswith("reddit/"):
                        type_weight *= config.REDDIT_WEIGHT
                    w = conf * type_weight
                    specific_score += _LABEL_MAP.get(label, 0.0) * w
                    specific_weight += w
            except Exception as exc:
                log.error("Sentiment analysis failed for %s: %s", ticker, exc)

        # Blend: company-specific sentiment + broad market sentiment
        if specific_weight > 0:
            ticker_sentiment = specific_score / specific_weight
            # Blend with market sentiment (market contributes 20% when we have specific news)
            blended = ticker_sentiment * 0.8 + market_sentiment * 0.2
        else:
            # No company-specific news — use only broad market sentiment
            blended = market_sentiment

        all_headlines = [it.title for it in items if it.title.strip()]
        total_articles = len(all_headlines)
        display_headlines = all_headlines[:5]

        scores[ticker] = SentimentScore(
            ticker=ticker,
            score=round(blended, 4),
            num_articles=total_articles,
            headlines=display_headlines,
            market_sentiment=round(market_sentiment, 4),
        )
        if total_articles > 0:
            log.info(
                "Sentiment for %s: %+.4f (specific=%+.4f, market=%+.4f, %d articles)",
                ticker,
                blended,
                specific_score / specific_weight if specific_weight > 0 else 0.0,
                market_sentiment,
                total_articles,
            )

    return scores


def get_sentiment_scores(tickers: list[str]) -> dict[str, SentimentScore]:
    """Convenience wrapper: fetch news then score sentiment."""
    news = fetch_news(tickers)
    return analyse_sentiment(news)
