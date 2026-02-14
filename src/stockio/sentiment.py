"""News fetching and financial sentiment analysis.

Scans multiple news sources for stories that could affect asset prices:
  - Per-ticker Yahoo Finance feeds (region-aware)
  - Built-in UK & global financial news feeds (BBC, Guardian, Reuters, etc.)
  - User-configured RSS feeds
  - Reddit posts from finance subreddits (cashtag + company name matching)
  - Crypto-specific subreddits (r/CryptoCurrency, r/Bitcoin, etc.)

Matches headlines to tickers by:
  - Ticker symbol (e.g. "VOD.L", "AAPL")
  - Cashtag (e.g. "$AAPL", "$VOD")
  - Company name (e.g. "Vodafone", "Rolls-Royce")
  - Forex pair names (e.g. "euro", "dollar", "yen")
  - Commodity names (e.g. "gold", "crude oil", "silver")
  - Crypto names (e.g. "bitcoin", "ethereum")
  - Broad market keywords (e.g. "Bank of England", "interest rate")

Uses a FinBERT transformer model to score sentiment as bullish / bearish /
neutral for each asset.
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
_model_loading = False
_model_error: str | None = None


def warmup_model() -> None:
    """Pre-load the FinBERT model in the background so it's ready for requests.

    Safe to call from a background thread at app startup.
    """
    try:
        _get_pipeline()
    except Exception as exc:
        log.warning("Background model warmup failed (will retry on demand): %s", exc)


def _get_pipeline():
    """Lazy-load the FinBERT sentiment analysis pipeline."""
    global _sentiment_pipeline, _model_loading, _model_error
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    if _model_loading:
        raise RuntimeError(
            "Sentiment model is still loading — please wait a moment and try again"
        )
    _model_loading = True
    _model_error = None
    try:
        log.info("Loading FinBERT sentiment model (first call — may take a moment) ...")
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU — set to 0 for GPU
        )
        log.info("Sentiment model loaded.")
    except Exception as exc:
        _model_error = str(exc)
        _model_loading = False
        raise
    _model_loading = False
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
    # Detailed breakdown for transparency
    news_score: float = 0.0  # sentiment from news sources only
    reddit_score: float = 0.0  # sentiment from Reddit only
    trump_score: float = 0.0  # sentiment from Trump/political sources
    news_count: int = 0  # number of news articles matched
    reddit_count: int = 0  # number of Reddit posts matched
    trump_count: int = 0  # number of Trump/political stories matched
    broad_count: int = 0  # number of broad market stories
    articles: list[dict] = field(default_factory=list)  # [{title, source, match_type, sentiment}]


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
# Trump / Political monitoring feeds
# ---------------------------------------------------------------------------

# These feeds are checked when TRUMP_MONITORING_ENABLED is true.
# Truth Social RSS for @realDonaldTrump is the primary direct source.
_TRUMP_FEEDS = [
    # White House — presidential actions (executive orders, proclamations)
    "https://www.whitehouse.gov/presidential-actions/feed/",
    # Google News RSS — Trump + market-moving topics
    "https://news.google.com/rss/search?q=trump+tariff+OR+trade+OR+executive+order&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=trump+economy+OR+sanctions+OR+market&hl=en-US&gl=US&ceid=US:en",
    # Political/economic news RSS
    "https://www.theguardian.com/us-news/donaldtrump/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US",
]

# ---------------------------------------------------------------------------
# Trump-specific market-moving keyword patterns
# ---------------------------------------------------------------------------

# These patterns identify Trump-related stories that tend to cause immediate,
# outsized market moves. They get weighted at TRUMP_WEIGHT (default 1.5x)
# instead of normal broad market weight.
_TRUMP_KEYWORDS: list[re.Pattern] = [
    # Direct mentions
    re.compile(r"\btrump\b", re.IGNORECASE),
    re.compile(r"\bpotus\b", re.IGNORECASE),
    re.compile(r"\bpresident\s+(trump|donald)", re.IGNORECASE),
    re.compile(r"\btruth\s+social\b", re.IGNORECASE),
    # Tariffs & trade (Trump's signature market mover)
    re.compile(r"\btrump.{0,20}tariff", re.IGNORECASE),
    re.compile(r"\btariff.{0,20}trump", re.IGNORECASE),
    re.compile(r"\btrade\s+war\b", re.IGNORECASE),
    re.compile(r"\btrade\s+deal\b", re.IGNORECASE),
    re.compile(r"\btrade\s+ban\b", re.IGNORECASE),
    re.compile(r"\bimport\s+dut", re.IGNORECASE),
    re.compile(r"\bchina\s+tariff", re.IGNORECASE),
    re.compile(r"\beu\s+tariff", re.IGNORECASE),
    re.compile(r"\bcanada\s+tariff", re.IGNORECASE),
    re.compile(r"\bmexico\s+tariff", re.IGNORECASE),
    re.compile(r"\bsteel\s+tariff", re.IGNORECASE),
    re.compile(r"\baluminium\s+tariff", re.IGNORECASE),
    re.compile(r"\baluminum\s+tariff", re.IGNORECASE),
    re.compile(r"\bauto\s+tariff", re.IGNORECASE),
    re.compile(r"\breciprocal\s+tariff", re.IGNORECASE),
    # Executive actions
    re.compile(r"\bexecutive\s+order\b", re.IGNORECASE),
    re.compile(r"\bpresidential\s+proclamation\b", re.IGNORECASE),
    re.compile(r"\bpresidential\s+memorand", re.IGNORECASE),
    re.compile(r"\bpresidential\s+action\b", re.IGNORECASE),
    # Sanctions & geopolitics
    re.compile(r"\btrump.{0,20}sanction", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}ban\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}restrict", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}threaten", re.IGNORECASE),
    # Deregulation / regulation
    re.compile(r"\btrump.{0,20}deregulat", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}regulat", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}repeal", re.IGNORECASE),
    # Fiscal / spending
    re.compile(r"\btrump.{0,20}tax\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}spend", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}deficit", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}debt\b", re.IGNORECASE),
    # Tech / sector-specific
    re.compile(r"\btrump.{0,20}tech\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}tiktok\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}crypto\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}bitcoin\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}energy\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}drill", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}oil\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}auto\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}pharma\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}drug\s+pric", re.IGNORECASE),
    re.compile(r"\btrump.{0,20}nato\b", re.IGNORECASE),
    # Market commentary referencing Trump
    re.compile(r"\bmarket.{0,20}trump\b", re.IGNORECASE),
    re.compile(r"\bstocks?.{0,15}trump\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,15}stocks?\b", re.IGNORECASE),
    re.compile(r"\btrump.{0,15}market\b", re.IGNORECASE),
    re.compile(r"\btrump\s+bump\b", re.IGNORECASE),
    re.compile(r"\btrump\s+slump\b", re.IGNORECASE),
    re.compile(r"\btrump\s+rally\b", re.IGNORECASE),
    re.compile(r"\bmaga\b", re.IGNORECASE),
]


def _is_trump_story(headline: str) -> bool:
    """Check if a headline is about Trump or his market-moving actions."""
    return any(pat.search(headline) for pat in _TRUMP_KEYWORDS)

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
    (r"\bsilver price", "ALL"),
    (r"\bcopper price", "ALL"),
    (r"\bnatural gas price", "ALL"),
    (r"\bplatinum\b", "ALL"),
    (r"\bpalladium\b", "ALL"),
    (r"\blithium price", "ALL"),
    (r"\biron ore\b", "ALL"),
    (r"\bcommodit", "ALL"),
    (r"\bwheat price", "ALL"),
    (r"\bcorn price", "ALL"),
    (r"\bsoybean", "ALL"),
    (r"\bcoffee price", "ALL"),
    (r"\bcotton price", "ALL"),
    (r"\bprecious metal", "ALL"),
    # --- Forex / Currencies ---
    (r"\bforex\b", "ALL"),
    (r"\bcurrency market", "ALL"),
    (r"\bdollar index\b", "ALL"),
    (r"\bdxy\b", "ALL"),
    (r"\beuro.{0,5}dollar", "ALL"),
    (r"\bpound.{0,5}dollar", "ALL"),
    (r"\bsterling\b", "ALL"),
    (r"\byen\b", "ALL"),
    (r"\byuan\b", "ALL"),
    (r"\bcurrency war", "ALL"),
    (r"\bfx market", "ALL"),
    (r"\bexchange rate", "ALL"),
    # --- Crypto ---
    (r"\bbitcoin\b", "ALL"),
    (r"\bethereum\b", "ALL"),
    (r"\bcrypto crash\b", "ALL"),
    (r"\bcrypto rall", "ALL"),
    (r"\bcryptocurrenc", "ALL"),
    (r"\bdefi\b", "ALL"),
    (r"\bnft\b", "ALL"),
    (r"\bblockchain\b", "ALL"),
    (r"\bstablecoin\b", "ALL"),
    (r"\bbinance\b", "ALL"),
    (r"\bcoinbase\b", "ALL"),
    (r"\bsec.{0,5}crypto", "ALL"),
    (r"\bcrypto.{0,5}regulation", "ALL"),
]

# ---------------------------------------------------------------------------
# Asset-specific keyword matching (for forex, commodities, crypto)
# ---------------------------------------------------------------------------

# Maps keyword patterns → list of ticker symbols they're relevant to.
# This allows non-equity assets to be matched by name in headlines.
_ASSET_KEYWORD_MAP: list[tuple[str, list[str]]] = [
    # --- Forex ---
    (r"\beur/?usd\b", ["EURUSD=X"]),
    (r"\beuro\b.{0,15}dollar\b", ["EURUSD=X"]),
    (r"\bgbp/?usd\b", ["GBPUSD=X"]),
    (r"\bpound\b.{0,15}dollar\b", ["GBPUSD=X"]),
    (r"\bcable\b", ["GBPUSD=X"]),
    (r"\busd/?jpy\b", ["USDJPY=X"]),
    (r"\bdollar\b.{0,15}yen\b", ["USDJPY=X"]),
    (r"\busd/?chf\b", ["USDCHF=X"]),
    (r"\baud/?usd\b", ["AUDUSD=X"]),
    (r"\baustrailian\s+dollar\b", ["AUDUSD=X"]),
    (r"\busd/?cad\b", ["USDCAD=X"]),
    (r"\bnzd/?usd\b", ["NZDUSD=X"]),
    (r"\beur/?gbp\b", ["EURGBP=X"]),
    (r"\beur/?jpy\b", ["EURJPY=X"]),
    (r"\bgbp/?jpy\b", ["GBPJPY=X"]),
    # --- Commodities ---
    (r"\bgold\b", ["GC=F"]),
    (r"\bxau\b", ["GC=F"]),
    (r"\bsilver\b", ["SI=F"]),
    (r"\bxag\b", ["SI=F"]),
    (r"\bcrude\s*oil\b", ["CL=F"]),
    (r"\bwti\b", ["CL=F"]),
    (r"\bbrent\b", ["CL=F"]),
    (r"\boil\s*price", ["CL=F"]),
    (r"\bnatural\s*gas\b", ["NG=F"]),
    (r"\bcopper\b", ["HG=F"]),
    (r"\bplatinum\b", ["PL=F"]),
    (r"\bpalladium\b", ["PA=F"]),
    (r"\bwheat\b", ["ZW=F"]),
    (r"\bcorn\b", ["ZC=F"]),
    (r"\bsoybean", ["ZS=F"]),
    (r"\bcoffee\b", ["KC=F"]),
    (r"\bcotton\b", ["CT=F"]),
    # --- Crypto ---
    (r"\bbitcoin\b", ["BTC-USD"]),
    (r"\bbtc\b", ["BTC-USD"]),
    (r"\bethereum\b", ["ETH-USD"]),
    (r"\beth\b", ["ETH-USD"]),
    (r"\bbnb\b", ["BNB-USD"]),
    (r"\bbinance\s*coin\b", ["BNB-USD"]),
    (r"\bsolana\b", ["SOL-USD"]),
    (r"\bsol\b", ["SOL-USD"]),
    (r"\bripple\b", ["XRP-USD"]),
    (r"\bxrp\b", ["XRP-USD"]),
    (r"\bcardano\b", ["ADA-USD"]),
    (r"\bada\b", ["ADA-USD"]),
    (r"\bdogecoin\b", ["DOGE-USD"]),
    (r"\bdoge\b", ["DOGE-USD"]),
    (r"\bavalanche\b", ["AVAX-USD"]),
    (r"\bpolkadot\b", ["DOT-USD"]),
    (r"\bpolygon\b", ["MATIC-USD"]),
    (r"\bmatic\b", ["MATIC-USD"]),
    (r"\bchainlink\b", ["LINK-USD"]),
    (r"\buniswap\b", ["UNI-USD"]),
    (r"\blitecoin\b", ["LTC-USD"]),
    (r"\bcosmos\b", ["ATOM-USD"]),
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
      - Forex pair names, commodity names, crypto names (from config)
    """
    try:
        from stockio.market_discovery import get_ticker_names
        names = get_ticker_names(tickers)
    except (ImportError, Exception):
        names = {}

    # Add display names for non-equity assets
    from stockio.config import COMMODITY_NAMES, CRYPTO_NAMES, get_asset_display_name
    for ticker in tickers:
        if ticker not in names or not names[ticker]:
            display = get_asset_display_name(ticker)
            if display != ticker:
                names[ticker] = display

    index: dict[str, list[str]] = {}

    for ticker in tickers:
        # Always index by ticker symbol (without exchange suffix and special chars)
        base = ticker.split(".")[0].split("=")[0].split("-")[0].lower()
        if len(base) >= 2:
            index.setdefault(base, []).append(ticker)

        # Index by company/asset name if we have it
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

    # Pre-compile asset keyword patterns for efficiency
    _compiled_asset_keywords = [
        (re.compile(pattern, re.IGNORECASE), asset_tickers)
        for pattern, asset_tickers in _ASSET_KEYWORD_MAP
    ]

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

        # 3a2. Match by asset keyword patterns (forex, commodities, crypto)
        for pat, asset_tickers in _compiled_asset_keywords:
            if pat.search(item.title):
                for t in asset_tickers:
                    if t in result and t not in matched_tickers and item.title not in seen[t]:
                        result[t].append(NewsItem(
                            title=item.title,
                            link=item.link,
                            published=item.published,
                            source=item.source,
                            match_type="name",
                        ))
                        seen[t].add(item.title)
                        matched_tickers.add(t)

        # 3b. Check if this is a Trump-related story (gets boosted weight)
        if _is_trump_story(item.title):
            for ticker in tickers:
                if ticker not in matched_tickers and item.title not in seen[ticker]:
                    trump_item = NewsItem(
                        title=item.title,
                        link=item.link,
                        published=item.published,
                        source=item.source,
                        match_type="trump",
                    )
                    result[ticker].append(trump_item)
                    seen[ticker].add(item.title)
                    matched_tickers.add(ticker)

        # 3c. Check if this is a broad market story
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

    # 5. Trump / political feeds (Truth Social, White House, etc.)
    trump_items = fetch_trump_feeds()
    if trump_items:
        for item in trump_items:
            # Trump posts affect all tickers (market-wide impact)
            for ticker in tickers:
                existing_titles = {it.title for it in result[ticker]}
                if item.title not in existing_titles:
                    result[ticker].append(item)

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
    trump_matches = sum(
        1 for items in result.values()
        for it in items if it.match_type == "trump"
    )

    log.info(
        "News matched: %d ticker, %d name, %d broad, %d reddit, %d trump/political",
        ticker_matches, name_matches, broad_matches, reddit_matches, trump_matches,
    )

    for t, items in result.items():
        if items:
            log.info("  %s: %d articles", t, len(items))

    return result


_FEED_USER_AGENT = "Stockio/1.0 (stock sentiment bot; +https://github.com)"


def _parse_feed(url: str, source: str, max_items: int) -> list[NewsItem]:
    try:
        # Fetch with requests first so we can send a proper User-Agent
        # (many sites block feedparser's bare urllib requests)
        resp = requests.get(
            url,
            headers={"User-Agent": _FEED_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
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
# Trump / Political monitoring
# ---------------------------------------------------------------------------

def fetch_trump_feeds() -> list[NewsItem]:
    """Fetch posts from Trump-specific feeds (White House, Google News, etc.).

    Returns a flat list of NewsItems with source tags identifying the origin.
    Each item is marked with match_type="trump" for special weighting.
    """
    if not config.TRUMP_MONITORING_ENABLED:
        return []

    items: list[NewsItem] = []
    seen_titles: set[str] = set()

    for feed_url in _TRUMP_FEEDS:
        try:
            raw_items = _parse_feed(feed_url, source="trump", max_items=20)
            count = 0
            for item in raw_items:
                if item.title not in seen_titles:
                    # Tag with a more specific source based on the URL
                    if "whitehouse" in feed_url:
                        source = "white_house"
                    elif "news.google" in feed_url:
                        source = "trump_news"
                    elif "guardian" in feed_url:
                        source = "trump_news"
                    else:
                        source = "trump_news"
                    items.append(NewsItem(
                        title=item.title,
                        link=item.link,
                        published=item.published,
                        source=source,
                        match_type="trump",
                    ))
                    seen_titles.add(item.title)
                    count += 1
            log.info("Trump feed %s: %d items", feed_url[:60], count)
        except Exception as exc:
            log.warning("Failed to fetch Trump feed %s: %s", feed_url, exc)

    if items:
        log.info("Fetched %d Trump/political items from %d feeds",
                 len(items), len(_TRUMP_FEEDS))
    return items


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

    # Determine which subreddits to fetch (include crypto subs if crypto tickers present)
    subreddits = list(config.REDDIT_SUBREDDITS)
    has_crypto = any(
        t.endswith("-USD") and t not in ("USD",) for t in tickers
    )
    if has_crypto and config.CRYPTO_ENABLED:
        for sub in config.CRYPTO_SUBREDDITS:
            if sub not in subreddits:
                subreddits.append(sub)

    all_posts: list[dict] = []
    for subreddit in subreddits:
        posts = _fetch_subreddit_posts(subreddit, limit=config.REDDIT_MAX_POSTS)
        all_posts.extend(posts)
        if posts:
            log.info("  r/%s: fetched %d posts", subreddit, len(posts))
        time.sleep(0.5)  # polite delay between subreddits

    log.info("Fetched %d Reddit posts from %d subreddits",
             len(all_posts), len(subreddits))

    # Pre-compile asset keyword patterns for Reddit matching too
    _compiled_asset_kw = [
        (re.compile(pattern, re.IGNORECASE), asset_tickers)
        for pattern, asset_tickers in _ASSET_KEYWORD_MAP
    ]

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

        # 2b. Asset keyword matching (forex, commodities, crypto)
        for pat, asset_tickers in _compiled_asset_kw:
            if pat.search(text):
                for t in asset_tickers:
                    if t in result and t not in matched_tickers and title not in seen[t]:
                        result[t].append(NewsItem(
                            title=title,
                            link=post["url"],
                            published="",
                            source=f"reddit/r/{post['subreddit']}",
                            match_type="name",
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
      - Trump/political stories: boosted by TRUMP_WEIGHT (default 1.5x)

    Returns ``{ticker: SentimentScore}``.
    """
    pipe = _get_pipeline()
    scores: dict[str, SentimentScore] = {}

    # First, compute broad market sentiment once (shared across tickers)
    broad_headlines: list[str] = []
    trump_headlines: list[str] = []
    seen_broad: set[str] = set()
    for items in news.values():
        for it in items:
            if it.title in seen_broad:
                continue
            if it.match_type == "trump":
                trump_headlines.append(it.title)
                seen_broad.add(it.title)
            elif it.match_type == "broad_market":
                broad_headlines.append(it.title)
                seen_broad.add(it.title)

    market_sentiment = 0.0
    trump_sentiment = 0.0

    # Score broad market headlines
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

    # Score Trump/political headlines separately (with boosted weight)
    if trump_headlines:
        truncated = [h[:512] for h in trump_headlines]
        try:
            results = pipe(truncated, batch_size=16, truncation=True)
            total_w = 0.0
            weighted_s = 0.0
            for res in results:
                label = res["label"].lower()
                conf = res["score"]
                weighted_s += _LABEL_MAP.get(label, 0.0) * conf
                total_w += conf
            trump_sentiment = weighted_s / total_w if total_w > 0 else 0.0
        except Exception as exc:
            log.warning("Trump sentiment analysis failed: %s", exc)

        log.info(
            "Trump/political sentiment: %+.4f (%d headlines, weight=%.1fx)",
            trump_sentiment, len(trump_headlines), config.TRUMP_WEIGHT,
        )

    # Blend market and trump sentiment (trump gets boosted weight)
    if trump_headlines and broad_headlines:
        trump_w = config.TRUMP_WEIGHT
        combined_market = (
            market_sentiment * len(broad_headlines)
            + trump_sentiment * len(trump_headlines) * trump_w
        ) / (len(broad_headlines) + len(trump_headlines) * trump_w)
        market_sentiment = combined_market
        log.info(
            "Combined market+trump sentiment: %+.4f",
            market_sentiment,
        )
    elif trump_headlines and not broad_headlines:
        market_sentiment = trump_sentiment

    # Now score each ticker
    for ticker, items in news.items():
        if not items:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[],
                market_sentiment=market_sentiment,
            )
            continue

        # Separate company-specific vs broad market vs trump items
        specific_items = [it for it in items if it.match_type not in ("broad_market", "trump")]
        trump_items = [it for it in items if it.match_type == "trump"]
        broad_items = [it for it in items if it.match_type == "broad_market"]
        # Trump items are scored as specific items but with boosted weight
        all_specific = specific_items + trump_items
        headlines = [it.title for it in all_specific if it.title.strip()]

        if not headlines and not broad_headlines and not trump_headlines:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[],
                market_sentiment=market_sentiment,
            )
            continue

        # Score company-specific + trump headlines and track per-article details
        specific_score = 0.0
        specific_weight = 0.0
        article_details: list[dict] = []

        # Track news vs reddit vs trump breakdown
        news_score_sum = 0.0
        news_weight_sum = 0.0
        reddit_score_sum = 0.0
        reddit_weight_sum = 0.0
        trump_score_sum = 0.0
        trump_weight_sum = 0.0

        if headlines:
            truncated = [h[:512] for h in headlines]
            try:
                results = pipe(truncated, batch_size=16, truncation=True)
                for i, res in enumerate(results):
                    label = res["label"].lower()
                    conf = res["score"]
                    raw_sentiment = _LABEL_MAP.get(label, 0.0) * conf
                    # Weight by match type
                    item = all_specific[i] if i < len(all_specific) else None
                    match_type = item.match_type if item else "ticker"
                    type_weight = _NAME_MATCH_WEIGHT if match_type == "name" else 1.0
                    is_reddit = item and item.source.startswith("reddit/")
                    is_trump = match_type == "trump"
                    # Scale down Reddit sources
                    if is_reddit:
                        type_weight *= config.REDDIT_WEIGHT
                    # Boost Trump/political stories
                    if is_trump:
                        type_weight *= config.TRUMP_WEIGHT
                    w = conf * type_weight
                    specific_score += _LABEL_MAP.get(label, 0.0) * w
                    specific_weight += w

                    # Track per-source breakdown
                    if is_trump:
                        trump_score_sum += raw_sentiment
                        trump_weight_sum += conf
                    elif is_reddit:
                        reddit_score_sum += raw_sentiment
                        reddit_weight_sum += conf
                    else:
                        news_score_sum += raw_sentiment
                        news_weight_sum += conf

                    # Record per-article detail
                    if item:
                        article_details.append({
                            "title": item.title,
                            "source": item.source,
                            "link": item.link,
                            "match_type": item.match_type,
                            "sentiment": round(raw_sentiment, 4),
                            "label": label,
                            "confidence": round(conf, 4),
                        })
            except Exception as exc:
                log.error("Sentiment analysis failed for %s: %s", ticker, exc)

        # Add broad market articles to the detail list (no individual scoring needed)
        for it in broad_items[:5]:
            article_details.append({
                "title": it.title,
                "source": it.source,
                "link": it.link,
                "match_type": "broad_market",
                "sentiment": round(market_sentiment, 4),
                "label": "market",
                "confidence": 0.0,
            })

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

        # Compute per-source averages
        avg_news = news_score_sum / news_weight_sum if news_weight_sum > 0 else 0.0
        avg_reddit = reddit_score_sum / reddit_weight_sum if reddit_weight_sum > 0 else 0.0
        avg_trump = trump_score_sum / trump_weight_sum if trump_weight_sum > 0 else 0.0
        news_count = sum(1 for it in specific_items if not it.source.startswith("reddit/"))
        reddit_count = sum(1 for it in specific_items if it.source.startswith("reddit/"))
        trump_count = len(trump_items)

        scores[ticker] = SentimentScore(
            ticker=ticker,
            score=round(blended, 4),
            num_articles=total_articles,
            headlines=display_headlines,
            market_sentiment=round(market_sentiment, 4),
            news_score=round(avg_news, 4),
            reddit_score=round(avg_reddit, 4),
            trump_score=round(avg_trump, 4),
            news_count=news_count,
            reddit_count=reddit_count,
            trump_count=trump_count,
            broad_count=len(broad_items),
            articles=article_details[:20],  # cap at 20 for memory
        )
        if total_articles > 0:
            log.info(
                "Sentiment for %s: %+.4f (news=%+.4f/%d, reddit=%+.4f/%d, trump=%+.4f/%d, market=%+.4f, broad=%d)",
                ticker,
                blended,
                avg_news, news_count,
                avg_reddit, reddit_count,
                avg_trump, trump_count,
                market_sentiment,
                len(broad_items),
            )

    return scores


def get_sentiment_scores(tickers: list[str]) -> dict[str, SentimentScore]:
    """Convenience wrapper: fetch news then score sentiment."""
    news = fetch_news(tickers)
    return analyse_sentiment(news)
