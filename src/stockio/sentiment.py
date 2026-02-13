"""News fetching and financial sentiment analysis.

Uses RSS feeds for news headlines and a FinBERT-based transformer model
to score sentiment as bullish / bearish / neutral for each stock.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import feedparser
import requests

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


@dataclass
class SentimentScore:
    """Aggregated sentiment for a ticker."""

    ticker: str
    score: float  # -1 (very bearish) to +1 (very bullish)
    num_articles: int
    headlines: list[str]


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------

_TICKER_FEEDS = {
    # Yahoo Finance per-ticker RSS — {ticker} is replaced at runtime
    "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
}


def fetch_news(tickers: list[str], max_per_source: int = 15) -> dict[str, list[NewsItem]]:
    """Fetch recent headlines relevant to each ticker.

    Returns ``{ticker: [NewsItem, ...]}``.
    """
    result: dict[str, list[NewsItem]] = {t: [] for t in tickers}

    # 1. Per-ticker RSS feeds
    for ticker in tickers:
        url = _TICKER_FEEDS["yahoo"].format(ticker=ticker)
        items = _parse_feed(url, source="yahoo", max_items=max_per_source)
        result[ticker].extend(items)

    # 2. General financial feeds — match headlines containing ticker symbols
    for feed_url in NEWS_FEEDS:
        items = _parse_feed(feed_url, source="general", max_items=max_per_source)
        for item in items:
            for ticker in tickers:
                if _headline_mentions(item.title, ticker):
                    result[ticker].append(item)

    for t, items in result.items():
        log.info("Fetched %d headlines for %s", len(items), t)
    return result


def _parse_feed(url: str, source: str, max_items: int) -> list[NewsItem]:
    try:
        feed = feedparser.parse(url)
        items: list[NewsItem] = []
        for entry in feed.entries[:max_items]:
            items.append(
                NewsItem(
                    title=entry.get("title", ""),
                    link=entry.get("link", ""),
                    published=entry.get("published", ""),
                    source=source,
                )
            )
        return items
    except Exception as exc:
        log.warning("Failed to parse feed %s: %s", url, exc)
        return []


def _headline_mentions(headline: str, ticker: str) -> bool:
    """Check whether *headline* plausibly mentions *ticker*."""
    pattern = rf"\b{re.escape(ticker)}\b"
    return bool(re.search(pattern, headline, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

# Map FinBERT labels to numeric scores
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def analyse_sentiment(news: dict[str, list[NewsItem]]) -> dict[str, SentimentScore]:
    """Score sentiment for each ticker based on fetched headlines.

    Returns ``{ticker: SentimentScore}``.
    """
    pipe = _get_pipeline()
    scores: dict[str, SentimentScore] = {}

    for ticker, items in news.items():
        if not items:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[]
            )
            continue

        headlines = [it.title for it in items if it.title.strip()]
        if not headlines:
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=0, headlines=[]
            )
            continue

        # Truncate headlines to 512 tokens max (FinBERT limit)
        truncated = [h[:512] for h in headlines]
        try:
            results = pipe(truncated, batch_size=16, truncation=True)
        except Exception as exc:
            log.error("Sentiment analysis failed for %s: %s", ticker, exc)
            scores[ticker] = SentimentScore(
                ticker=ticker, score=0.0, num_articles=len(headlines), headlines=headlines
            )
            continue

        # Weighted average: weight by model confidence
        total_weight = 0.0
        weighted_sum = 0.0
        for res in results:
            label = res["label"].lower()
            conf = res["score"]
            weighted_sum += _LABEL_MAP.get(label, 0.0) * conf
            total_weight += conf

        avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        scores[ticker] = SentimentScore(
            ticker=ticker,
            score=round(avg, 4),
            num_articles=len(headlines),
            headlines=headlines[:5],  # keep top 5 for display
        )
        log.info("Sentiment for %s: %.4f (%d articles)", ticker, avg, len(headlines))

    return scores


def get_sentiment_scores(tickers: list[str]) -> dict[str, SentimentScore]:
    """Convenience wrapper: fetch news then score sentiment."""
    news = fetch_news(tickers)
    return analyse_sentiment(news)
