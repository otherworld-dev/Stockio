"""Dynamic stock discovery across multiple markets/exchanges.

Fetches comprehensive ticker lists from Yahoo Finance's screener API,
caches them in SQLite, and provides the bot with a rotating batch of
tickers to analyse each cycle.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass

import requests

from stockio import config
from stockio.config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported markets
# ---------------------------------------------------------------------------

SUPPORTED_MARKETS: dict[str, dict] = {
    "LSE": {
        "name": "London Stock Exchange – Main Market",
        "yahoo_exchanges": ["LSE"],
        "suffix": ".L",
        "region": "GB",
        "currency": "GBP",
        "news_lang": "en-GB",
    },
    "AIM": {
        "name": "AIM (Alternative Investment Market)",
        "yahoo_exchanges": ["AIM"],
        "suffix": ".L",
        "region": "GB",
        "currency": "GBP",
        "news_lang": "en-GB",
    },
    "NYSE": {
        "name": "New York Stock Exchange",
        "yahoo_exchanges": ["NYQ"],
        "suffix": "",
        "region": "US",
        "currency": "USD",
        "news_lang": "en-US",
    },
    "NASDAQ": {
        "name": "NASDAQ",
        "yahoo_exchanges": ["NMS", "NGM", "NCM"],
        "suffix": "",
        "region": "US",
        "currency": "USD",
        "news_lang": "en-US",
    },
    "EURONEXT": {
        "name": "Euronext (Paris)",
        "yahoo_exchanges": ["PAR"],
        "suffix": ".PA",
        "region": "FR",
        "currency": "EUR",
        "news_lang": "en-US",
    },
    "XETRA": {
        "name": "Deutsche Börse (Xetra)",
        "yahoo_exchanges": ["GER"],
        "suffix": ".DE",
        "region": "DE",
        "currency": "EUR",
        "news_lang": "en-US",
    },
}


@dataclass
class DiscoveredTicker:
    symbol: str
    name: str
    market: str
    exchange: str
    currency: str
    market_cap: float | None


# ---------------------------------------------------------------------------
# Database helpers (uses the same DB as portfolio)
# ---------------------------------------------------------------------------


def _init_market_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_tickers (
            symbol       TEXT NOT NULL,
            market       TEXT NOT NULL,
            name         TEXT DEFAULT '',
            exchange     TEXT DEFAULT '',
            currency     TEXT DEFAULT '',
            market_cap   REAL,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (symbol, market)
        );

        CREATE TABLE IF NOT EXISTS market_refresh (
            market        TEXT PRIMARY KEY,
            last_refresh  TEXT NOT NULL,
            ticker_count  INTEGER NOT NULL DEFAULT 0
        );
        """
    )


@contextmanager
def _get_conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_market_tables(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Yahoo Finance screener API
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _yahoo_session() -> tuple[requests.Session, str]:
    """Create a Yahoo Finance session and obtain a crumb token."""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    # Visit Yahoo Finance to set cookies
    session.get("https://finance.yahoo.com", timeout=15)

    # Fetch the crumb tied to the session cookies
    crumb_resp = session.get(
        "https://query2.finance.yahoo.com/v1/test/getcrumb",
        timeout=10,
    )
    crumb_resp.raise_for_status()
    crumb = crumb_resp.text.strip()
    log.info("Yahoo Finance session established (crumb obtained)")
    return session, crumb


def _screener_query(
    session: requests.Session,
    crumb: str,
    exchange_code: str,
    offset: int = 0,
    size: int = 250,
) -> dict:
    """Run a single Yahoo Finance screener query for an exchange."""
    body = {
        "offset": offset,
        "size": size,
        "sortField": "intradaymarketcap",
        "sortType": "DESC",
        "quoteType": "EQUITY",
        "query": {
            "operator": "AND",
            "operands": [
                {"operator": "EQ", "operands": ["exchange", exchange_code]},
            ],
        },
    }

    resp = session.post(
        f"https://query2.finance.yahoo.com/v1/finance/screener?crumb={crumb}",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_exchange_tickers(
    session: requests.Session,
    crumb: str,
    exchange_code: str,
    max_results: int,
) -> list[DiscoveredTicker]:
    """Paginate through the Yahoo screener to get all equities on an exchange."""
    tickers: list[DiscoveredTicker] = []
    offset = 0
    page_size = 250

    while offset < max_results:
        try:
            data = _screener_query(session, crumb, exchange_code, offset, page_size)
        except Exception as exc:
            log.warning(
                "Screener query failed for %s at offset %d: %s",
                exchange_code, offset, exc,
            )
            break

        result_list = data.get("finance", {}).get("result", [])
        if not result_list:
            break

        quotes = result_list[0].get("quotes", [])
        if not quotes:
            break

        for q in quotes:
            tickers.append(
                DiscoveredTicker(
                    symbol=q.get("symbol", ""),
                    name=q.get("shortName") or q.get("longName", ""),
                    market="",  # filled in by caller
                    exchange=exchange_code,
                    currency=q.get("currency", ""),
                    market_cap=q.get("marketCap"),
                )
            )

        total = result_list[0].get("total", 0)
        offset += page_size
        if offset >= total:
            break

        # Polite delay between pages
        time.sleep(0.5)

    return tickers


# ---------------------------------------------------------------------------
# Discovery orchestration
# ---------------------------------------------------------------------------


def discover_market(market_key: str) -> list[DiscoveredTicker]:
    """Discover all tickers on a single market using Yahoo Finance screener.

    Returns the list of discovered tickers.
    """
    market_def = SUPPORTED_MARKETS.get(market_key)
    if market_def is None:
        log.error("Unknown market: %s (supported: %s)",
                  market_key, ", ".join(SUPPORTED_MARKETS))
        return []

    log.info("Discovering tickers on %s (%s) ...", market_key, market_def["name"])

    try:
        session, crumb = _yahoo_session()
    except Exception as exc:
        log.error("Failed to establish Yahoo Finance session: %s", exc)
        return []

    all_tickers: list[DiscoveredTicker] = []
    for exch_code in market_def["yahoo_exchanges"]:
        log.info("  Querying exchange code: %s", exch_code)
        tickers = _fetch_exchange_tickers(
            session, crumb, exch_code,
            max_results=config.MAX_TICKERS_PER_MARKET,
        )
        for t in tickers:
            t.market = market_key
        all_tickers.extend(tickers)
        log.info("  Found %d tickers on %s", len(tickers), exch_code)

    log.info("Total discovered for %s: %d tickers", market_key, len(all_tickers))
    return all_tickers


def refresh_market(market_key: str) -> int:
    """Discover tickers for a market and cache them in the database.

    Returns the number of tickers cached.
    """
    tickers = discover_market(market_key)
    if not tickers:
        log.warning("No tickers discovered for %s — keeping existing cache", market_key)
        return 0

    now = dt.datetime.utcnow().isoformat()

    with _get_conn() as conn:
        # Clear old entries for this market and re-insert
        conn.execute("DELETE FROM market_tickers WHERE market = ?", (market_key,))
        conn.executemany(
            "INSERT INTO market_tickers "
            "(symbol, market, name, exchange, currency, market_cap, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (t.symbol, t.market, t.name, t.exchange, t.currency, t.market_cap, now)
                for t in tickers
            ],
        )
        conn.execute(
            "INSERT OR REPLACE INTO market_refresh (market, last_refresh, ticker_count) "
            "VALUES (?, ?, ?)",
            (market_key, now, len(tickers)),
        )

    log.info("Cached %d tickers for %s", len(tickers), market_key)
    return len(tickers)


def refresh_all_markets() -> dict[str, int]:
    """Refresh ticker caches for all configured markets.

    Returns ``{market: ticker_count}``.
    """
    results: dict[str, int] = {}
    for market_key in config.MARKETS:
        count = refresh_market(market_key)
        results[market_key] = count
    return results


def maybe_refresh() -> None:
    """Refresh markets whose cache has expired (older than MARKET_REFRESH_HOURS)."""
    now = dt.datetime.utcnow()
    needs_refresh: list[str] = []

    with _get_conn() as conn:
        for market_key in config.MARKETS:
            row = conn.execute(
                "SELECT last_refresh FROM market_refresh WHERE market = ?",
                (market_key,),
            ).fetchone()

            if row is None:
                needs_refresh.append(market_key)
            else:
                last = dt.datetime.fromisoformat(row["last_refresh"])
                elapsed_hours = (now - last).total_seconds() / 3600
                if elapsed_hours >= config.MARKET_REFRESH_HOURS:
                    needs_refresh.append(market_key)

    for market_key in needs_refresh:
        log.info("Market cache expired for %s — refreshing", market_key)
        refresh_market(market_key)


# ---------------------------------------------------------------------------
# Ticker retrieval (used by the bot)
# ---------------------------------------------------------------------------


def get_cached_tickers(markets: list[str] | None = None) -> list[str]:
    """Return all cached ticker symbols for the given (or configured) markets.

    Returns symbols in descending market-cap order.
    """
    markets = markets or config.MARKETS
    if not markets:
        return list(config.WATCHLIST)

    with _get_conn() as conn:
        placeholders = ",".join("?" for _ in markets)
        rows = conn.execute(
            f"SELECT symbol FROM market_tickers "
            f"WHERE market IN ({placeholders}) "
            f"ORDER BY COALESCE(market_cap, 0) DESC",
            markets,
        ).fetchall()

    symbols = [r["symbol"] for r in rows]

    # Also include any manually configured watchlist tickers
    for t in config.WATCHLIST:
        if t and t not in symbols:
            symbols.append(t)

    return symbols


def get_market_summary() -> list[dict]:
    """Return a summary of cached markets for display / API."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT market, last_refresh, ticker_count FROM market_refresh "
            "ORDER BY market"
        ).fetchall()
    return [
        {
            "market": r["market"],
            "name": SUPPORTED_MARKETS.get(r["market"], {}).get("name", r["market"]),
            "last_refresh": r["last_refresh"],
            "ticker_count": r["ticker_count"],
        }
        for r in rows
    ]


def get_ticker_count() -> int:
    """Return total number of cached tickers across all configured markets."""
    with _get_conn() as conn:
        placeholders = ",".join("?" for _ in config.MARKETS)
        if not config.MARKETS:
            return len(config.WATCHLIST)
        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM market_tickers "
            f"WHERE market IN ({placeholders})",
            config.MARKETS,
        ).fetchone()
    return (row["cnt"] if row else 0) + len(config.WATCHLIST)


def get_market_region(ticker: str) -> str:
    """Return the news region code (e.g. 'GB', 'US') for a ticker."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT market FROM market_tickers WHERE symbol = ? LIMIT 1",
            (ticker,),
        ).fetchone()

    if row:
        market_def = SUPPORTED_MARKETS.get(row["market"], {})
        return market_def.get("region", "US")

    # Guess from suffix
    if ticker.endswith(".L"):
        return "GB"
    if ticker.endswith(".PA"):
        return "FR"
    if ticker.endswith(".DE"):
        return "DE"
    return "US"


def get_news_lang(ticker: str) -> str:
    """Return the news language tag for a ticker (e.g. 'en-GB')."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT market FROM market_tickers WHERE symbol = ? LIMIT 1",
            (ticker,),
        ).fetchone()

    if row:
        market_def = SUPPORTED_MARKETS.get(row["market"], {})
        return market_def.get("news_lang", "en-US")

    if ticker.endswith(".L"):
        return "en-GB"
    return "en-US"
