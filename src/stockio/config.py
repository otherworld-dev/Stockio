"""Central configuration loaded from environment / .env file."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "stockio.db"

# Budget
INITIAL_BUDGET_GBP = float(os.getenv("STOCKIO_BUDGET", "500.00"))

# Trading mode
MODE = os.getenv("STOCKIO_MODE", "paper")  # "paper" or "live"

# Alpaca (live trading)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# User-configured news feeds (in addition to built-in feeds in sentiment.py)
NEWS_FEEDS = [
    f.strip()
    for f in os.getenv("STOCKIO_NEWS_FEEDS", "").split(",")
    if f.strip()
]

# ---------------------------------------------------------------------------
# Markets & Watchlist
# ---------------------------------------------------------------------------

# Markets to scan for stocks (comma-separated).
# Supported: LSE, AIM, NYSE, NASDAQ, EURONEXT, XETRA
# Set to empty string to use STOCKIO_WATCHLIST instead.
MARKETS = [
    m.strip().upper()
    for m in os.getenv("STOCKIO_MARKETS", "LSE,AIM,NYSE,NASDAQ").split(",")
    if m.strip()
]

# Static watchlist — used as ADDITIONAL tickers on top of market discovery.
# If STOCKIO_MARKETS is empty, this is the only source of tickers.
WATCHLIST = [
    t.strip()
    for t in os.getenv("STOCKIO_WATCHLIST", "").split(",")
    if t.strip()
]

# How often to refresh the full list of tickers from each market (hours)
MARKET_REFRESH_HOURS = int(os.getenv("STOCKIO_MARKET_REFRESH_HOURS", "24"))

# How many tickers to analyse per trading cycle (the bot rotates through all)
BATCH_SIZE = int(os.getenv("STOCKIO_BATCH_SIZE", "50"))

# Include penny stocks (price below threshold)
INCLUDE_PENNY_STOCKS = os.getenv("STOCKIO_INCLUDE_PENNY_STOCKS", "true").lower() in (
    "true", "1", "yes",
)

# Maximum number of tickers to discover per market (safety limit)
MAX_TICKERS_PER_MARKET = int(os.getenv("STOCKIO_MAX_TICKERS_PER_MARKET", "5000"))

# ---------------------------------------------------------------------------
# Reddit / Social Media
# ---------------------------------------------------------------------------

REDDIT_ENABLED = os.getenv("STOCKIO_REDDIT_ENABLED", "true").lower() in (
    "true", "1", "yes",
)

# Subreddits to monitor (comma-separated)
REDDIT_SUBREDDITS = [
    s.strip()
    for s in os.getenv(
        "STOCKIO_SUBREDDITS",
        "wallstreetbets,stocks,investing,UKInvesting,pennystocks,StockMarket",
    ).split(",")
    if s.strip()
]

# Max posts to fetch per subreddit
REDDIT_MAX_POSTS = int(os.getenv("STOCKIO_REDDIT_MAX_POSTS", "25"))

# How much weight Reddit sentiment gets vs news (0.0–1.0)
REDDIT_WEIGHT = float(os.getenv("STOCKIO_REDDIT_WEIGHT", "0.3"))

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

INTERVAL_MINUTES = int(os.getenv("STOCKIO_INTERVAL_MINUTES", "30"))
RETRAIN_HOURS = int(os.getenv("STOCKIO_RETRAIN_HOURS", "24"))

# Risk management
MAX_POSITION_PCT = float(os.getenv("STOCKIO_MAX_POSITION_PCT", "20"))
STOP_LOSS_PCT = float(os.getenv("STOCKIO_STOP_LOSS_PCT", "5"))
TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_TAKE_PROFIT_PCT", "15"))

# Logging
LOG_LEVEL = os.getenv("STOCKIO_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a consistently-configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
