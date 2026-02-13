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

# News feeds
NEWS_FEEDS = [
    f.strip()
    for f in os.getenv(
        "STOCKIO_NEWS_FEEDS",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^FTSE&region=GB&lang=en-GB",
    ).split(",")
    if f.strip()
]

# Watchlist
WATCHLIST = [
    t.strip()
    for t in os.getenv(
        "STOCKIO_WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,JNJ"
    ).split(",")
    if t.strip()
]

# Scheduling
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
