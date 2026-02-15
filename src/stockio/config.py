"""Central configuration loaded from environment / .env file."""

import os
import logging
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "stockio.db"


def get_db_path(instance_id: str = "") -> Path:
    """Return the database path for a named instance.

    - ``""`` / ``"default"`` → the default ``stockio.db``
    - ``"paper"`` → ``stockio_paper.db``
    - ``"live"`` → ``stockio_live.db``
    """
    if not instance_id or instance_id == "default":
        return DB_PATH
    return DATA_DIR / f"stockio_{instance_id}.db"


# ---------------------------------------------------------------------------
# Asset types
# ---------------------------------------------------------------------------


class AssetType(str, Enum):
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


# ---------------------------------------------------------------------------
# Market metadata — trading hours, sessions, and descriptions
# ---------------------------------------------------------------------------

MARKET_INFO: dict[str, dict] = {
    "equity": {
        "label": "Equities",
        "description": "Stocks listed on major exchanges",
        "currency": "GBP / USD / EUR",
        "schedule": "Weekdays only",
        "sessions": [
            {"name": "LSE / AIM", "open": "08:00", "close": "16:30", "tz": "Europe/London",
             "tz_label": "GMT/BST", "pre_market": "07:00", "after_hours": "17:00"},
            {"name": "NYSE", "open": "09:30", "close": "16:00", "tz": "America/New_York",
             "tz_label": "ET", "pre_market": "04:00", "after_hours": "20:00"},
            {"name": "NASDAQ", "open": "09:30", "close": "16:00", "tz": "America/New_York",
             "tz_label": "ET", "pre_market": "04:00", "after_hours": "20:00"},
            {"name": "Euronext", "open": "09:00", "close": "17:30", "tz": "Europe/Paris",
             "tz_label": "CET", "pre_market": "07:15", "after_hours": "17:30"},
            {"name": "Xetra", "open": "09:00", "close": "17:30", "tz": "Europe/Berlin",
             "tz_label": "CET", "pre_market": "08:00", "after_hours": "17:30"},
        ],
        "notes": "Closed weekends and public holidays. Most liquid during first and last hour of each session.",
    },
    "forex": {
        "label": "Forex",
        "description": "Major and minor currency pairs",
        "currency": "USD (quoted)",
        "schedule": "24 hours, Mon-Fri",
        "sessions": [
            {"name": "Sydney", "open": "22:00", "close": "07:00", "tz": "UTC",
             "tz_label": "UTC", "volatility": "low"},
            {"name": "Tokyo", "open": "00:00", "close": "09:00", "tz": "UTC",
             "tz_label": "UTC", "volatility": "medium"},
            {"name": "London", "open": "08:00", "close": "17:00", "tz": "UTC",
             "tz_label": "UTC", "volatility": "high"},
            {"name": "New York", "open": "13:00", "close": "22:00", "tz": "UTC",
             "tz_label": "UTC", "volatility": "high"},
        ],
        "notes": "Most liquid during London-NY overlap (13:00-17:00 UTC). "
                 "Closed from Friday 22:00 UTC to Sunday 22:00 UTC.",
    },
    "commodity": {
        "label": "Commodities",
        "description": "Futures contracts for metals, energy, and agriculture",
        "currency": "USD",
        "schedule": "Near 24h, Mon-Fri",
        "sessions": [
            {"name": "CME Globex", "open": "18:00", "close": "17:00", "tz": "America/Chicago",
             "tz_label": "CT", "note": "60-min break 17:00-18:00 CT"},
            {"name": "Metals (COMEX)", "open": "18:00", "close": "17:00", "tz": "America/New_York",
             "tz_label": "ET", "note": "Gold, Silver, Copper"},
            {"name": "Energy (NYMEX)", "open": "18:00", "close": "17:00", "tz": "America/New_York",
             "tz_label": "ET", "note": "Crude Oil, Natural Gas"},
            {"name": "Agriculture (CBOT)", "open": "19:00", "close": "13:20", "tz": "America/Chicago",
             "tz_label": "CT", "note": "Wheat, Corn, Soybeans"},
        ],
        "notes": "Futures trade nearly 24h with a daily 60-min maintenance break. "
                 "Most volume during US morning session.",
    },
    "crypto": {
        "label": "Crypto",
        "description": "Cryptocurrencies traded against USD",
        "currency": "USD",
        "schedule": "24/7",
        "sessions": [
            {"name": "Global", "open": "00:00", "close": "23:59", "tz": "UTC",
             "tz_label": "UTC", "note": "Never closes"},
        ],
        "notes": "Trades 24/7/365 with no market close. Highest volatility during US market hours. "
                 "Weekend liquidity tends to be lower.",
    },
}


# Budget
INITIAL_BUDGET_GBP = float(os.getenv("STOCKIO_BUDGET", "500.00"))

# Alpaca (live trading)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# OANDA (forex trading)
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_PRACTICE = os.getenv("OANDA_PRACTICE", "true").lower() in ("true", "1", "yes")

# Trading mode — auto-detect "live" when broker keys are present
_explicit_mode = os.getenv("STOCKIO_MODE")
if _explicit_mode:
    MODE = _explicit_mode
elif ALPACA_API_KEY and ALPACA_SECRET_KEY:
    MODE = "live"  # Alpaca keys present → use AlpacaExecutor
elif OANDA_API_KEY and OANDA_ACCOUNT_ID:
    MODE = "live"  # OANDA keys present → use OandaExecutor for forex
else:
    MODE = "paper"

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
# Forex (currency pairs)
# ---------------------------------------------------------------------------

FOREX_ENABLED = os.getenv("STOCKIO_FOREX_ENABLED", "true").lower() in (
    "true", "1", "yes",
)

# Major and minor forex pairs (Yahoo Finance format: EURUSD=X)
FOREX_PAIRS = [
    p.strip()
    for p in os.getenv(
        "STOCKIO_FOREX_PAIRS",
        "EURUSD=X,GBPUSD=X,USDJPY=X,USDCHF=X,AUDUSD=X,USDCAD=X,NZDUSD=X,"
        "EURGBP=X,EURJPY=X,GBPJPY=X,EURCHF=X,AUDJPY=X,EURAUD=X,GBPCHF=X",
    ).split(",")
    if p.strip()
]

# Risk management for forex (more conservative — leveraged markets)
FOREX_MAX_POSITION_PCT = float(os.getenv("STOCKIO_FOREX_MAX_POSITION_PCT", "10"))
FOREX_STOP_LOSS_PCT = float(os.getenv("STOCKIO_FOREX_STOP_LOSS_PCT", "2"))
FOREX_TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_FOREX_TAKE_PROFIT_PCT", "5"))

# ---------------------------------------------------------------------------
# Commodities (gold, silver, oil, etc.)
# ---------------------------------------------------------------------------

COMMODITIES_ENABLED = os.getenv("STOCKIO_COMMODITIES_ENABLED", "true").lower() in (
    "true", "1", "yes",
)

# Major commodities (Yahoo Finance futures format: GC=F)
COMMODITY_SYMBOLS = [
    s.strip()
    for s in os.getenv(
        "STOCKIO_COMMODITY_SYMBOLS",
        "GC=F,SI=F,CL=F,NG=F,HG=F,PL=F,PA=F,ZW=F,ZC=F,ZS=F,KC=F,CT=F",
    ).split(",")
    if s.strip()
]

# Friendly names for display
COMMODITY_NAMES: dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil (WTI)",
    "NG=F": "Natural Gas",
    "HG=F": "Copper",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
    "ZW=F": "Wheat",
    "ZC=F": "Corn",
    "ZS=F": "Soybeans",
    "KC=F": "Coffee",
    "CT=F": "Cotton",
}

# Risk management for commodities
COMMODITY_MAX_POSITION_PCT = float(os.getenv("STOCKIO_COMMODITY_MAX_POSITION_PCT", "15"))
COMMODITY_STOP_LOSS_PCT = float(os.getenv("STOCKIO_COMMODITY_STOP_LOSS_PCT", "4"))
COMMODITY_TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_COMMODITY_TAKE_PROFIT_PCT", "10"))

# ---------------------------------------------------------------------------
# Cryptocurrency
# ---------------------------------------------------------------------------

CRYPTO_ENABLED = os.getenv("STOCKIO_CRYPTO_ENABLED", "true").lower() in (
    "true", "1", "yes",
)

# Major crypto (Yahoo Finance format: BTC-USD)
CRYPTO_SYMBOLS = [
    s.strip()
    for s in os.getenv(
        "STOCKIO_CRYPTO_SYMBOLS",
        "BTC-USD,ETH-USD,BNB-USD,SOL-USD,XRP-USD,ADA-USD,DOGE-USD,AVAX-USD,"
        "DOT-USD,MATIC-USD,LINK-USD,UNI-USD,LTC-USD,ATOM-USD",
    ).split(",")
    if s.strip()
]

# Friendly names for display
CRYPTO_NAMES: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
    "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "DOT-USD": "Polkadot",
    "MATIC-USD": "Polygon",
    "LINK-USD": "Chainlink",
    "UNI-USD": "Uniswap",
    "LTC-USD": "Litecoin",
    "ATOM-USD": "Cosmos",
}

# Risk management for crypto (wider thresholds — high volatility)
CRYPTO_MAX_POSITION_PCT = float(os.getenv("STOCKIO_CRYPTO_MAX_POSITION_PCT", "10"))
CRYPTO_STOP_LOSS_PCT = float(os.getenv("STOCKIO_CRYPTO_STOP_LOSS_PCT", "8"))
CRYPTO_TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_CRYPTO_TAKE_PROFIT_PCT", "20"))

# Crypto-specific subreddits
CRYPTO_SUBREDDITS = [
    s.strip()
    for s in os.getenv(
        "STOCKIO_CRYPTO_SUBREDDITS",
        "CryptoCurrency,Bitcoin,ethereum,CryptoMarkets,altcoin",
    ).split(",")
    if s.strip()
]

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
# Trump / Political monitoring
# ---------------------------------------------------------------------------

TRUMP_MONITORING_ENABLED = os.getenv("STOCKIO_TRUMP_MONITORING", "true").lower() in (
    "true", "1", "yes",
)

# How much extra weight Trump/political stories get vs normal broad market
# This multiplier is applied ON TOP of normal weighting because these events
# tend to cause immediate, outsized market moves (tariffs, executive orders, etc.)
TRUMP_WEIGHT = float(os.getenv("STOCKIO_TRUMP_WEIGHT", "1.5"))

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

INTERVAL_MINUTES = int(os.getenv("STOCKIO_INTERVAL_MINUTES", "30"))
RETRAIN_HOURS = int(os.getenv("STOCKIO_RETRAIN_HOURS", "24"))

# Risk management — long positions
MAX_POSITION_PCT = float(os.getenv("STOCKIO_MAX_POSITION_PCT", "20"))
STOP_LOSS_PCT = float(os.getenv("STOCKIO_STOP_LOSS_PCT", "5"))
TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_TAKE_PROFIT_PCT", "15"))

# Risk management — short positions (betting against)
SHORT_SELLING_ENABLED = os.getenv("STOCKIO_SHORT_SELLING", "true").lower() in (
    "true", "1", "yes",
)
MAX_SHORT_POSITION_PCT = float(os.getenv("STOCKIO_MAX_SHORT_POSITION_PCT", "15"))
SHORT_STOP_LOSS_PCT = float(os.getenv("STOCKIO_SHORT_STOP_LOSS_PCT", "5"))
SHORT_TAKE_PROFIT_PCT = float(os.getenv("STOCKIO_SHORT_TAKE_PROFIT_PCT", "10"))
MAX_TOTAL_SHORT_PCT = float(os.getenv("STOCKIO_MAX_TOTAL_SHORT_PCT", "30"))

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


# ---------------------------------------------------------------------------
# Asset type helpers
# ---------------------------------------------------------------------------


def get_asset_type(ticker: str) -> AssetType:
    """Determine the asset type from a ticker symbol."""
    if ticker in FOREX_PAIRS or ticker.endswith("=X"):
        return AssetType.FOREX
    if ticker in COMMODITY_SYMBOLS or (ticker.endswith("=F") and ticker not in FOREX_PAIRS):
        return AssetType.COMMODITY
    if ticker in CRYPTO_SYMBOLS or (
        "-USD" in ticker and not ticker.endswith(".L") and not ticker.endswith("=X")
    ):
        return AssetType.CRYPTO
    return AssetType.EQUITY


def get_risk_params(asset_type: AssetType) -> dict:
    """Return risk management parameters for a given asset type."""
    if asset_type == AssetType.FOREX:
        return {
            "max_position_pct": FOREX_MAX_POSITION_PCT,
            "stop_loss_pct": FOREX_STOP_LOSS_PCT,
            "take_profit_pct": FOREX_TAKE_PROFIT_PCT,
        }
    if asset_type == AssetType.COMMODITY:
        return {
            "max_position_pct": COMMODITY_MAX_POSITION_PCT,
            "stop_loss_pct": COMMODITY_STOP_LOSS_PCT,
            "take_profit_pct": COMMODITY_TAKE_PROFIT_PCT,
        }
    if asset_type == AssetType.CRYPTO:
        return {
            "max_position_pct": CRYPTO_MAX_POSITION_PCT,
            "stop_loss_pct": CRYPTO_STOP_LOSS_PCT,
            "take_profit_pct": CRYPTO_TAKE_PROFIT_PCT,
        }
    # Default: equity
    return {
        "max_position_pct": MAX_POSITION_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
    }


def get_asset_display_name(ticker: str) -> str:
    """Return a friendly display name for a ticker."""
    if ticker in COMMODITY_NAMES:
        return COMMODITY_NAMES[ticker]
    if ticker in CRYPTO_NAMES:
        return CRYPTO_NAMES[ticker]
    # Forex: convert EURUSD=X → EUR/USD
    if ticker.endswith("=X") and len(ticker) == 8:
        base = ticker[:3]
        quote = ticker[3:6]
        return f"{base}/{quote}"
    return ticker
