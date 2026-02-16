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
            {"name": "LSE / AIM", "open": "08:00", "close": "16:30",
             "tz_label": "GMT", "pre_market": "07:00", "after_hours": "17:00"},
            {"name": "NYSE", "open": "14:30", "close": "21:00",
             "tz_label": "GMT", "pre_market": "09:00", "after_hours": "01:00"},
            {"name": "NASDAQ", "open": "14:30", "close": "21:00",
             "tz_label": "GMT", "pre_market": "09:00", "after_hours": "01:00"},
            {"name": "Euronext", "open": "08:00", "close": "16:30",
             "tz_label": "GMT", "pre_market": "06:15", "after_hours": "16:30"},
            {"name": "Xetra", "open": "08:00", "close": "16:30",
             "tz_label": "GMT", "pre_market": "07:00", "after_hours": "16:30"},
        ],
        "notes": "Closed weekends and public holidays. Most liquid during first and last hour of each session.",
    },
    "forex": {
        "label": "Forex",
        "description": "Major and minor currency pairs",
        "currency": "USD (quoted)",
        "schedule": "24 hours, Mon-Fri",
        "sessions": [
            {"name": "Sydney", "open": "22:00", "close": "07:00",
             "tz_label": "GMT", "volatility": "low"},
            {"name": "Tokyo", "open": "00:00", "close": "09:00",
             "tz_label": "GMT", "volatility": "medium"},
            {"name": "London", "open": "08:00", "close": "17:00",
             "tz_label": "GMT", "volatility": "high"},
            {"name": "New York", "open": "13:00", "close": "22:00",
             "tz_label": "GMT", "volatility": "high"},
        ],
        "notes": "Most liquid during London-NY overlap (13:00-17:00 GMT). "
                 "Closed from Friday 22:00 GMT to Sunday 22:00 GMT.",
    },
    "commodity": {
        "label": "Commodities",
        "description": "Futures contracts for metals, energy, and agriculture",
        "currency": "USD",
        "schedule": "Near 24h, Mon-Fri",
        "sessions": [
            {"name": "CME Globex", "open": "23:00", "close": "22:00",
             "tz_label": "GMT", "note": "60-min break 22:00-23:00 GMT"},
            {"name": "Metals (COMEX)", "open": "23:00", "close": "22:00",
             "tz_label": "GMT", "note": "Gold, Silver, Copper"},
            {"name": "Energy (NYMEX)", "open": "23:00", "close": "22:00",
             "tz_label": "GMT", "note": "Crude Oil, Natural Gas"},
            {"name": "Agriculture (CBOT)", "open": "01:00", "close": "19:20",
             "tz_label": "GMT", "note": "Wheat, Corn, Soybeans"},
        ],
        "notes": "Futures trade nearly 24h with a daily 60-min maintenance break. "
                 "Most volume during US afternoon GMT (14:00-21:00).",
    },
    "crypto": {
        "label": "Crypto",
        "description": "Cryptocurrencies traded against USD",
        "currency": "USD",
        "schedule": "24/7",
        "sessions": [
            {"name": "Global", "open": "00:00", "close": "23:59",
             "tz_label": "GMT", "note": "Never closes"},
        ],
        "notes": "Trades 24/7/365 with no market close. Highest volatility during US hours (14:00-21:00 GMT). "
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

# Simulated transaction costs (paper trading only).
# Spread is the half-spread applied to each side (buy pays +spread, sell receives -spread).
# Slippage models execution delay / market impact.  Commission is a flat % per trade.
# All values are percentages of the trade price (e.g. 0.05 = 0.05%).
EQUITY_SPREAD_PCT = float(os.getenv("STOCKIO_EQUITY_SPREAD_PCT", "0.05"))
EQUITY_SLIPPAGE_PCT = float(os.getenv("STOCKIO_EQUITY_SLIPPAGE_PCT", "0.02"))
EQUITY_COMMISSION_PCT = float(os.getenv("STOCKIO_EQUITY_COMMISSION_PCT", "0.0"))

FOREX_SPREAD_PCT = float(os.getenv("STOCKIO_FOREX_SPREAD_PCT", "0.01"))
FOREX_SLIPPAGE_PCT = float(os.getenv("STOCKIO_FOREX_SLIPPAGE_PCT", "0.005"))
FOREX_COMMISSION_PCT = float(os.getenv("STOCKIO_FOREX_COMMISSION_PCT", "0.0"))

COMMODITY_SPREAD_PCT = float(os.getenv("STOCKIO_COMMODITY_SPREAD_PCT", "0.05"))
COMMODITY_SLIPPAGE_PCT = float(os.getenv("STOCKIO_COMMODITY_SLIPPAGE_PCT", "0.03"))
COMMODITY_COMMISSION_PCT = float(os.getenv("STOCKIO_COMMODITY_COMMISSION_PCT", "0.0"))

CRYPTO_SPREAD_PCT = float(os.getenv("STOCKIO_CRYPTO_SPREAD_PCT", "0.10"))
CRYPTO_SLIPPAGE_PCT = float(os.getenv("STOCKIO_CRYPTO_SLIPPAGE_PCT", "0.05"))
CRYPTO_COMMISSION_PCT = float(os.getenv("STOCKIO_CRYPTO_COMMISSION_PCT", "0.0"))

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
            "spread_pct": FOREX_SPREAD_PCT,
            "slippage_pct": FOREX_SLIPPAGE_PCT,
            "commission_pct": FOREX_COMMISSION_PCT,
        }
    if asset_type == AssetType.COMMODITY:
        return {
            "max_position_pct": COMMODITY_MAX_POSITION_PCT,
            "stop_loss_pct": COMMODITY_STOP_LOSS_PCT,
            "take_profit_pct": COMMODITY_TAKE_PROFIT_PCT,
            "spread_pct": COMMODITY_SPREAD_PCT,
            "slippage_pct": COMMODITY_SLIPPAGE_PCT,
            "commission_pct": COMMODITY_COMMISSION_PCT,
        }
    if asset_type == AssetType.CRYPTO:
        return {
            "max_position_pct": CRYPTO_MAX_POSITION_PCT,
            "stop_loss_pct": CRYPTO_STOP_LOSS_PCT,
            "take_profit_pct": CRYPTO_TAKE_PROFIT_PCT,
            "spread_pct": CRYPTO_SPREAD_PCT,
            "slippage_pct": CRYPTO_SLIPPAGE_PCT,
            "commission_pct": CRYPTO_COMMISSION_PCT,
        }
    # Default: equity
    return {
        "max_position_pct": MAX_POSITION_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "spread_pct": EQUITY_SPREAD_PCT,
        "slippage_pct": EQUITY_SLIPPAGE_PCT,
        "commission_pct": EQUITY_COMMISSION_PCT,
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


# ---------------------------------------------------------------------------
# Settings registry — describes every tunable parameter for the UI
# ---------------------------------------------------------------------------

# Each entry: (module_attr, type, section, label, description)
# Types: "float", "int", "bool"
# Sections group settings in the UI.

SETTINGS_REGISTRY: list[dict] = [
    # ── Scheduling ────────────────────────────────────────────────────
    {"attr": "INTERVAL_MINUTES",  "type": "int",   "section": "Scheduling",            "label": "Trading interval (mins)",        "desc": "How often the bot runs a full analysis cycle"},
    {"attr": "RETRAIN_HOURS",     "type": "int",   "section": "Scheduling",            "label": "Model retrain interval (hrs)",   "desc": "How often to retrain the ML model"},
    {"attr": "BATCH_SIZE",        "type": "int",   "section": "Scheduling",            "label": "Tickers per cycle",              "desc": "How many tickers to analyse each cycle"},

    # ── Feature toggles ──────────────────────────────────────────────
    {"attr": "SHORT_SELLING_ENABLED",      "type": "bool", "section": "Feature Toggles",   "label": "Short selling",              "desc": "Allow the bot to open short positions"},
    {"attr": "INCLUDE_PENNY_STOCKS",       "type": "bool", "section": "Feature Toggles",   "label": "Include penny stocks",       "desc": "Include low-price stocks in analysis"},
    {"attr": "REDDIT_ENABLED",             "type": "bool", "section": "Feature Toggles",   "label": "Reddit sentiment",           "desc": "Monitor Reddit for sentiment signals"},
    {"attr": "TRUMP_MONITORING_ENABLED",   "type": "bool", "section": "Feature Toggles",   "label": "Trump/political monitoring", "desc": "Monitor political news for market impact"},

    # ── Sentiment weights ─────────────────────────────────────────────
    {"attr": "REDDIT_WEIGHT",     "type": "float", "section": "Sentiment Weights",     "label": "Reddit weight",                  "desc": "How much Reddit sentiment influences signals (0.0-1.0)"},
    {"attr": "REDDIT_MAX_POSTS",  "type": "int",   "section": "Sentiment Weights",     "label": "Reddit max posts",               "desc": "Max posts to fetch per subreddit"},
    {"attr": "TRUMP_WEIGHT",      "type": "float", "section": "Sentiment Weights",     "label": "Trump/political weight",         "desc": "Extra multiplier for political news impact"},

    # ── Equity risk ───────────────────────────────────────────────────
    {"attr": "MAX_POSITION_PCT",       "type": "float", "section": "Equity Risk",       "label": "Max position %",           "desc": "Max % of portfolio per equity position"},
    {"attr": "STOP_LOSS_PCT",          "type": "float", "section": "Equity Risk",       "label": "Stop-loss %",              "desc": "Auto-sell if price drops this much"},
    {"attr": "TAKE_PROFIT_PCT",        "type": "float", "section": "Equity Risk",       "label": "Take-profit %",            "desc": "Auto-sell if price rises this much"},

    # ── Short risk ────────────────────────────────────────────────────
    {"attr": "MAX_SHORT_POSITION_PCT", "type": "float", "section": "Short Selling",     "label": "Max short position %",     "desc": "Max % of portfolio per short position"},
    {"attr": "SHORT_STOP_LOSS_PCT",    "type": "float", "section": "Short Selling",     "label": "Short stop-loss %",        "desc": "Auto-cover if price rises this much"},
    {"attr": "SHORT_TAKE_PROFIT_PCT",  "type": "float", "section": "Short Selling",     "label": "Short take-profit %",      "desc": "Auto-cover if price drops this much"},
    {"attr": "MAX_TOTAL_SHORT_PCT",    "type": "float", "section": "Short Selling",     "label": "Max total short exposure %", "desc": "Max combined short exposure as % of portfolio"},

    # ── Forex risk ────────────────────────────────────────────────────
    {"attr": "FOREX_MAX_POSITION_PCT", "type": "float", "section": "Forex Risk",        "label": "Max position %",           "desc": "Max % of portfolio per forex position"},
    {"attr": "FOREX_STOP_LOSS_PCT",    "type": "float", "section": "Forex Risk",        "label": "Stop-loss %",              "desc": "Auto-close if price moves against by this much"},
    {"attr": "FOREX_TAKE_PROFIT_PCT",  "type": "float", "section": "Forex Risk",        "label": "Take-profit %",            "desc": "Auto-close if price moves in favour by this much"},

    # ── Commodity risk ────────────────────────────────────────────────
    {"attr": "COMMODITY_MAX_POSITION_PCT", "type": "float", "section": "Commodity Risk", "label": "Max position %",           "desc": "Max % of portfolio per commodity position"},
    {"attr": "COMMODITY_STOP_LOSS_PCT",    "type": "float", "section": "Commodity Risk", "label": "Stop-loss %",              "desc": "Auto-close if price drops this much"},
    {"attr": "COMMODITY_TAKE_PROFIT_PCT",  "type": "float", "section": "Commodity Risk", "label": "Take-profit %",            "desc": "Auto-close if price rises this much"},

    # ── Crypto risk ───────────────────────────────────────────────────
    {"attr": "CRYPTO_MAX_POSITION_PCT", "type": "float", "section": "Crypto Risk",      "label": "Max position %",           "desc": "Max % of portfolio per crypto position"},
    {"attr": "CRYPTO_STOP_LOSS_PCT",    "type": "float", "section": "Crypto Risk",      "label": "Stop-loss %",              "desc": "Auto-close if price drops this much"},
    {"attr": "CRYPTO_TAKE_PROFIT_PCT",  "type": "float", "section": "Crypto Risk",      "label": "Take-profit %",            "desc": "Auto-close if price rises this much"},

    # ── Equity transaction costs ──────────────────────────────────────
    {"attr": "EQUITY_SPREAD_PCT",     "type": "float", "section": "Equity Costs",       "label": "Spread %",                 "desc": "Simulated half-spread per trade"},
    {"attr": "EQUITY_SLIPPAGE_PCT",   "type": "float", "section": "Equity Costs",       "label": "Slippage %",               "desc": "Simulated execution slippage"},
    {"attr": "EQUITY_COMMISSION_PCT", "type": "float", "section": "Equity Costs",       "label": "Commission %",             "desc": "Commission fee per trade"},

    # ── Forex transaction costs ───────────────────────────────────────
    {"attr": "FOREX_SPREAD_PCT",     "type": "float", "section": "Forex Costs",         "label": "Spread %",                 "desc": "Simulated half-spread per trade"},
    {"attr": "FOREX_SLIPPAGE_PCT",   "type": "float", "section": "Forex Costs",         "label": "Slippage %",               "desc": "Simulated execution slippage"},
    {"attr": "FOREX_COMMISSION_PCT", "type": "float", "section": "Forex Costs",         "label": "Commission %",             "desc": "Commission fee per trade"},

    # ── Commodity transaction costs ───────────────────────────────────
    {"attr": "COMMODITY_SPREAD_PCT",     "type": "float", "section": "Commodity Costs",  "label": "Spread %",                "desc": "Simulated half-spread per trade"},
    {"attr": "COMMODITY_SLIPPAGE_PCT",   "type": "float", "section": "Commodity Costs",  "label": "Slippage %",              "desc": "Simulated execution slippage"},
    {"attr": "COMMODITY_COMMISSION_PCT", "type": "float", "section": "Commodity Costs",  "label": "Commission %",            "desc": "Commission fee per trade"},

    # ── Crypto transaction costs ──────────────────────────────────────
    {"attr": "CRYPTO_SPREAD_PCT",     "type": "float", "section": "Crypto Costs",       "label": "Spread %",                 "desc": "Simulated half-spread per trade"},
    {"attr": "CRYPTO_SLIPPAGE_PCT",   "type": "float", "section": "Crypto Costs",       "label": "Slippage %",               "desc": "Simulated execution slippage"},
    {"attr": "CRYPTO_COMMISSION_PCT", "type": "float", "section": "Crypto Costs",       "label": "Commission %",             "desc": "Commission fee per trade"},
]

# Build a quick lookup by attr name
_SETTINGS_BY_ATTR: dict[str, dict] = {s["attr"]: s for s in SETTINGS_REGISTRY}

# Ordered section names for consistent UI rendering
SETTINGS_SECTIONS: list[str] = list(dict.fromkeys(s["section"] for s in SETTINGS_REGISTRY))


def _cast(value: str, typ: str) -> float | int | bool:
    """Convert a string DB value to the correct Python type."""
    if typ == "bool":
        return value.lower() in ("true", "1", "yes")
    if typ == "int":
        return int(float(value))
    return float(value)


def get_all_settings() -> dict[str, dict]:
    """Return current values for all registered settings, grouped by section."""
    import sys
    mod = sys.modules[__name__]
    sections: dict[str, list[dict]] = {}
    for s in SETTINGS_REGISTRY:
        entry = {
            "attr": s["attr"],
            "label": s["label"],
            "desc": s["desc"],
            "type": s["type"],
            "value": getattr(mod, s["attr"]),
        }
        sections.setdefault(s["section"], []).append(entry)
    return sections


def apply_setting(attr: str, value: float | int | bool) -> bool:
    """Apply a single setting change to the running config module.

    Returns True if the attribute was recognised and updated.
    """
    import sys
    if attr not in _SETTINGS_BY_ATTR:
        return False
    mod = sys.modules[__name__]
    setattr(mod, attr, value)
    return True


def load_settings_from_db(db_get_setting) -> int:
    """Load any persisted setting overrides from the database.

    *db_get_setting* should be ``portfolio.get_setting`` (passed in to
    avoid circular imports).  Returns the number of overrides applied.
    """
    count = 0
    for s in SETTINGS_REGISTRY:
        raw = db_get_setting(f"cfg:{s['attr']}", "")
        if raw:
            try:
                apply_setting(s["attr"], _cast(raw, s["type"]))
                count += 1
            except (ValueError, TypeError):
                pass
    return count
