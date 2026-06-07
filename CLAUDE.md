# Stockio — Automated Forex Trading Bot

## What This Is
A multi-strategy automated forex trading bot that runs on OANDA practice/live accounts. Uses LightGBM ML model + Claude Haiku for sentiment analysis and trade advisory. Features a Flask web dashboard for monitoring and control.

## Architecture

### Two Services
- **stockio** (`stockio run`) — headless CLI bot (not used when web dashboard is running)
- **stockio-web** (`stockio web`) — Flask dashboard + bot threads. This is what runs in production.

### Bot Instances (run simultaneously as threads in stockio-web)
| Slot | Strategy | OANDA Account | Notes |
|------|----------|---------------|-------|
| paper | ML (LightGBM) | Practice primary | Also fetches shared sentiment for all bots |
| live | Auto (best paper) | Live account | Auto-selects #1 paper strategy each cycle |
| trend | Consensus | Practice sub-account | Only trades when 2+ strategies agree |
| meanrev | Mean Reversion | Practice sub-account | RSI extremes + trend confirmation |
| momentum | Best Signal | Practice sub-account | Picks highest confidence from all 3 scorers |
| llm | LLM-Only | Practice sub-account | Claude decides all trades directly |

Strategy overrides are in `bot_manager.py`:
```python
_STRATEGY_OVERRIDE = {"trend": "consensus", "momentum": "best_signal"}
```

### Scoring Strategies (in `strategy/strategies.py`)
- `score_trend` — EMA alignment + ADX/MACD, RSI guard at extremes
- `score_meanrev` — RSI <35/>65 + Bollinger/Stochastic, blocks falling knives
- `score_momentum` — ADX >25 gate + MACD/EMA agreement
- `score_consensus` — Runs all 3 above, only trades when 2+ agree
- `score_best_signal` — Runs all 3, picks single highest confidence
- `LLMScorer` — Claude Haiku batch-scores all instruments per cycle

### Sentiment Sharing
Paper bot fetches news sentiment (NewsAPI + RSS + Claude) every hour. Strategy bots read from shared state — no duplicate API calls. This means **paper must be running** for strategies to get sentiment data.

### LLM Advisor (separate from LLM strategy)
Each bot has its own LLM advisor (`llm_advisor.py`) that can veto trades based on market regime assessment. The advisor sees shadow outcomes (vetoed trades that would have won/lost) as feedback. Both the LLM scorer and advisor use `llm_strategy_model` (Sonnet 4.6) for better reasoning, while sentiment stays on `llm_model` (Haiku) to keep costs low. Both are self-adapting: they receive their own performance stats (win rate, P&L by instrument, losing streaks) in each prompt.

### Shadow Tracking
When trades are vetoed/skipped, phantom outcomes are recorded and resolved later to measure veto accuracy. Results feed back into the LLM advisor prompt. Data persists in `shadow_outcomes.parquet`.

## Project Structure
```
config/
  settings.toml          # Trading parameters, risk limits, indicators
  instruments.toml       # 14 instruments (11 forex + 3 commodities)
src/stockio/
  engine.py              # TradingEngine — main cycle loop
  config.py              # Settings (pydantic, loads from .env + settings.toml)
  db.py                  # SQLite per-instance (trades, snapshots, settings)
  broker/
    oanda.py             # OANDA v20 REST API
    yahoo.py             # Yahoo Finance fallback
  strategy/
    strategies.py        # All scoring functions + LLMScorer
    scorer.py            # InstrumentScorer (ML), OutcomeTracker, shadow tracking
    llm_advisor.py       # Claude trade advisor with veto power
    sentiment.py         # News sentiment (NewsAPI + RSS + Claude scoring)
    indicators.py        # Technical indicators (RSI, MACD, ADX, BB, EMA, etc.)
  web/
    app.py               # Flask routes (dashboard API)
    bot_manager.py       # Thread-based bot lifecycle management
  templates/
    dashboard.html       # Single-page dashboard (vanilla JS + Chart.js)
data/                    # SQLite DBs + parquet files (gitignored)
models/                  # LightGBM model files (gitignored)
scripts/deploy/          # systemd services + setup/update scripts
```

## Server Setup

### Prerequisites
- Python 3.12+
- Linux server (tested on Ubuntu in LXC container)
- OANDA practice + live accounts

### Installation
```bash
# On the server
sudo bash scripts/deploy/setup.sh
```

### Environment Variables (`.env` at `/opt/stockio/.env`)
```
# Practice account (paper bot + strategy sub-accounts share this token)
OANDA_PRACTICE_ACCOUNT_ID=101-004-XXXXXXXX-001
OANDA_PRACTICE_API_TOKEN=<practice-api-token>

# Live account (separate credentials)
OANDA_LIVE_ACCOUNT_ID=001-004-XXXXXXXX-003
OANDA_LIVE_API_TOKEN=<live-api-token>

# Strategy competition sub-accounts (all use practice API token)
OANDA_TREND_ACCOUNT_ID=101-004-XXXXXXXX-002
OANDA_MEANREV_ACCOUNT_ID=101-004-XXXXXXXX-003
OANDA_MOMENTUM_ACCOUNT_ID=101-004-XXXXXXXX-004
OANDA_LLM_ACCOUNT_ID=101-004-XXXXXXXX-005

# API keys
ANTHROPIC_API_KEY=<anthropic-key>
NEWS_API_KEY=<newsapi-key>

# Optional
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### Deployment
```bash
git pull && sudo bash scripts/deploy/update.sh
```

### Services
```bash
sudo systemctl start stockio-web    # Web dashboard + bot threads
sudo systemctl start stockio        # Headless CLI bot (alternative, don't run both)
sudo journalctl -u stockio-web -f   # View web dashboard logs
```

### Dashboard
- URL: `http://<server-ip>:5000`
- Paper tab: strategy competition leaderboard + all bot controls
- Live tab: single-bot view with auto-strategy selection, portfolio cards, session P&L

## Per-Strategy Settings
Each bot has independent settings stored in its own SQLite DB. Change via the gear icon on each leaderboard row, or via API:
```bash
curl -X POST "http://<server>:5000/api/settings?instance=meanrev" \
  -H "Content-Type: application/json" \
  -d '{"min_confidence": 0.25, "take_profit_atr_mult": 2.0, "risk_per_trade": 0.02}'
```

Key tunable parameters:
- `min_confidence` — minimum signal confidence to trade (0.0-1.0)
- `risk_per_trade` — fraction of equity risked per trade (0.01 = 1%)
- `stop_loss_atr_mult` — stop-loss distance in ATR multiples
- `take_profit_atr_mult` — take-profit distance in ATR multiples
- `max_positions` — max concurrent open trades
- `cycle_seconds` — how often the bot runs a trading cycle
- `granularity` — candle timeframe (M1, M5, M15, M30, H1, H4, D)

## Current Strategy Settings (as of May 2026)
| Strategy | min_confidence | risk_per_trade | TP mult | SL mult |
|----------|---------------|----------------|---------|---------|
| Meanrev | 0.25 | 0.02 | 2.0 | 1.5 |
| Trend (Consensus) | 0.30 | 0.02 | 2.5 | 1.5 |
| Best Signal | 0.30 | 0.01 | 3.0 | 1.5 |
| Paper (ML) | 0.40 | 0.01 | 3.0 | 1.5 |
| LLM | 0.45 | 0.01 | 3.0 | 1.5 |

## Trading Cycle (every 15 min)
1. Fetch candles for all 14 instruments from OANDA
2. Compute technical indicators (RSI, MACD, ADX, EMA, BB, Stoch, ATR)
3. Score instruments using the bot's strategy function
4. Rank by confidence, filter correlated pairs
5. Get LLM advisor approval/veto (with shadow tracking of vetoed trades)
6. Execute trades via OANDA API (with position sizing, SL/TP)
7. Check open positions for trailing stop updates
8. Resolve pending outcomes for model training
9. Record snapshot for equity chart

## Important Notes
- **JPY pairs** need 3 decimal places for prices (pip_size=0.01). Non-JPY use 5. The engine handles this via `pip_size` from instruments.toml.
- **Market hours**: Bots skip all activity Fri 22:00 → Sun 22:00 UTC (no API calls, no sentiment, saves costs)
- **DB settings override config**: Values saved via the dashboard settings modal take priority over settings.toml defaults
- **Parquet files**: `training_data.parquet` (ML training data) and `shadow_outcomes.parquet` (veto tracking) can corrupt. If `parquet_flush_failed` errors appear, move the file and restart.
- **Live bot requires paper bot running**: Live reads shared sentiment from paper. Without paper, live gets stale/no sentiment.
