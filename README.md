# Stockio — AI Stock Trading Bot

An AI-powered stock trading bot that monitors markets, analyses news sentiment, and makes intelligent buy/sell/short decisions to grow an initial £500 investment.

## How It Works

Stockio runs a continuous trading loop that:

1. **Fetches live market data** — OHLCV prices via Yahoo Finance for your watchlist
2. **Computes technical indicators** — RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR, OBV, and more
3. **Analyses news sentiment** — Fetches RSS headlines from financial outlets, Google News, and Reddit; scores them with FinBERT (a financial sentiment transformer)
4. **Monitors Trump/political activity** — Tracks White House executive orders, tariffs, and trade policy via dedicated feeds with configurable weighting (default 1.5x) for outsized market impact
5. **Generates ML trade signals** — A Gradient Boosting classifier trained on technical indicators predicts price direction; this is combined with sentiment to produce BUY/SELL/SHORT/COVER/HOLD signals
6. **Executes trades** — Paper trading by default (simulated with real prices), with Alpaca broker integration ready for live trading. Supports both long positions (buy low, sell high) and short positions (bet against a stock)
7. **Manages risk** — Position sizing limits, stop-loss, take-profit, and a hard safety guard that automatically covers short positions before the portfolio can go negative
8. **Learns over time** — The ML model automatically retrains on fresh data every 24 hours, improving predictions as it accumulates more market history

## Architecture

```
src/stockio/
├── config.py            # Central configuration (env vars, paths, defaults)
├── market_discovery.py  # Multi-market stock discovery and caching
├── market_data.py       # Yahoo Finance data + technical indicators
├── sentiment.py         # RSS/Reddit/Trump feed fetching + FinBERT sentiment scoring
├── strategy.py          # ML model training, prediction, signal generation
├── portfolio.py         # SQLite portfolio tracker, risk management, P&L
├── executor.py          # Paper & Alpaca trade execution
├── bot.py               # Main orchestrator — ties everything together
├── webapp.py            # Flask web dashboard + REST API
├── templates/           # HTML dashboard template
└── cli.py               # Click CLI entry point

deploy/
├── setup.sh              # One-command LXC container setup
├── update.sh             # Deploy new code to an existing install
├── stockio-web.service   # systemd unit for the web dashboard
└── stockio-bot.service   # systemd unit for the trading bot
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (optional — defaults work for paper trading)
cp .env.example .env
# Edit .env to customise watchlist, budget, etc.

# 3. Train the ML model
stockio train

# 4. Start the web dashboard
stockio web

# Or run the bot headless
stockio run
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `stockio web --port 5000` | Start the web dashboard |
| `stockio run` | Start the trading bot loop (headless) |
| `stockio train --period 2y` | Manually retrain the ML model |
| `stockio status` | Show current portfolio (cash, holdings, P&L) |
| `stockio history --limit 20` | Show recent trade history |
| `stockio signals` | Dry-run: show what the bot would trade right now |
| `stockio backtest --ticker AAPL --period 1y` | Backtest the strategy on historical data |

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCKIO_BUDGET` | `500.00` | Initial budget in GBP |
| `STOCKIO_MODE` | `paper` | `paper` (simulated) or `live` (real money) |
| `STOCKIO_WATCHLIST` | `AAPL,MSFT,...` | Comma-separated tickers to monitor |
| `STOCKIO_INTERVAL_MINUTES` | `30` | Trading loop frequency |
| `STOCKIO_RETRAIN_HOURS` | `24` | How often to retrain the model |
| `STOCKIO_MAX_POSITION_PCT` | `20` | Max % of portfolio in one position |
| `STOCKIO_STOP_LOSS_PCT` | `5` | Stop-loss threshold (%) |
| `STOCKIO_TAKE_PROFIT_PCT` | `15` | Take-profit threshold (%) |
| `STOCKIO_SHORT_SELLING` | `true` | Enable short selling (betting against stocks) |
| `STOCKIO_MAX_SHORT_POSITION_PCT` | `15` | Max % of portfolio in a single short |
| `STOCKIO_SHORT_STOP_LOSS_PCT` | `5` | Short stop-loss threshold (%) |
| `STOCKIO_SHORT_TAKE_PROFIT_PCT` | `10` | Short take-profit threshold (%) |
| `STOCKIO_MAX_TOTAL_SHORT_PCT` | `30` | Max total short exposure as % of portfolio |
| `STOCKIO_TRUMP_MONITORING` | `true` | Enable Trump/political feed monitoring |
| `STOCKIO_TRUMP_WEIGHT` | `1.5` | Extra weight multiplier for Trump/political stories |
| `STOCKIO_REDDIT_WEIGHT` | `0.3` | Weight for Reddit sentiment (0.0–1.0) |

## Deploy to LXC Container

Run the setup script as root on a fresh Debian/Ubuntu LXC container:

```bash
# From the repo root
sudo bash deploy/setup.sh
```

This will:
- Install Python and dependencies
- Create a `stockio` service user and add your user to its group
- Install the app to `/opt/stockio`
- Set up two systemd services (web dashboard + trading bot)
- Start the web dashboard on port 5000

**Log out and back in** after setup so the group membership takes effect, then train the model and start trading:

```bash
# Train the ML model (run as your own user)
stockio train --period 2y

# Start the trading bot
sudo systemctl start stockio-bot
```

```bash
# Manage the services
sudo systemctl start stockio-bot       # Start trading
sudo systemctl stop stockio-bot        # Stop trading
sudo systemctl status stockio-web      # Check web status
sudo journalctl -u stockio-bot -f      # Follow bot logs
sudo journalctl -u stockio-web -f      # Follow web logs
```

The web dashboard is available at `http://<container-ip>:5000` and lets you start/stop the bot, view portfolio status, see live trade signals, monitor Trump/political activity, and review trade history.

### Updating

After pulling new code into your local repo, deploy it to the container:

```bash
# From the repo root
sudo bash deploy/update.sh
```

This copies the updated source, installs any new dependencies, and restarts the services. Your `.env` config and database are preserved.

## Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

## Disclaimer

This software is for educational and research purposes. Algorithmic trading involves substantial risk. Never trade with money you cannot afford to lose. Past performance does not guarantee future results.
