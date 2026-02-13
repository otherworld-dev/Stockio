# Stockio — AI Stock Trading Bot

An AI-powered stock trading bot that monitors markets, analyses news sentiment, and makes intelligent buy/sell decisions to grow an initial £500 investment.

## How It Works

Stockio runs a continuous trading loop that:

1. **Fetches live market data** — OHLCV prices via Yahoo Finance for your watchlist
2. **Computes technical indicators** — RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR, OBV, and more
3. **Analyses news sentiment** — Fetches RSS headlines and scores them with FinBERT (a financial sentiment transformer)
4. **Generates ML trade signals** — A Gradient Boosting classifier trained on technical indicators predicts price direction; this is combined with sentiment to produce BUY/SELL/HOLD signals
5. **Executes trades** — Paper trading by default (simulated with real prices), with Alpaca broker integration ready for live trading
6. **Manages risk** — Position sizing limits, stop-loss, and take-profit rules protect the portfolio
7. **Learns over time** — The ML model automatically retrains on fresh data every 24 hours, improving predictions as it accumulates more market history

## Architecture

```
src/stockio/
├── config.py       # Central configuration (env vars, paths, defaults)
├── market_data.py  # Yahoo Finance data + technical indicators
├── sentiment.py    # RSS news fetching + FinBERT sentiment scoring
├── strategy.py     # ML model training, prediction, signal generation
├── portfolio.py    # SQLite portfolio tracker, risk management, P&L
├── executor.py     # Paper & Alpaca trade execution
├── bot.py          # Main orchestrator — ties everything together
└── cli.py          # Click CLI entry point
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

# 4. Run the bot
stockio run
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `stockio run` | Start the trading bot loop |
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

## Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

## Disclaimer

This software is for educational and research purposes. Algorithmic trading involves substantial risk. Never trade with money you cannot afford to lose. Past performance does not guarantee future results.
