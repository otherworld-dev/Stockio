"""Main bot orchestrator — ties all modules together into a trading loop.

The bot:
  1. Fetches current prices for the watchlist
  2. Checks existing positions for stop-loss / take-profit exits
  3. Gathers news and scores sentiment
  4. Generates ML + sentiment trade signals
  5. Executes trades via the configured executor
  6. Logs a portfolio summary
  7. Periodically retrains the ML model on fresh data
"""

from __future__ import annotations

import datetime as dt
import time
import traceback

import schedule

from stockio import config
from stockio.config import get_logger
from stockio.executor import get_executor
from stockio.market_data import get_current_prices
from stockio.portfolio import get_positions, portfolio_summary
from stockio.sentiment import get_sentiment_scores
from stockio.strategy import Signal, generate_signals, train_model

log = get_logger(__name__)


class StockioBot:
    """The main trading bot."""

    def __init__(self) -> None:
        self.executor = get_executor()
        self.watchlist = list(config.WATCHLIST)
        self._last_retrain: dt.datetime | None = None

    # ------------------------------------------------------------------
    # Core trading cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        """Execute one full trading cycle."""
        log.info("=" * 60)
        log.info("Starting trading cycle at %s", dt.datetime.utcnow().isoformat())
        log.info("=" * 60)

        try:
            self._maybe_retrain()
            self._execute_cycle()
        except Exception:
            log.error("Trading cycle failed:\n%s", traceback.format_exc())

    def _execute_cycle(self) -> None:
        # 1. Fetch current prices
        log.info("Fetching prices for %d tickers ...", len(self.watchlist))
        prices = get_current_prices(self.watchlist)
        if not prices:
            log.warning("Could not fetch any prices — skipping cycle")
            return
        log.info("Got prices for %d / %d tickers", len(prices), len(self.watchlist))

        # 2. Check stop-loss / take-profit on existing positions
        for pos in get_positions():
            if pos.ticker in prices:
                self.executor.check_exits(pos.ticker, prices[pos.ticker])

        # 3. Sentiment analysis
        log.info("Analysing news sentiment ...")
        try:
            sentiments = get_sentiment_scores(list(prices.keys()))
        except Exception as exc:
            log.warning("Sentiment analysis failed (%s) — continuing without it", exc)
            sentiments = {}

        # 4. Generate signals
        log.info("Generating trade signals ...")
        signals = generate_signals(list(prices.keys()), sentiments=sentiments)

        # 5. Execute trades
        buy_count = sell_count = 0
        for sig in signals:
            if sig.ticker not in prices:
                continue
            price = prices[sig.ticker]
            trade = self.executor.execute(sig, price)
            if trade is not None:
                if trade.side == "BUY":
                    buy_count += 1
                else:
                    sell_count += 1

        log.info("Executed %d buys, %d sells this cycle", buy_count, sell_count)

        # 6. Portfolio summary
        summary = portfolio_summary(prices)
        self._log_summary(summary)

    # ------------------------------------------------------------------
    # ML model retraining
    # ------------------------------------------------------------------

    def _maybe_retrain(self) -> None:
        now = dt.datetime.utcnow()
        if self._last_retrain is not None:
            elapsed = (now - self._last_retrain).total_seconds() / 3600
            if elapsed < config.RETRAIN_HOURS:
                return

        log.info("Retraining ML model ...")
        try:
            _, _, _, acc = train_model(self.watchlist)
            self._last_retrain = now
            log.info("Model retrained — CV accuracy: %.4f", acc)
        except Exception:
            log.error("Model training failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(summary: dict) -> None:
        log.info("-" * 50)
        log.info("PORTFOLIO SUMMARY")
        log.info("-" * 50)
        log.info(
            "Cash: £%.2f | Holdings: £%.2f | Total: £%.2f",
            summary["cash"],
            summary["holdings_value"],
            summary["total_value"],
        )
        log.info(
            "P&L: £%.2f (%.2f%%)",
            summary["total_pnl"],
            summary["total_pnl_pct"],
        )
        for h in summary["holdings"]:
            log.info(
                "  %s: %.4f shares @ £%.2f → £%.2f (P&L: £%.2f / %.2f%%)",
                h["ticker"],
                h["shares"],
                h["avg_cost"],
                h["market_value"],
                h["pnl"],
                h["pnl_pct"],
            )
        log.info("-" * 50)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the bot with a scheduled trading loop."""
        log.info("Stockio bot starting (mode=%s, interval=%dm)", config.MODE, config.INTERVAL_MINUTES)
        log.info("Watchlist: %s", ", ".join(self.watchlist))
        log.info("Budget: £%.2f", config.INITIAL_BUDGET_GBP)

        # Run immediately on start
        self.run_cycle()

        # Schedule recurring runs
        schedule.every(config.INTERVAL_MINUTES).minutes.do(self.run_cycle)

        log.info("Scheduler started — next run in %d minutes", config.INTERVAL_MINUTES)
        try:
            while True:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("Bot stopped by user")
