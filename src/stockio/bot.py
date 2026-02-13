"""Main bot orchestrator — ties all modules together into a trading loop.

The bot:
  1. Refreshes the market ticker cache (if stale)
  2. Selects the next batch of tickers to analyse (rotating through all)
  3. Fetches current prices for the batch
  4. Checks existing positions for stop-loss / take-profit exits
  5. Gathers news and scores sentiment
  6. Generates ML + sentiment trade signals
  7. Executes trades via the configured executor
  8. Logs a portfolio summary
  9. Periodically retrains the ML model on fresh data
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
from stockio.market_discovery import (
    get_cached_tickers,
    get_ticker_count,
    maybe_refresh,
)
from stockio.portfolio import get_positions, portfolio_summary, record_bot_log, record_snapshot
from stockio.sentiment import get_sentiment_scores
from stockio.strategy import Signal, generate_signals, train_model

log = get_logger(__name__)


class StockioBot:
    """The main trading bot."""

    def __init__(self) -> None:
        self.executor = get_executor()
        self._last_retrain: dt.datetime | None = None
        self._batch_offset: int = 0

    # ------------------------------------------------------------------
    # Core trading cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        """Execute one full trading cycle."""
        log.info("=" * 60)
        log.info("Starting trading cycle at %s", dt.datetime.utcnow().isoformat())
        log.info("=" * 60)

        try:
            self._maybe_refresh_markets()
            self._maybe_retrain()
            self._execute_cycle()
        except Exception:
            log.error("Trading cycle failed:\n%s", traceback.format_exc())

    def _maybe_refresh_markets(self) -> None:
        """Refresh market ticker caches if they are stale."""
        if not config.MARKETS:
            return
        try:
            maybe_refresh()
        except Exception:
            log.error("Market refresh failed:\n%s", traceback.format_exc())

    def _get_batch(self) -> list[str]:
        """Return the next batch of tickers to analyse.

        Rotates through the full universe across cycles.
        Always includes tickers for which we hold positions.
        """
        all_tickers = get_cached_tickers()

        if not all_tickers:
            log.warning("No tickers available — check market config or STOCKIO_WATCHLIST")
            return []

        total = len(all_tickers)
        batch_size = config.BATCH_SIZE

        # If the universe is small enough, just use all of them
        if total <= batch_size:
            log.info(
                "Universe has %d tickers (batch_size=%d) — analysing all",
                total, batch_size,
            )
            self._batch_offset = 0
            return all_tickers

        # Get the rotating batch
        start = self._batch_offset % total
        batch_tickers = []

        # Slice from the sorted universe
        if start + batch_size <= total:
            batch_tickers = all_tickers[start : start + batch_size]
        else:
            # Wrap around
            batch_tickers = all_tickers[start:] + all_tickers[: batch_size - (total - start)]

        # Always include tickers we currently hold positions in
        held_tickers = {pos.ticker for pos in get_positions()}
        for t in held_tickers:
            if t not in batch_tickers:
                batch_tickers.append(t)

        # Advance the offset for next cycle
        self._batch_offset = (start + batch_size) % total

        cycles_for_full = (total + batch_size - 1) // batch_size
        log.info(
            "Batch %d/%d: analysing %d tickers (of %d total, full rotation every %d cycles)",
            (start // batch_size) + 1,
            cycles_for_full,
            len(batch_tickers),
            total,
            cycles_for_full,
        )

        return batch_tickers

    def _execute_cycle(self) -> None:
        cycle_log: list[dict] = []

        # 1. Get the batch of tickers for this cycle
        batch = self._get_batch()
        if not batch:
            return

        # 2. Fetch current prices
        log.info("Fetching prices for %d tickers ...", len(batch))
        prices = get_current_prices(batch)
        if not prices:
            log.warning("Could not fetch any prices — skipping cycle")
            return
        log.info("Got prices for %d / %d tickers", len(prices), len(batch))

        # 3. Check stop-loss / take-profit on existing positions
        for pos in get_positions():
            if pos.ticker in prices:
                result = self.executor.check_exits(pos.ticker, prices[pos.ticker])
                if result is not None:
                    cycle_log.append({
                        "type": "exit",
                        "ticker": pos.ticker,
                        "side": result.side,
                        "reason": result.reason,
                    })

        # 4. Sentiment analysis
        log.info("Analysing news sentiment ...")
        try:
            sentiments = get_sentiment_scores(list(prices.keys()))
        except Exception as exc:
            log.warning("Sentiment analysis failed (%s) — continuing without it", exc)
            sentiments = {}

        if sentiments:
            # Log broad market sentiment once (from the first ticker that has it)
            market_sent_logged = False
            for sent in sentiments.values():
                if not market_sent_logged and sent.market_sentiment != 0.0:
                    mkt_dir = "bullish" if sent.market_sentiment > 0.05 else "bearish" if sent.market_sentiment < -0.05 else "neutral"
                    log.info(
                        "  BROAD MARKET sentiment: %+.4f (%s)",
                        sent.market_sentiment, mkt_dir,
                    )
                    cycle_log.append({
                        "type": "sentiment",
                        "ticker": "_MARKET",
                        "score": sent.market_sentiment,
                        "direction": mkt_dir,
                        "num_articles": 0,
                        "headlines": [],
                    })
                    market_sent_logged = True
                    break

            for ticker, sent in sentiments.items():
                if sent.num_articles > 0:
                    direction = "bullish" if sent.score > 0.05 else "bearish" if sent.score < -0.05 else "neutral"
                    log.info(
                        "  %s sentiment: %+.4f (%s, %d articles)",
                        ticker, sent.score, direction, sent.num_articles,
                    )
                    for hl in sent.headlines[:3]:
                        log.info("    - %s", hl)
                    cycle_log.append({
                        "type": "sentiment",
                        "ticker": ticker,
                        "score": sent.score,
                        "direction": direction,
                        "num_articles": sent.num_articles,
                        "headlines": sent.headlines[:3],
                    })

        # 5. Generate signals
        log.info("Generating trade signals ...")
        signals = generate_signals(list(prices.keys()), sentiments=sentiments)

        # 5b. Log the bot's reasoning for each ticker
        log.info("-" * 50)
        log.info("SIGNAL ANALYSIS")
        log.info("-" * 50)
        for sig in signals:
            price = prices.get(sig.ticker)
            price_str = f" @ £{price:.2f}" if price is not None else ""
            log.info(
                "  %s%s → %s (confidence=%.2f)",
                sig.ticker, price_str, sig.signal.value, sig.confidence,
            )
            for reason in sig.reasons:
                log.info("    • %s", reason)
            cycle_log.append({
                "type": "signal",
                "ticker": sig.ticker,
                "signal": sig.signal.value,
                "confidence": round(sig.confidence, 4),
                "price": round(price, 2) if price is not None else None,
                "reasons": sig.reasons,
            })
        log.info("-" * 50)

        # 6. Execute trades
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
                cycle_log.append({
                    "type": "trade",
                    "ticker": trade.ticker,
                    "side": trade.side,
                    "shares": round(trade.shares, 4),
                    "price": round(trade.price, 2),
                    "total": round(trade.total, 2),
                    "reason": trade.reason,
                })

        log.info("Executed %d buys, %d sells this cycle", buy_count, sell_count)

        # 7. Portfolio summary + snapshot for charts
        summary = portfolio_summary(prices)
        self._log_summary(summary)
        record_snapshot(prices)

        # 8. Save the cycle reasoning log
        record_bot_log(cycle_log)

    # ------------------------------------------------------------------
    # ML model retraining
    # ------------------------------------------------------------------

    def _maybe_retrain(self) -> None:
        now = dt.datetime.utcnow()
        if self._last_retrain is not None:
            elapsed = (now - self._last_retrain).total_seconds() / 3600
            if elapsed < config.RETRAIN_HOURS:
                return

        # For training, use a representative sample of tickers from each market
        # (training on thousands of tickers would be too slow)
        training_tickers = self._get_training_tickers()

        log.info("Retraining ML model on %d tickers ...", len(training_tickers))
        try:
            _, _, _, acc = train_model(training_tickers)
            self._last_retrain = now
            log.info("Model retrained — CV accuracy: %.4f", acc)
        except Exception:
            log.error("Model training failed:\n%s", traceback.format_exc())

    def _get_training_tickers(self) -> list[str]:
        """Select a representative sample of tickers for model training.

        Uses top tickers by market cap from each market (up to 50 per market)
        plus any held positions.
        """
        all_tickers = get_cached_tickers()
        max_training = 100  # reasonable limit for training speed

        if len(all_tickers) <= max_training:
            return all_tickers

        # Take the top N by market cap (they're already sorted by market cap)
        training = all_tickers[:max_training]

        # Ensure held positions are included
        for pos in get_positions():
            if pos.ticker not in training:
                training.append(pos.ticker)

        return training

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
        total_tickers = get_ticker_count()
        markets_str = ", ".join(config.MARKETS) if config.MARKETS else "none (watchlist only)"

        log.info("Stockio bot starting (mode=%s, interval=%dm)", config.MODE, config.INTERVAL_MINUTES)
        log.info("Markets: %s", markets_str)
        log.info("Total tickers in universe: %d", total_tickers)
        log.info("Batch size: %d tickers per cycle", config.BATCH_SIZE)
        log.info("Budget: £%.2f", config.INITIAL_BUDGET_GBP)

        if config.WATCHLIST:
            log.info("Additional watchlist: %s", ", ".join(config.WATCHLIST))

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
