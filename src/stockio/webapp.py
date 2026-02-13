"""Flask web application — dashboard for monitoring and controlling Stockio."""

from __future__ import annotations

import datetime as dt
import threading

from flask import Flask, jsonify, render_template, request

from stockio import __version__, config
from stockio.config import get_logger
from stockio.market_data import get_current_prices
from stockio.portfolio import (
    get_cash,
    get_positions,
    get_snapshots,
    get_trade_history,
    portfolio_summary,
)

log = get_logger(__name__)

app = Flask(
    __name__,
    template_folder=str(config.PROJECT_ROOT / "src" / "stockio" / "templates"),
    static_folder=str(config.PROJECT_ROOT / "src" / "stockio" / "static"),
)

# Reference to the bot thread (set by run_webapp)
_bot_thread: threading.Thread | None = None
_bot_instance = None
_bot_running = False


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html", version=__version__)


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------


@app.route("/api/status")
def api_status():
    """Return current portfolio status as JSON."""
    try:
        prices = get_current_prices(config.WATCHLIST)
        summary = portfolio_summary(prices)
        summary["bot_running"] = _bot_running
        summary["mode"] = config.MODE
        summary["watchlist"] = config.WATCHLIST
        summary["timestamp"] = dt.datetime.utcnow().isoformat()
        return jsonify(summary)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/trades")
def api_trades():
    """Return recent trade history as JSON."""
    limit = request.args.get("limit", 50, type=int)
    trades = get_trade_history(limit=limit)
    return jsonify([
        {
            "id": t.id,
            "ticker": t.ticker,
            "side": t.side,
            "shares": t.shares,
            "price": round(t.price, 2),
            "total": round(t.total, 2),
            "timestamp": t.timestamp,
            "reason": t.reason,
        }
        for t in trades
    ])


@app.route("/api/signals")
def api_signals():
    """Generate and return current trade signals."""
    try:
        from stockio.strategy import generate_signals

        prices = get_current_prices(config.WATCHLIST)
        tickers = list(prices.keys())

        # Try sentiment — fall back gracefully
        sentiments = {}
        try:
            from stockio.sentiment import get_sentiment_scores
            sentiments = get_sentiment_scores(tickers)
        except Exception:
            pass

        signals = generate_signals(tickers, sentiments=sentiments)
        return jsonify([
            {
                "ticker": s.ticker,
                "signal": s.signal.value,
                "confidence": round(s.confidence, 4),
                "reasons": s.reasons,
            }
            for s in signals
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    """Start the trading bot in a background thread."""
    global _bot_thread, _bot_instance, _bot_running

    if _bot_running:
        return jsonify({"status": "already_running"})

    from stockio.bot import StockioBot

    _bot_instance = StockioBot()
    _bot_thread = threading.Thread(target=_run_bot, daemon=True)
    _bot_thread.start()
    _bot_running = True
    return jsonify({"status": "started"})


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    """Signal the bot to stop."""
    global _bot_running
    _bot_running = False
    return jsonify({"status": "stopped"})


@app.route("/api/snapshots")
def api_snapshots():
    """Return portfolio value snapshots for charting."""
    limit = request.args.get("limit", 500, type=int)
    snapshots = get_snapshots(limit=limit)
    return jsonify(snapshots)


@app.route("/api/config")
def api_config():
    """Return current configuration (non-sensitive)."""
    return jsonify({
        "mode": config.MODE,
        "budget": config.INITIAL_BUDGET_GBP,
        "watchlist": config.WATCHLIST,
        "interval_minutes": config.INTERVAL_MINUTES,
        "retrain_hours": config.RETRAIN_HOURS,
        "max_position_pct": config.MAX_POSITION_PCT,
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "take_profit_pct": config.TAKE_PROFIT_PCT,
    })


# ------------------------------------------------------------------
# Bot runner (in thread)
# ------------------------------------------------------------------


def _run_bot():
    global _bot_running
    import time
    import traceback

    log.info("Bot thread started")
    try:
        while _bot_running:
            try:
                _bot_instance.run_cycle()
            except Exception:
                log.error("Cycle error:\n%s", traceback.format_exc())
            # Wait for the interval, checking stop flag every 10s
            for _ in range(config.INTERVAL_MINUTES * 6):
                if not _bot_running:
                    break
                time.sleep(10)
    finally:
        _bot_running = False
        log.info("Bot thread stopped")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run_webapp(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the Flask web app."""
    app.run(host=host, port=port, debug=debug)
