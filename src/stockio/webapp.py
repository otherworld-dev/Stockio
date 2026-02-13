"""Flask web application — dashboard for monitoring and controlling Stockio."""

from __future__ import annotations

import datetime as dt
import subprocess
import threading

from flask import Flask, jsonify, render_template, request

from stockio import __version__, config
from stockio.config import get_logger
from stockio.market_data import get_current_prices
from stockio.market_discovery import (
    SUPPORTED_MARKETS,
    get_cached_tickers,
    get_market_summary,
    get_ticker_count,
    refresh_all_markets,
)
from stockio.portfolio import (
    get_bot_logs,
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
# systemd helpers
# ------------------------------------------------------------------


def _try_systemctl(action: str) -> bool:
    """Try to start/stop the bot via systemd. Returns True if it worked.

    Uses sudo with a passwordless rule installed by setup.sh so the
    stockio service user can manage the bot service.
    """
    if action not in ("start", "stop", "restart"):
        return False
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", action, "stockio-bot"],
            check=True, capture_output=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _systemd_bot_running() -> bool:
    """Check if the bot systemd service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "stockio-bot"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
        # Use held positions for status (not the full universe)
        held_tickers = [p.ticker for p in get_positions()]
        prices = get_current_prices(held_tickers) if held_tickers else {}
        summary = portfolio_summary(prices)
        summary["bot_running"] = _bot_running or _systemd_bot_running()
        summary["mode"] = config.MODE
        summary["markets"] = config.MARKETS
        summary["total_tickers"] = get_ticker_count()
        summary["batch_size"] = config.BATCH_SIZE
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
    """Generate and return current trade signals for held positions + top tickers."""
    try:
        from stockio.strategy import generate_signals

        # Use held positions + a small sample from the universe
        held = [p.ticker for p in get_positions()]
        all_tickers = get_cached_tickers()
        sample = all_tickers[:20]  # top 20 by market cap

        combined = list(dict.fromkeys(held + sample))  # deduplicate, preserve order
        prices = get_current_prices(combined)
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
    """Start the trading bot — prefers systemd, falls back to in-process thread."""
    global _bot_thread, _bot_instance, _bot_running

    # Try systemd first (works when deployed)
    if _try_systemctl("start"):
        return jsonify({"status": "started", "via": "systemd"})

    # Fall back to in-process thread (dev mode)
    if _bot_running:
        return jsonify({"status": "already_running"})

    from stockio.bot import StockioBot

    _bot_instance = StockioBot()
    _bot_thread = threading.Thread(target=_run_bot, daemon=True)
    _bot_thread.start()
    _bot_running = True
    return jsonify({"status": "started", "via": "thread"})


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    """Stop the trading bot."""
    global _bot_running

    if _try_systemctl("stop"):
        return jsonify({"status": "stopped", "via": "systemd"})

    _bot_running = False
    return jsonify({"status": "stopped", "via": "thread"})


@app.route("/api/bot-log")
def api_bot_log():
    """Return recent bot reasoning logs."""
    limit = request.args.get("limit", 5, type=int)
    logs = get_bot_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/snapshots")
def api_snapshots():
    """Return portfolio value snapshots for charting."""
    limit = request.args.get("limit", 500, type=int)
    snapshots = get_snapshots(limit=limit)
    return jsonify(snapshots)


@app.route("/api/markets")
def api_markets():
    """Return market discovery status and ticker counts."""
    try:
        summary = get_market_summary()
        return jsonify({
            "configured_markets": config.MARKETS,
            "supported_markets": {
                k: {"name": v["name"], "region": v["region"], "currency": v["currency"]}
                for k, v in SUPPORTED_MARKETS.items()
            },
            "cached_markets": summary,
            "total_tickers": get_ticker_count(),
            "batch_size": config.BATCH_SIZE,
            "extra_watchlist": config.WATCHLIST,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/markets/refresh", methods=["POST"])
def api_markets_refresh():
    """Force-refresh all configured market ticker caches."""
    try:
        results = refresh_all_markets()
        return jsonify({
            "status": "refreshed",
            "results": results,
            "total_tickers": get_ticker_count(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sentiment-detail")
def api_sentiment_detail():
    """Run sentiment analysis on-demand and return full breakdown.

    Returns per-ticker sentiment with source breakdown (news vs Reddit vs
    broad market) and per-article details including individual sentiment scores.
    """
    try:
        from stockio.sentiment import get_sentiment_scores

        # Use held positions + top tickers by market cap
        held = [p.ticker for p in get_positions()]
        all_tickers = get_cached_tickers()
        sample = all_tickers[:15]  # top 15 by market cap
        combined = list(dict.fromkeys(held + sample))

        prices = get_current_prices(combined)
        tickers = list(prices.keys())

        sentiments = get_sentiment_scores(tickers)

        result = []
        for ticker, sent in sentiments.items():
            if sent.num_articles == 0 and sent.market_sentiment == 0.0:
                continue
            result.append({
                "ticker": ticker,
                "score": sent.score,
                "direction": "bullish" if sent.score > 0.05 else "bearish" if sent.score < -0.05 else "neutral",
                "num_articles": sent.num_articles,
                "news_score": sent.news_score,
                "reddit_score": sent.reddit_score,
                "trump_score": sent.trump_score,
                "news_count": sent.news_count,
                "reddit_count": sent.reddit_count,
                "trump_count": sent.trump_count,
                "broad_count": sent.broad_count,
                "market_sentiment": sent.market_sentiment,
                "headlines": sent.headlines,
                "articles": sent.articles,
            })

        # Sort by most articles first, then by absolute score
        result.sort(key=lambda x: (-x["num_articles"], -abs(x["score"])))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/news-feed")
def api_news_feed():
    """Return the latest news and Reddit posts from the most recent bot cycle.

    Extracts article details from the bot log so no additional API calls needed.
    """
    try:
        logs = get_bot_logs(limit=1)
        if not logs:
            return jsonify([])

        entries = logs[0].get("entries", [])
        articles = []
        seen_titles: set = set()

        for entry in entries:
            if entry.get("type") != "sentiment":
                continue
            ticker = entry.get("ticker", "")
            if ticker == "_MARKET":
                continue

            for article in entry.get("articles", []):
                title = article.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                articles.append({
                    "title": title,
                    "source": article.get("source", ""),
                    "link": article.get("link", ""),
                    "match_type": article.get("match_type", ""),
                    "matched_ticker": ticker,
                    "sentiment": article.get("sentiment", 0),
                    "label": article.get("label", ""),
                })

        # Sort: Reddit first (more interesting), then by absolute sentiment
        articles.sort(key=lambda a: (
            0 if a["source"].startswith("reddit/") else 1,
            -abs(a["sentiment"]),
        ))

        return jsonify(articles[:100])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/config")
def api_config():
    """Return current configuration (non-sensitive)."""
    return jsonify({
        "mode": config.MODE,
        "budget": config.INITIAL_BUDGET_GBP,
        "markets": config.MARKETS,
        "watchlist": config.WATCHLIST,
        "total_tickers": get_ticker_count(),
        "batch_size": config.BATCH_SIZE,
        "include_penny_stocks": config.INCLUDE_PENNY_STOCKS,
        "interval_minutes": config.INTERVAL_MINUTES,
        "retrain_hours": config.RETRAIN_HOURS,
        "market_refresh_hours": config.MARKET_REFRESH_HOURS,
        "max_position_pct": config.MAX_POSITION_PCT,
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "take_profit_pct": config.TAKE_PROFIT_PCT,
        "reddit_enabled": config.REDDIT_ENABLED,
        "reddit_subreddits": config.REDDIT_SUBREDDITS,
        "reddit_weight": config.REDDIT_WEIGHT,
        "trump_monitoring_enabled": config.TRUMP_MONITORING_ENABLED,
        "trump_weight": config.TRUMP_WEIGHT,
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
