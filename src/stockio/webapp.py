"""Flask web application — dashboard for monitoring and controlling Stockio.

Supports running multiple independent bot instances simultaneously (e.g. a
paper simulation *and* a live-trading bot, each with its own database).
"""

from __future__ import annotations

import datetime as dt
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    get_setting,
    get_snapshots,
    get_trade_history,
    portfolio_summary,
    reset_all_data,
    set_setting,
    use_db,
)

log = get_logger(__name__)

app = Flask(
    __name__,
    template_folder=str(config.PROJECT_ROOT / "src" / "stockio" / "templates"),
    static_folder=str(config.PROJECT_ROOT / "src" / "stockio" / "static"),
)

# Pre-load the FinBERT model at import time.  With gunicorn --preload this
# runs once in the master process before forking, so the ~440 MB model is
# loaded into memory only once and shared with the worker via copy-on-write.
try:
    from stockio.sentiment import warmup_model
    warmup_model()
except Exception:
    pass  # logged inside warmup_model

# Restore saved trading mode from the database (survives restarts).
try:
    saved_mode = get_setting("trading_mode")
    if saved_mode in ("paper", "live"):
        config.MODE = saved_mode
except Exception:
    pass  # DB might not exist yet on first run


# ---------------------------------------------------------------------------
# Multi-instance bot management
# ---------------------------------------------------------------------------


@dataclass
class BotSlot:
    """One independently-running bot instance."""

    name: str                                   # "paper" or "live"
    mode: str                                   # "paper" or "live"
    db_path: Path = field(default_factory=lambda: config.DB_PATH)
    thread: threading.Thread | None = None
    bot: Any = None                             # StockioBot
    running: bool = False
    generation: int = 0


# Two fixed slots — both can run simultaneously
_slots: dict[str, BotSlot] = {
    "paper": BotSlot(
        name="paper",
        mode="paper",
        db_path=config.get_db_path("paper"),
    ),
    "live": BotSlot(
        name="live",
        mode="live",
        db_path=config.get_db_path("live"),
    ),
}



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


@app.route("/health")
def health():
    """Lightweight health check — no DB, no model, no imports."""
    return jsonify({"status": "ok", "version": __version__})


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html", version=__version__)


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------


@app.route("/api/status")
def api_status():
    """Return current portfolio status as JSON.

    Query params:
        instance  – ``paper`` or ``live`` (default: ``paper``).
    """
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    try:
        with use_db(slot.db_path):
            # In live mode, pull data directly from Alpaca instead of local DB
            if slot.mode == "live" and config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
                summary = _alpaca_portfolio_summary()
            else:
                held_tickers = [p.ticker for p in get_positions()]
                prices = get_current_prices(held_tickers) if held_tickers else {}
                summary = portfolio_summary(prices)

        summary["bot_running"] = slot.running or _systemd_bot_running()
        summary["mode"] = slot.mode
        summary["instance"] = slot.name
        summary["executor"] = (type(slot.bot.executor).__name__
                                if slot.bot else "none")
        summary["alpaca_paper"] = "paper" in config.ALPACA_BASE_URL
        summary["alpaca_keys_set"] = bool(config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY)
        summary["oanda_keys_set"] = bool(config.OANDA_API_KEY and config.OANDA_ACCOUNT_ID)
        summary["markets"] = config.MARKETS
        summary["total_tickers"] = get_ticker_count()
        summary["batch_size"] = config.BATCH_SIZE
        summary["timestamp"] = dt.datetime.utcnow().isoformat()

        # Include summary of all instances
        summary["instances"] = {
            name: {"running": s.running, "mode": s.mode}
            for name, s in _slots.items()
        }

        return jsonify(summary)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _alpaca_portfolio_summary() -> dict:
    """Build a portfolio summary dict from Alpaca account data.

    Returns the same shape as portfolio_summary() so the frontend
    works identically regardless of data source.
    """
    from alpaca.trading.client import TradingClient

    is_paper = "paper" in config.ALPACA_BASE_URL
    client = TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=is_paper,
    )

    acct = client.get_account()
    positions = client.get_all_positions()

    equity = float(acct.equity)
    cash = float(acct.cash)
    long_value = float(acct.long_market_value)
    short_value = abs(float(acct.short_market_value))
    # Alpaca doesn't track an "initial budget" — use last_equity as baseline
    initial = float(acct.last_equity) if acct.last_equity else equity

    holdings = []
    for p in positions:
        qty = float(p.qty)
        avg_cost = float(p.avg_entry_price)
        price = float(p.current_price)
        mkt_val = float(p.market_value)
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        direction = "short" if qty < 0 else "long"

        holdings.append({
            "ticker": p.symbol,
            "shares": abs(qty),
            "avg_cost": round(avg_cost, 2),
            "current_price": round(price, 2),
            "market_value": round(mkt_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "direction": direction,
            "asset_type": "equity",
            "display_name": p.symbol,
        })

    num_long = sum(1 for h in holdings if h["direction"] != "short")
    num_short = sum(1 for h in holdings if h["direction"] == "short")
    total_pnl = equity - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial else 0

    return {
        "cash": round(cash, 2),
        "holdings_value": round(equity - cash, 2),
        "long_value": round(long_value, 2),
        "short_value": round(short_value, 2),
        "num_long": num_long,
        "num_short": num_short,
        "total_value": round(equity, 2),
        "initial_budget": round(initial, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "num_positions": len(holdings),
        "holdings": holdings,
    }


@app.route("/api/trades")
def api_trades():
    """Return recent trade history as JSON.

    In live mode, fetches recent orders from Alpaca so the dashboard
    reflects real broker activity rather than stale local paper trades.

    Query params:
        instance  – ``paper`` or ``live`` (default: ``paper``).
    """
    limit = request.args.get("limit", 50, type=int)
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])

    if slot.mode == "live" and config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        try:
            return jsonify(_alpaca_trade_history(limit))
        except Exception as exc:
            log.warning("Failed to fetch Alpaca orders, falling back to local: %s", exc)

    with use_db(slot.db_path):
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


def _alpaca_trade_history(limit: int = 50) -> list[dict]:
    """Fetch recent filled orders from Alpaca."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    is_paper = "paper" in config.ALPACA_BASE_URL
    client = TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=is_paper,
    )

    req = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=limit,
    )
    orders = client.get_orders(req)

    result = []
    for o in orders:
        if o.filled_qty and float(o.filled_qty) > 0:
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price) if o.filled_avg_price else 0
            result.append({
                "id": str(o.id),
                "ticker": o.symbol,
                "side": o.side.value,
                "shares": qty,
                "price": round(price, 2),
                "total": round(qty * price, 2),
                "timestamp": o.filled_at.isoformat() if o.filled_at else o.submitted_at.isoformat(),
                "reason": f"Alpaca order ({o.type.value})",
            })
    return result


@app.route("/api/signals")
def api_signals():
    """Generate and return current trade signals for held positions + top tickers."""
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    try:
        from stockio.strategy import generate_signals

        # Use held positions + a small sample from the universe
        with use_db(slot.db_path):
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

        with use_db(slot.db_path):
            positions_map = {
                p.ticker: p.direction for p in get_positions()
                if p.ticker in prices
            }
        signals = generate_signals(tickers, sentiments=sentiments, positions=positions_map)
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


@app.route("/api/instances")
def api_instances():
    """List all bot instances and their status."""
    return jsonify({
        name: {
            "running": s.running,
            "mode": s.mode,
            "executor": type(s.bot.executor).__name__ if s.bot else "none",
            "db": str(s.db_path),
        }
        for name, s in _slots.items()
    })


@app.route("/api/instances/<name>/start", methods=["POST"])
def api_instance_start(name: str):
    """Start a specific bot instance (paper or live)."""
    if name not in _slots:
        return jsonify({"error": f"Unknown instance: {name}"}), 400

    slot = _slots[name]

    if slot.mode == "live":
        has_alpaca = bool(config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY)
        has_oanda = bool(config.OANDA_API_KEY and config.OANDA_ACCOUNT_ID)
        if not has_alpaca and not has_oanda:
            return jsonify({
                "error": "Cannot start live instance — "
                         "set ALPACA_API_KEY/ALPACA_SECRET_KEY and/or "
                         "OANDA_API_KEY/OANDA_ACCOUNT_ID"
            }), 400

    if slot.running:
        return jsonify({"status": "already_running", "instance": name})

    try:
        from stockio.bot import StockioBot
        slot.bot = StockioBot(db_path=slot.db_path, mode=slot.mode)
    except Exception as exc:
        log.exception("Failed to initialise %s bot", name)
        return jsonify({"error": str(exc)}), 500

    slot.generation += 1
    slot.running = True
    slot.thread = threading.Thread(
        target=_run_instance, args=(slot,), daemon=True,
    )
    slot.thread.start()
    executor_name = type(slot.bot.executor).__name__
    log.info("Started %s instance (mode=%s, executor=%s)", name, slot.mode, executor_name)
    return jsonify({
        "status": "started", "instance": name,
        "mode": slot.mode, "executor": executor_name,
    })


@app.route("/api/instances/<name>/stop", methods=["POST"])
def api_instance_stop(name: str):
    """Stop a specific bot instance."""
    if name not in _slots:
        return jsonify({"error": f"Unknown instance: {name}"}), 400

    slot = _slots[name]
    slot.generation += 1
    slot.running = False
    log.info("Stopping %s instance", name)
    return jsonify({"status": "stopped", "instance": name})


# Legacy endpoints — map to the instance system
@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    """Start the trading bot (legacy endpoint).

    If systemd is available, tries that first.  Otherwise starts the
    instance matching the current ``config.MODE``.
    """
    if _try_systemctl("start"):
        return jsonify({"status": "started", "via": "systemd"})

    instance = "live" if config.MODE == "live" else "paper"
    return api_instance_start(instance)


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    """Stop the trading bot (legacy endpoint)."""
    if _try_systemctl("stop"):
        return jsonify({"status": "stopped", "via": "systemd"})

    # Stop whichever instances are running
    for slot in _slots.values():
        if slot.running:
            slot.generation += 1
            slot.running = False
    return jsonify({"status": "stopped"})


@app.route("/api/mode", methods=["POST"])
def api_mode():
    """Switch trading mode (legacy endpoint).

    Updates the global config.MODE and persists it.  Does NOT stop running
    instances — they keep their original mode.
    """
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "").lower()
    if mode not in ("paper", "live"):
        return jsonify({"error": "Invalid mode — must be 'paper' or 'live'"}), 400

    has_alpaca = bool(config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY)
    has_oanda = bool(config.OANDA_API_KEY and config.OANDA_ACCOUNT_ID)
    if mode == "live" and not has_alpaca and not has_oanda:
        return jsonify({
            "error": "Cannot switch to live mode — "
                     "set ALPACA_API_KEY/ALPACA_SECRET_KEY and/or "
                     "OANDA_API_KEY/OANDA_ACCOUNT_ID"
        }), 400

    config.MODE = mode
    set_setting("trading_mode", mode)
    log.info("Trading mode switched to: %s", mode)
    return jsonify({
        "status": "ok", "mode": mode,
        "instances": {
            name: {"running": s.running, "mode": s.mode}
            for name, s in _slots.items()
        },
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all portfolio data for a specific instance.

    Query params:
        instance  – ``paper`` or ``live`` (default: ``paper``).
    """
    instance = (request.get_json(silent=True) or {}).get(
        "instance", request.args.get("instance", "paper"),
    )
    slot = _slots.get(instance, _slots["paper"])

    # Stop the instance if running
    if slot.running:
        slot.generation += 1
        slot.running = False
        if slot.thread is not None:
            slot.thread.join(timeout=30)
    if _systemd_bot_running():
        _try_systemctl("stop")
        import time
        time.sleep(2)

    try:
        with use_db(slot.db_path):
            reset_all_data()
        return jsonify({
            "status": "reset",
            "instance": instance,
            "cash": config.INITIAL_BUDGET_GBP,
        })
    except Exception as exc:
        log.exception("Reset failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/bot-log")
def api_bot_log():
    """Return recent bot reasoning logs."""
    limit = request.args.get("limit", 5, type=int)
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    with use_db(slot.db_path):
        logs = get_bot_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/snapshots")
def api_snapshots():
    """Return portfolio value snapshots for charting."""
    limit = request.args.get("limit", 500, type=int)
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    with use_db(slot.db_path):
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
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    try:
        from stockio.sentiment import get_sentiment_scores

        # Use held positions + top tickers by market cap
        with use_db(slot.db_path):
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
        log.exception("Sentiment detail request failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/news-feed")
def api_news_feed():
    """Return the latest news and Reddit posts from the most recent bot cycle.

    Extracts article details from the bot log so no additional API calls needed.
    """
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    try:
        with use_db(slot.db_path):
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


@app.route("/api/trump-feed")
def api_trump_feed():
    """Return Trump/political stories from the most recent bot cycle.

    Filters the news feed for trump-related items (match_type=trump or
    source in truth_social, white_house, trump_news).  If the bot hasn't
    completed a cycle yet, fetches the feeds live so the tab isn't empty.
    """
    instance = request.args.get("instance", "paper")
    slot = _slots.get(instance, _slots["paper"])
    try:
        if not config.TRUMP_MONITORING_ENABLED:
            return jsonify({
                "enabled": False,
                "weight": config.TRUMP_WEIGHT,
                "items": [],
            })

        with use_db(slot.db_path):
            logs = get_bot_logs(limit=1)
        items = []
        seen_titles: set = set()

        if logs:
            entries = logs[0].get("entries", [])
            for entry in entries:
                if entry.get("type") != "sentiment":
                    continue
                ticker = entry.get("ticker", "")

                for article in entry.get("articles", []):
                    title = article.get("title", "")
                    if title in seen_titles:
                        continue
                    # Include if it's a trump match_type or from a trump source
                    is_trump = (
                        article.get("match_type") == "trump"
                        or article.get("source") in ("truth_social", "white_house", "trump_news")
                    )
                    if not is_trump:
                        continue
                    seen_titles.add(title)
                    items.append({
                        "title": title,
                        "source": article.get("source", ""),
                        "link": article.get("link", ""),
                        "match_type": article.get("match_type", ""),
                        "matched_ticker": ticker if ticker != "_MARKET" else "",
                        "sentiment": article.get("sentiment", 0),
                        "label": article.get("label", ""),
                    })

        # Fallback: if no bot cycle has completed yet, fetch feeds live so
        # the Trump tab isn't blank on first load / after an OOM restart.
        if not items:
            from stockio.sentiment import fetch_trump_feeds
            for news_item in fetch_trump_feeds():
                if news_item.title not in seen_titles:
                    seen_titles.add(news_item.title)
                    items.append({
                        "title": news_item.title,
                        "source": news_item.source,
                        "link": news_item.link,
                        "match_type": "trump",
                        "matched_ticker": "",
                        "sentiment": 0,
                        "label": "pending",
                    })

        # Sort by absolute sentiment (most impactful first)
        items.sort(key=lambda a: -abs(a["sentiment"]))

        return jsonify({
            "enabled": True,
            "weight": config.TRUMP_WEIGHT,
            "items": items[:50],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/alpaca")
def api_alpaca():
    """Return Alpaca account status and positions (read-only).

    Works regardless of MODE — just needs valid API keys.
    """
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        return jsonify({"connected": False, "error": "No Alpaca API keys configured"})

    try:
        from alpaca.trading.client import TradingClient

        is_paper = "paper" in config.ALPACA_BASE_URL
        client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=is_paper,
        )

        acct = client.get_account()
        positions = client.get_all_positions()

        pos_list = []
        for p in positions:
            qty = float(p.qty)
            market_value = float(p.market_value)
            unrealized_pl = float(p.unrealized_pl)
            unrealized_plpc = float(p.unrealized_plpc) * 100
            pos_list.append({
                "ticker": p.symbol,
                "qty": abs(qty),
                "direction": "short" if qty < 0 else "long",
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": abs(market_value),
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            })

        return jsonify({
            "connected": True,
            "paper": is_paper,
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "long_market_value": float(acct.long_market_value),
            "short_market_value": float(acct.short_market_value),
            "positions": pos_list,
            "timestamp": dt.datetime.utcnow().isoformat(),
        })
    except Exception as exc:
        return jsonify({"connected": False, "error": str(exc)})


@app.route("/api/oanda")
def api_oanda():
    """Return OANDA account status and positions (read-only)."""
    if not config.OANDA_API_KEY or not config.OANDA_ACCOUNT_ID:
        return jsonify({"connected": False, "error": "No OANDA API keys configured"})

    try:
        import oandapyV20
        from oandapyV20.endpoints.accounts import AccountDetails

        env = "practice" if config.OANDA_PRACTICE else "live"
        client = oandapyV20.API(
            access_token=config.OANDA_API_KEY,
            environment=env,
        )
        r = AccountDetails(config.OANDA_ACCOUNT_ID)
        client.request(r)
        acct = r.response.get("account", {})

        balance = float(acct.get("balance", 0))
        nav = float(acct.get("NAV", 0))
        unrealised_pl = float(acct.get("unrealizedPL", 0))
        open_trade_count = int(acct.get("openTradeCount", 0))

        positions = []
        for p in acct.get("positions", []):
            long_units = int(float(p.get("long", {}).get("units", 0)))
            short_units = int(float(p.get("short", {}).get("units", 0)))
            if long_units == 0 and short_units == 0:
                continue

            if long_units > 0:
                avg_price = float(p["long"].get("averagePrice", 0))
                pl = float(p["long"].get("unrealizedPL", 0))
                positions.append({
                    "instrument": p["instrument"],
                    "direction": "long",
                    "units": long_units,
                    "avg_price": avg_price,
                    "unrealized_pl": pl,
                })
            if short_units < 0:
                avg_price = float(p["short"].get("averagePrice", 0))
                pl = float(p["short"].get("unrealizedPL", 0))
                positions.append({
                    "instrument": p["instrument"],
                    "direction": "short",
                    "units": abs(short_units),
                    "avg_price": avg_price,
                    "unrealized_pl": pl,
                })

        return jsonify({
            "connected": True,
            "practice": config.OANDA_PRACTICE,
            "balance": balance,
            "nav": nav,
            "unrealised_pl": unrealised_pl,
            "open_trades": open_trade_count,
            "positions": positions,
            "timestamp": dt.datetime.utcnow().isoformat(),
        })
    except Exception as exc:
        return jsonify({"connected": False, "error": str(exc)})


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
        # Equity risk params
        "max_position_pct": config.MAX_POSITION_PCT,
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "take_profit_pct": config.TAKE_PROFIT_PCT,
        # Multi-asset settings
        "forex_enabled": config.FOREX_ENABLED,
        "forex_pairs": len(config.FOREX_PAIRS),
        "forex_risk": {
            "max_position_pct": config.FOREX_MAX_POSITION_PCT,
            "stop_loss_pct": config.FOREX_STOP_LOSS_PCT,
            "take_profit_pct": config.FOREX_TAKE_PROFIT_PCT,
        },
        "commodities_enabled": config.COMMODITIES_ENABLED,
        "commodity_symbols": len(config.COMMODITY_SYMBOLS),
        "commodity_risk": {
            "max_position_pct": config.COMMODITY_MAX_POSITION_PCT,
            "stop_loss_pct": config.COMMODITY_STOP_LOSS_PCT,
            "take_profit_pct": config.COMMODITY_TAKE_PROFIT_PCT,
        },
        "crypto_enabled": config.CRYPTO_ENABLED,
        "crypto_symbols": len(config.CRYPTO_SYMBOLS),
        "crypto_risk": {
            "max_position_pct": config.CRYPTO_MAX_POSITION_PCT,
            "stop_loss_pct": config.CRYPTO_STOP_LOSS_PCT,
            "take_profit_pct": config.CRYPTO_TAKE_PROFIT_PCT,
        },
        # Social / news
        "reddit_enabled": config.REDDIT_ENABLED,
        "reddit_subreddits": config.REDDIT_SUBREDDITS,
        "reddit_weight": config.REDDIT_WEIGHT,
        "trump_monitoring_enabled": config.TRUMP_MONITORING_ENABLED,
        "trump_weight": config.TRUMP_WEIGHT,
    })


# ------------------------------------------------------------------
# Bot runner (in thread)
# ------------------------------------------------------------------


def _run_instance(slot: BotSlot):
    """Thread target for running a single bot instance."""
    import time
    import traceback

    gen = slot.generation
    log.info("[%s] Bot thread started (generation %d, executor=%s)",
             slot.name, gen, type(slot.bot.executor).__name__)
    try:
        while slot.running and gen == slot.generation:
            try:
                slot.bot.run_cycle()
            except Exception:
                log.error("[%s] Cycle error:\n%s", slot.name, traceback.format_exc())
            # Wait for the interval, checking stop flag every 10s
            for _ in range(config.INTERVAL_MINUTES * 6):
                if not slot.running or gen != slot.generation:
                    break
                time.sleep(10)
    finally:
        if gen == slot.generation:
            slot.running = False
        log.info("[%s] Bot thread stopped (generation %d)", slot.name, gen)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run_webapp(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the Flask web app."""
    app.run(host=host, port=port, debug=debug)
