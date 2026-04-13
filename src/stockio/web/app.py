"""Flask web application — dashboard for monitoring and controlling Stockio."""

from __future__ import annotations

import contextlib
from pathlib import Path

import structlog
from flask import Flask, jsonify, render_template, request

from stockio import db
from stockio.config import load_settings
from stockio.web.bot_manager import get_slot, get_slots, init_slots, start_bot, stop_bot

log = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

app = Flask(
    __name__,
    template_folder=str(_PROJECT_ROOT / "src" / "stockio" / "templates"),
)


@app.before_request
def _set_db_for_request():
    """Route DB to the requested instance."""
    instance = request.args.get("instance", "paper")
    settings = load_settings()
    db.set_active_db(settings.get_db_path(instance))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Bot control
# ---------------------------------------------------------------------------


@app.route("/api/instances")
def api_instances():
    """Return status of all bot instances."""
    slots = get_slots()
    result = {}
    for name, slot in slots.items():
        result[name] = {
            "name": name,
            "running": slot.running,
            "generation": slot.generation,
            "last_error": slot.last_error,
        }
    return jsonify(result)


@app.route("/api/instances/<name>/start", methods=["POST"])
def api_start_instance(name: str):
    ok = start_bot(name)

    # Immediately fetch account data so dashboard updates without waiting
    account_data = None
    if ok:
        import time

        time.sleep(1)  # Brief wait for broker to initialize
        slot = get_slot(name)
        if slot and slot.engine:
            with contextlib.suppress(Exception):
                acct = slot.engine._broker.get_account()
                account_data = {
                    "balance": acct.balance,
                    "equity": acct.equity,
                    "unrealized_pnl": acct.unrealized_pnl,
                    "open_positions": acct.open_position_count,
                    "currency": acct.currency,
                }

    return jsonify({"ok": ok, "instance": name, "account": account_data})


@app.route("/api/instances/<name>/stop", methods=["POST"])
def api_stop_instance(name: str):
    ok = stop_bot(name)
    return jsonify({"ok": ok, "instance": name})


# ---------------------------------------------------------------------------
# Portfolio & status
# ---------------------------------------------------------------------------


@app.route("/api/status")
def api_status():
    """Portfolio summary — balance, equity, P&L."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)

    # Try live broker data (thread-safe via OandaBroker._lock)
    account = None
    if slot and slot.engine:
        with contextlib.suppress(Exception):
            account = slot.engine._broker.get_account()

    if account:
        return jsonify({
            "balance": account.balance,
            "equity": account.equity,
            "unrealized_pnl": account.unrealized_pnl,
            "margin_used": account.margin_used,
            "open_positions": account.open_position_count,
            "currency": account.currency,
            "source": "live",
        })

    # Fallback to latest DB snapshot
    snapshots = db.get_snapshots(limit=1)
    latest = snapshots[-1] if snapshots else None
    if latest:
        return jsonify({
            "balance": latest["balance"],
            "equity": latest["equity"],
            "unrealized_pnl": latest["unrealized_pnl"],
            "open_positions": latest["open_positions"],
            "source": "snapshot",
        })

    return jsonify({
        "balance": 0,
        "equity": 0,
        "unrealized_pnl": 0,
        "open_positions": 0,
        "source": "empty",
    })


# ---------------------------------------------------------------------------
# Trades & P&L
# ---------------------------------------------------------------------------


@app.route("/api/trades")
def api_trades():
    limit = request.args.get("limit", 50, type=int)
    trades = db.get_trade_history(limit=limit)
    return jsonify(trades)


@app.route("/api/pnl")
def api_pnl():
    return jsonify(db.get_pnl_summary())


@app.route("/api/open-positions")
def api_open_positions():
    """Open positions with live P&L from the broker."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify([])

    positions = []
    with contextlib.suppress(Exception):
        for p in slot.engine._broker.get_positions():
            # Match with DB trade for SL/TP
            db_trade = None
            for t in db.get_open_trades():
                if t["trade_id"] == p.trade_id:
                    db_trade = t
                    break
            positions.append({
                "instrument": p.instrument,
                "direction": p.direction.value,
                "units": p.units,
                "entry_price": p.entry_price,
                "unrealized_pnl": p.unrealized_pnl,
                "trade_id": p.trade_id,
                "stop_loss": db_trade["stop_loss"] if db_trade else None,
                "take_profit": db_trade["take_profit"] if db_trade else None,
            })
    return jsonify(positions)


@app.route("/api/snapshots")
def api_snapshots():
    limit = request.args.get("limit", 500, type=int)
    return jsonify(db.get_snapshots(limit=limit))


# ---------------------------------------------------------------------------
# Signals & sentiment
# ---------------------------------------------------------------------------


@app.route("/api/signals")
def api_signals():
    """Return last cycle's signals from the running engine."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify([])

    # Access the engine's latest features to reconstruct signals
    engine = slot.engine
    signals = []
    for name, features in engine._latest_features.items():
        sentiment = engine._sentiment.get(name, 0.0)
        sig = engine.scorer.score_instrument(name, features, sentiment)
        signals.append({
            "instrument": sig.instrument,
            "direction": sig.direction.value,
            "confidence": round(sig.confidence, 3),
            "sentiment": round(sentiment, 3),
            "rsi_14": round(features.get("rsi_14", 0), 1),
            "macd": round(features.get("macd_histogram", 0), 5),
            "adx": round(features.get("adx", 0), 1),
        })
    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return jsonify(signals)


@app.route("/api/sentiment")
def api_sentiment():
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot:
        return jsonify({})
    return jsonify({
        "combined": slot.last_sentiment,
        "trump": slot.last_trump_sentiment,
    })


# ---------------------------------------------------------------------------
# Bot log
# ---------------------------------------------------------------------------


@app.route("/api/bot-log")
def api_bot_log():
    limit = request.args.get("limit", 10, type=int)
    return jsonify(db.get_bot_logs(limit=limit))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        settings = load_settings()
        defaults = {
            "granularity": settings.granularity,
            "min_confidence": settings.min_confidence,
            "risk_per_trade": settings.risk_per_trade,
            "stop_loss_atr_mult": settings.stop_loss_atr_mult,
            "take_profit_atr_mult": settings.take_profit_atr_mult,
            "max_positions": settings.max_positions,
            "daily_loss_limit": settings.daily_loss_limit,
            "max_drawdown": settings.max_drawdown,
            "max_margin_pct": settings.max_margin_pct,
            "cycle_seconds": settings.cycle_seconds,
            "sentiment_refresh_seconds": settings.sentiment_refresh_seconds,
        }
        # Override with any DB-saved settings
        result = {}
        for k, v in defaults.items():
            saved = db.get_setting(k)
            try:
                result[k] = type(v)(saved) if saved else v
            except (ValueError, TypeError):
                result[k] = v
        return jsonify(result)
    else:
        allowed = {
            "min_confidence", "risk_per_trade", "stop_loss_atr_mult",
            "take_profit_atr_mult", "max_positions", "daily_loss_limit",
            "max_drawdown", "max_margin_pct", "cycle_seconds",
            "sentiment_refresh_seconds", "disable_daily_limit",
        }
        data = request.get_json(silent=True) or {}
        for key, value in data.items():
            if key in allowed:
                db.set_setting(key, str(value))
        return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Engine status (model, circuit breakers, etc.)
# ---------------------------------------------------------------------------


@app.route("/api/engine-status")
def api_engine_status():
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify({"running": False})

    engine = slot.engine

    # Read cycle_seconds from DB settings (same logic as bot loop)
    try:
        saved_cycle = db.get_setting("cycle_seconds")
        cycle_secs = int(saved_cycle) if saved_cycle else engine._settings.cycle_seconds
    except (ValueError, TypeError):
        cycle_secs = engine._settings.cycle_seconds

    risk = engine.risk
    peak = risk._peak_equity
    daily_lim = db.get_float_setting("daily_loss_limit", engine._settings.daily_loss_limit)
    weekly_lim = db.get_float_setting("weekly_loss_limit", engine._settings.weekly_loss_limit)
    max_dd = db.get_float_setting("max_drawdown", engine._settings.max_drawdown)

    return jsonify({
        "running": slot.running,
        "cycle_count": engine.cycle_count,
        "instruments_warmed": len(engine._warmed_up),
        "model_accuracy": (
            round(engine._outcome_tracker.rolling_accuracy, 3)
            if engine._outcome_tracker.rolling_accuracy is not None
            else None
        ),
        "training_samples": engine._outcome_tracker.training_data_count,
        "pending_outcomes": engine._outcome_tracker.pending_count,
        "halted": engine.risk.is_halted,
        "halt_reason": engine.risk.halt_reason,
        "last_cycle_time": (
            engine._last_cycle_time.isoformat()
            if engine._last_cycle_time else None
        ),
        "cycle_seconds": cycle_secs,
        "daily_pnl": round(risk._daily_pnl, 2),
        "weekly_pnl": round(risk._weekly_pnl, 2),
        "peak_equity": round(peak, 2),
        "daily_loss_limit": daily_lim,
        "weekly_loss_limit": weekly_lim,
        "max_drawdown": max_dd,
        "current_drawdown": (
            round((peak - engine.last_account.equity) / peak, 4)
            if peak > 0 and engine.last_account else 0
        ),
        "scoring_mode": engine._scorer.mode if hasattr(engine._scorer, 'mode') else None,
        "disable_daily_limit": bool(db.get_setting("disable_daily_limit")),
    })


# ---------------------------------------------------------------------------
# Economic calendar
# ---------------------------------------------------------------------------


@app.route("/api/calendar")
def api_calendar():
    """Return upcoming economic events."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify([])
    return jsonify(slot.engine._calendar.get_all_upcoming_dict())


# ---------------------------------------------------------------------------
# Visualisation APIs
# ---------------------------------------------------------------------------


@app.route("/api/sentiment/breakdown")
def api_sentiment_breakdown():
    """Per-instrument sentiment breakdown: news vs trump scores + headlines."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.sentiment_analyzer:
        return jsonify({})
    return jsonify(slot.sentiment_analyzer.get_breakdown())


@app.route("/api/trump-headlines")
def api_trump_headlines():
    """Return cached Trump/political headlines."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.sentiment_analyzer:
        return jsonify({"headlines": [], "last_refresh": None})
    analyzer = slot.sentiment_analyzer
    return jsonify({
        "headlines": analyzer.get_trump_headlines(),
        "last_refresh": (
            analyzer._cache_time.isoformat() if analyzer._cache_time else None
        ),
    })


@app.route("/api/indicators")
def api_indicators():
    """Full technical indicator values per instrument."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify({})

    result = {}
    for name, features in slot.engine._latest_features.items():
        result[name] = {
            "rsi_7": round(features.get("rsi_7", 0), 1),
            "rsi_14": round(features.get("rsi_14", 0), 1),
            "macd_histogram": round(features.get("macd_histogram", 0), 5),
            "stoch_k": round(features.get("stoch_k", 0), 1),
            "stoch_d": round(features.get("stoch_d", 0), 1),
            "atr": round(features.get("atr", 0), 5),
            "bb_percent_b": round(features.get("bb_percent_b", 0), 3),
            "adx": round(features.get("adx", 0), 1),
            "ema_cross_short_mid": round(features.get("ema_cross_short_mid", 0), 5),
            "ema_cross_mid_long": round(features.get("ema_cross_mid_long", 0), 5),
            "close_vs_ema_long": round(features.get("close_vs_ema_long", 0), 5),
            "range_vs_atr": round(features.get("range_vs_atr", 0), 2),
        }
    return jsonify(result)


@app.route("/api/scoring-breakdown")
def api_scoring_breakdown():
    """Decision waterfall — how each factor contributes to the score."""
    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)
    if not slot or not slot.engine:
        return jsonify({})

    engine = slot.engine
    result = {}
    for name, features in engine._latest_features.items():
        sentiment = engine._sentiment.get(name, 0.0)
        rsi_14 = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        ema_cross = features.get("ema_cross_short_mid", 0)
        close_vs_ema = features.get("close_vs_ema_long", 0)
        adx = features.get("adx", 0)
        bb_pct = features.get("bb_percent_b", 0.5)

        # Reconstruct the rules-based scoring breakdown
        components = []

        # RSI
        if rsi_14 < 30:
            components.append({"name": "RSI (oversold)", "value": rsi_14, "score": 1.5})
        elif rsi_14 < 40:
            components.append({"name": "RSI (low)", "value": rsi_14, "score": 0.5})
        elif rsi_14 > 70:
            components.append({"name": "RSI (overbought)", "value": rsi_14, "score": -1.5})
        elif rsi_14 > 60:
            components.append({"name": "RSI (high)", "value": rsi_14, "score": -0.5})
        else:
            components.append({"name": "RSI (neutral)", "value": rsi_14, "score": 0})

        # MACD
        components.append({
            "name": "MACD",
            "value": round(macd_hist, 5),
            "score": 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0),
        })

        # EMA Cross
        components.append({
            "name": "EMA Cross",
            "value": round(ema_cross, 5),
            "score": 1.0 if ema_cross > 0 else (-1.0 if ema_cross < 0 else 0),
        })

        # Price vs EMA
        components.append({
            "name": "Price vs EMA",
            "value": round(close_vs_ema, 5),
            "score": 0.5 if close_vs_ema > 0.001 else (-0.5 if close_vs_ema < -0.001 else 0),
        })

        # Bollinger
        components.append({
            "name": "Bollinger %B",
            "value": round(bb_pct, 3),
            "score": 0.5 if bb_pct < 0.1 else (-0.5 if bb_pct > 0.9 else 0),
        })

        # Sentiment
        components.append({
            "name": "Sentiment",
            "value": round(sentiment, 3),
            "score": round(sentiment, 3),
        })

        # ADX multiplier
        adx_mult = 0.5 if adx < 20 else 1.0
        components.append({
            "name": "ADX (trend strength)",
            "value": round(adx, 1),
            "score": adx_mult,
            "is_multiplier": True,
        })

        # Final signal
        sig = engine.scorer.score_instrument(name, features, sentiment)
        result[name] = {
            "direction": sig.direction.value,
            "confidence": round(sig.confidence, 3),
            "components": components,
            "mode": "ml" if engine.scorer._model is not None else "rules",
        }

    return jsonify(result)


@app.route("/api/model-learning")
def api_model_learning():
    """Model learning progress — accuracy, samples, retrain history."""
    import json

    slot_name = request.args.get("instance", "paper")
    slot = get_slot(slot_name)

    result = {
        "rolling_accuracy": None,
        "training_samples": 0,
        "pending_outcomes": 0,
        "is_degraded": False,
        "last_retrain": None,
        "cv_accuracies": [],
        "mean_cv_accuracy": None,
        "recent_outcomes": [],
    }

    if slot and slot.engine:
        tracker = slot.engine._outcome_tracker
        result["rolling_accuracy"] = (
            round(tracker.rolling_accuracy, 3)
            if tracker.rolling_accuracy is not None
            else None
        )
        result["training_samples"] = tracker.training_data_count
        result["pending_outcomes"] = tracker.pending_count
        result["is_degraded"] = tracker.is_degraded
        result["recent_outcomes"] = [
            1 if x else 0 for x in list(tracker._recent_outcomes)
        ]
    else:
        # No engine running — read from persisted data
        import pandas as pd

        settings = load_settings()
        parquet_path = settings.data_dir / "training_data.parquet"
        if parquet_path.exists():
            with contextlib.suppress(Exception):
                df = pd.read_parquet(parquet_path)
                result["training_samples"] = len(df)
                if not df.empty:
                    recent = df.tail(50)["label"].tolist()
                    result["recent_outcomes"] = [int(x) for x in recent]
                    wins = sum(recent)
                    result["rolling_accuracy"] = (
                        round(wins / len(recent), 3) if recent else None
                    )

        # Pending outcomes from DB
        pending = db.load_pending_outcomes()
        result["pending_outcomes"] = len(pending)

    # Read model metadata if it exists
    settings = load_settings()
    meta_path = settings.models_dir / "model_meta.json"
    if meta_path.exists():
        with contextlib.suppress(Exception):
            meta = json.loads(meta_path.read_text())
            result["last_retrain"] = meta.get("train_date")
            result["cv_accuracies"] = meta.get("cv_accuracies", [])
            result["mean_cv_accuracy"] = meta.get("mean_accuracy")

    return jsonify(result)


@app.route("/api/trade-outcomes")
def api_trade_outcomes():
    """Trade outcome statistics — win/loss ratio, per-instrument breakdown."""
    import pandas as pd

    settings = load_settings()
    parquet_path = settings.data_dir / "training_data.parquet"

    result = {
        "total": 0,
        "wins": 0,
        "losses": 0,
        "win_ratio": None,
        "by_instrument": {},
        "by_direction": {},
        "recent": [],
    }

    if not parquet_path.exists():
        return jsonify(result)

    with contextlib.suppress(Exception):
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return jsonify(result)

        result["total"] = len(df)
        result["wins"] = int(df["label"].sum())
        result["losses"] = result["total"] - result["wins"]
        result["win_ratio"] = (
            round(result["wins"] / result["total"], 3)
            if result["total"] > 0
            else None
        )

        # Per instrument
        for inst, grp in df.groupby("instrument"):
            wins = int(grp["label"].sum())
            total = len(grp)
            result["by_instrument"][inst] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_ratio": round(wins / total, 3) if total > 0 else None,
            }

        # Per direction
        for direction, grp in df.groupby("direction"):
            wins = int(grp["label"].sum())
            total = len(grp)
            result["by_direction"][direction] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_ratio": round(wins / total, 3) if total > 0 else None,
            }

        # Recent outcomes
        recent = df.tail(20).to_dict("records")
        for r in recent:
            result["recent"].append({
                "instrument": r.get("instrument"),
                "direction": r.get("direction"),
                "confidence": round(r.get("confidence", 0), 3),
                "label": int(r.get("label", 0)),
                "timestamp": r.get("timestamp", ""),
            })

    return jsonify(result)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Create and configure the Flask app."""
    settings = load_settings()
    init_slots(settings)
    db.set_default_db(settings.get_db_path())
    return app
