"""Flask web application — dashboard for monitoring and controlling Stockio."""

from __future__ import annotations

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
    return jsonify({"ok": ok, "instance": name})


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

    # Try to get live account data from the engine's broker
    import contextlib

    account = None
    if slot and slot.engine:
        with contextlib.suppress(Exception):
            account = slot.engine._broker.get_account()

    # Get latest snapshot from DB as fallback
    snapshots = db.get_snapshots(limit=1)
    latest = snapshots[-1] if snapshots else None

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
    elif latest:
        return jsonify({
            "balance": latest["balance"],
            "equity": latest["equity"],
            "unrealized_pnl": latest["unrealized_pnl"],
            "open_positions": latest["open_positions"],
            "source": "snapshot",
        })
    else:
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


@app.route("/api/snapshots")
def api_snapshots():
    return jsonify(db.get_snapshots())


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
        return jsonify({
            "granularity": settings.granularity,
            "min_confidence": settings.min_confidence,
            "risk_per_trade": settings.risk_per_trade,
            "stop_loss_atr_mult": settings.stop_loss_atr_mult,
            "take_profit_atr_mult": settings.take_profit_atr_mult,
            "max_positions": settings.max_positions,
            "daily_loss_limit": settings.daily_loss_limit,
            "max_drawdown": settings.max_drawdown,
            "cycle_seconds": settings.cycle_seconds,
        })
    else:
        data = request.get_json(silent=True) or {}
        for key, value in data.items():
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
    })


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Create and configure the Flask app."""
    settings = load_settings()
    init_slots(settings)
    db.set_default_db(settings.get_db_path())
    return app
