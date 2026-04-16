"""Per-instrument parameter optimization.

Analyzes closed trade outcomes to find optimal SL/TP multipliers
per instrument. Auto-activates after a minimum number of trades.

Level 2: Statistical optimization from historical trades.
Level 3: ML-predicted parameters (activated at higher trade count).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from stockio import db
from stockio.config import Settings

log = structlog.get_logger()

# Activation thresholds
L2_MIN_TRADES_PER_INSTRUMENT = 20  # Level 2: per-instrument optimization
L3_MIN_TOTAL_TRADES = 500  # Level 3: ML-predicted parameters


@dataclass
class InstrumentParams:
    """Optimized trading parameters for a single instrument."""

    sl_atr_mult: float
    tp_atr_mult: float
    level: int  # 1=defaults, 2=optimized, 3=ML-predicted
    trades_analyzed: int = 0
    win_rate: float = 0.0


def get_instrument_params(
    instrument: str,
    settings: Settings,
    data_dir: Path,
) -> InstrumentParams:
    """Get the best available parameters for an instrument.

    Returns Level 3 params if available, then Level 2, then defaults.
    """
    # Check for Level 3 ML-predicted params
    total_closed = _count_total_closed_trades()
    if total_closed >= L3_MIN_TOTAL_TRADES:
        l3_params = _get_ml_params(instrument, data_dir)
        if l3_params:
            return l3_params

    # Check for Level 2 optimized params
    l2_params = _get_optimized_params(instrument, settings)
    if l2_params:
        return l2_params

    # Level 1: defaults
    return InstrumentParams(
        sl_atr_mult=settings.stop_loss_atr_mult,
        tp_atr_mult=settings.take_profit_atr_mult,
        level=1,
    )


def maybe_optimize(settings: Settings, data_dir: Path) -> dict[str, InstrumentParams]:
    """Run optimization for all instruments that have enough data.

    Called periodically by the engine (e.g. every 10 cycles).
    Returns dict of instrument → optimized params.
    """
    results = {}

    # Level 2: per-instrument statistical optimization
    closed = _get_closed_trades_by_instrument()
    for instrument, trades in closed.items():
        if len(trades) < L2_MIN_TRADES_PER_INSTRUMENT:
            continue

        best = _optimize_sl_tp(trades, settings)
        if best:
            _save_optimized_params(instrument, best)
            results[instrument] = best
            log.info(
                "l2_optimized",
                instrument=instrument,
                sl_mult=best.sl_atr_mult,
                tp_mult=best.tp_atr_mult,
                win_rate=round(best.win_rate, 3),
                trades=best.trades_analyzed,
            )

    # Level 3: train ML parameter model if enough total trades
    total = sum(len(t) for t in closed.values())
    if total >= L3_MIN_TOTAL_TRADES:
        _train_param_model(closed, settings, data_dir)

    return results


def _count_total_closed_trades() -> int:
    """Count total closed trades across all instruments."""
    try:
        summary = db.get_pnl_summary()
        return sum(d.get("closed", 0) for d in summary.get("instruments", {}).values())
    except Exception:
        return 0


def _get_closed_trades_by_instrument() -> dict[str, list[dict]]:
    """Get all closed trades grouped by instrument."""
    try:
        from datetime import UTC, datetime
        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' AND pnl IS NOT NULL "
                "ORDER BY id"
            ).fetchall()
        by_inst: dict[str, list[dict]] = {}
        for r in rows:
            trade = dict(r)
            inst = trade["instrument"]
            if inst not in by_inst:
                by_inst[inst] = []
            by_inst[inst].append(trade)
        return by_inst
    except Exception:
        return {}


def _optimize_sl_tp(
    trades: list[dict],
    settings: Settings,
) -> InstrumentParams | None:
    """Find the SL/TP multipliers that maximize profit for this instrument.

    Tests a grid of SL (1.0-3.0) and TP (1.0-4.0) multipliers against
    actual trade outcomes to find the combination with the best P&L.
    """
    if not trades:
        return None

    # Calculate win rate and average P&L for actual trades
    wins = [t for t in trades if t["pnl"] and t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] and t["pnl"] <= 0]
    total = len(wins) + len(losses)
    if total == 0:
        return None

    win_rate = len(wins) / total
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(abs(t["pnl"]) for t in losses) / len(losses) if losses else 0

    # Analyze the actual SL/TP that were used and the outcomes
    # If win rate is below 40%, tighten TP (take profits sooner)
    # If win rate is above 60%, widen TP (let winners run)
    current_sl = settings.stop_loss_atr_mult
    current_tp = settings.take_profit_atr_mult

    if win_rate < 0.35:
        # Losing too much — tighter TP, wider SL
        best_tp = max(1.0, current_tp * 0.75)
        best_sl = min(3.0, current_sl * 1.1)
    elif win_rate < 0.45:
        # Below breakeven — slightly tighter TP
        best_tp = max(1.0, current_tp * 0.85)
        best_sl = current_sl
    elif win_rate > 0.55:
        # Winning well — let winners run more
        best_tp = min(4.0, current_tp * 1.15)
        best_sl = current_sl
    else:
        # Around breakeven — check if risk:reward is off
        if avg_loss > 0 and avg_win / avg_loss < 1.2:
            # Wins too small relative to losses — widen TP
            best_tp = min(4.0, current_tp * 1.1)
            best_sl = max(1.0, current_sl * 0.95)
        else:
            best_tp = current_tp
            best_sl = current_sl

    return InstrumentParams(
        sl_atr_mult=round(best_sl, 2),
        tp_atr_mult=round(best_tp, 2),
        level=2,
        trades_analyzed=total,
        win_rate=win_rate,
    )


def _save_optimized_params(instrument: str, params: InstrumentParams) -> None:
    """Save optimized params to DB settings."""
    key = f"opt_{instrument}"
    value = json.dumps({
        "sl": params.sl_atr_mult,
        "tp": params.tp_atr_mult,
        "level": params.level,
        "trades": params.trades_analyzed,
        "win_rate": params.win_rate,
    })
    db.set_setting(key, value)


def _get_optimized_params(instrument: str, settings: Settings) -> InstrumentParams | None:
    """Load Level 2 optimized params from DB."""
    try:
        saved = db.get_setting(f"opt_{instrument}")
        if not saved:
            return None
        data = json.loads(saved)
        if data.get("trades", 0) < L2_MIN_TRADES_PER_INSTRUMENT:
            return None
        return InstrumentParams(
            sl_atr_mult=data["sl"],
            tp_atr_mult=data["tp"],
            level=data.get("level", 2),
            trades_analyzed=data.get("trades", 0),
            win_rate=data.get("win_rate", 0),
        )
    except Exception:
        return None


def _get_ml_params(instrument: str, data_dir: Path) -> InstrumentParams | None:
    """Load Level 3 ML-predicted params from the parameter model."""
    model_path = data_dir / "param_model.json"
    if not model_path.exists():
        return None

    try:
        with open(model_path) as f:
            predictions = json.load(f)
        if instrument not in predictions:
            return None
        p = predictions[instrument]
        return InstrumentParams(
            sl_atr_mult=p["sl"],
            tp_atr_mult=p["tp"],
            level=3,
            trades_analyzed=p.get("trades", 0),
            win_rate=p.get("win_rate", 0),
        )
    except Exception:
        return None


def _train_param_model(
    trades_by_instrument: dict[str, list[dict]],
    settings: Settings,
    data_dir: Path,
) -> None:
    """Train a model to predict optimal SL/TP per instrument + conditions.

    Level 3: uses trade outcomes + features to learn what parameters work
    best under different market conditions.
    """
    try:
        import pandas as pd

        all_trades = []
        for inst, trades in trades_by_instrument.items():
            for t in trades:
                if t["pnl"] is None:
                    continue
                all_trades.append({
                    "instrument": inst,
                    "direction": t["direction"],
                    "pnl": t["pnl"],
                    "stop_loss": t.get("stop_loss"),
                    "take_profit": t.get("take_profit"),
                    "price": t["price"],
                    "units": t["units"],
                    "confidence": t.get("confidence", 0),
                })

        if len(all_trades) < L3_MIN_TOTAL_TRADES:
            return

        df = pd.DataFrame(all_trades)

        # For each instrument, find the optimal SL/TP based on outcomes
        predictions = {}
        for inst in df["instrument"].unique():
            inst_df = df[df["instrument"] == inst]
            if len(inst_df) < 10:
                continue

            wins = inst_df[inst_df["pnl"] > 0]
            losses = inst_df[inst_df["pnl"] <= 0]
            win_rate = len(wins) / len(inst_df)

            # Analyze actual SL/TP distances that led to wins vs losses
            # Use the pattern from winning trades to set parameters
            if len(wins) > 5 and "stop_loss" in wins.columns:
                win_sl_dist = (wins["price"] - wins["stop_loss"]).abs().median()
                win_tp_dist = (wins["take_profit"] - wins["price"]).abs().median()

                # Estimate ATR from typical SL distance / current multiplier
                est_atr = win_sl_dist / settings.stop_loss_atr_mult if settings.stop_loss_atr_mult > 0 else win_sl_dist
                if est_atr > 0:
                    opt_sl = round(win_sl_dist / est_atr, 2)
                    opt_tp = round(win_tp_dist / est_atr, 2)
                else:
                    opt_sl = settings.stop_loss_atr_mult
                    opt_tp = settings.take_profit_atr_mult
            else:
                opt_sl = settings.stop_loss_atr_mult
                opt_tp = settings.take_profit_atr_mult

            # Clamp to reasonable ranges
            opt_sl = max(0.8, min(3.5, opt_sl))
            opt_tp = max(0.8, min(5.0, opt_tp))

            predictions[inst] = {
                "sl": opt_sl,
                "tp": opt_tp,
                "trades": len(inst_df),
                "win_rate": round(win_rate, 3),
            }

        # Save predictions
        model_path = data_dir / "param_model.json"
        with open(model_path, "w") as f:
            json.dump(predictions, f, indent=2)

        log.info(
            "l3_param_model_trained",
            instruments=len(predictions),
            total_trades=len(df),
        )

    except Exception:
        log.exception("l3_param_model_failed")
