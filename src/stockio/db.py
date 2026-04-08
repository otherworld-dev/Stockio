"""SQLite persistence — trades, snapshots, bot logs, settings.

Ported from old branch portfolio.py, adapted to master's data models.
Supports per-instance databases (paper.db vs live.db) via context vars.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import structlog

from stockio.broker.models import OrderRequest

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Per-instance database routing
# ---------------------------------------------------------------------------

_active_db: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_db", default=None,
)

_default_db_path: str = ""


def set_default_db(path: str | Path) -> None:
    """Set the default DB path (called once at startup from config)."""
    global _default_db_path
    _default_db_path = str(path)


def set_active_db(db_path: str | Path) -> None:
    """Set the database path for the current context (thread)."""
    _active_db.set(str(db_path))


@contextmanager
def use_db(db_path: str | Path):
    """Context manager to temporarily route all operations to *db_path*."""
    token = _active_db.set(str(db_path))
    try:
        yield
    finally:
        _active_db.reset(token)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    direction  TEXT NOT NULL,
    units      INTEGER NOT NULL,
    price      REAL NOT NULL,
    stop_loss  REAL,
    take_profit REAL,
    confidence REAL,
    trade_id   TEXT,
    timestamp  TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    balance        REAL NOT NULL,
    equity         REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    open_positions INTEGER NOT NULL DEFAULT 0,
    cycle          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bot_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle     INTEGER NOT NULL DEFAULT 0,
    summary   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _get_db_path() -> str:
    """Resolve the active database path."""
    path = _active_db.get(None)
    if path:
        return path
    if _default_db_path:
        return _default_db_path
    raise RuntimeError("No database path configured — call set_default_db() first")


@contextmanager
def _get_conn():
    """Open a connection to the active SQLite database."""
    path = _get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Trade recording
# ---------------------------------------------------------------------------


def record_trade(
    order: OrderRequest,
    trade_id: str,
    fill_price: float,
    reason: str = "",
) -> None:
    """Persist a completed trade."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (instrument, direction, units, price, stop_loss, take_profit,
                confidence, trade_id, timestamp, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order.instrument,
                order.direction.value,
                order.units,
                fill_price,
                order.stop_loss_price,
                order.take_profit_price,
                order.signal_confidence,
                trade_id,
                datetime.now(UTC).isoformat(),
                reason,
            ),
        )


def get_trade_history(limit: int = 50) -> list[dict]:
    """Return recent trades, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Snapshots (for P&L chart)
# ---------------------------------------------------------------------------


def record_snapshot(
    balance: float,
    equity: float,
    unrealized_pnl: float,
    open_positions: int,
    cycle: int,
) -> None:
    """Record a portfolio snapshot at the end of a cycle."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO snapshots
               (timestamp, balance, equity, unrealized_pnl, open_positions, cycle)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                balance,
                equity,
                unrealized_pnl,
                open_positions,
                cycle,
            ),
        )


def get_snapshots(limit: int = 500) -> list[dict]:
    """Return recent snapshots for charting."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]  # Chronological order


# ---------------------------------------------------------------------------
# Bot log (thinking/reasoning)
# ---------------------------------------------------------------------------


def record_bot_log(cycle: int, summary: dict) -> None:
    """Record a cycle summary for the 'bot thinking' panel."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO bot_log (timestamp, cycle, summary) VALUES (?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                cycle,
                json.dumps(summary),
            ),
        )


def get_bot_logs(limit: int = 30) -> list[dict]:
    """Return recent bot log entries, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bot_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for r in rows:
            entry = dict(r)
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                entry["summary"] = json.loads(entry["summary"])
            results.append(entry)
        return results


# ---------------------------------------------------------------------------
# P&L summary
# ---------------------------------------------------------------------------


def get_pnl_summary() -> dict:
    """Compute P&L summary from trade history."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM trades ORDER BY id").fetchall()

    if not rows:
        return {"total_trades": 0, "instruments": {}}

    by_instrument: dict[str, list[dict]] = {}
    for r in rows:
        trade = dict(r)
        inst = trade["instrument"]
        if inst not in by_instrument:
            by_instrument[inst] = []
        by_instrument[inst].append(trade)

    return {
        "total_trades": len(rows),
        "instruments": {
            inst: {
                "trades": len(trades),
                "last_direction": trades[-1]["direction"],
                "last_price": trades[-1]["price"],
            }
            for inst, trades in by_instrument.items()
        },
    }


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


def get_setting(key: str, default: str = "") -> str:
    """Read a persistent setting."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Write a persistent setting."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
