"""Portfolio manager — tracks holdings, cash, and trade history in SQLite.

Enforces risk management rules (position sizing, stop-loss, take-profit) and
provides P&L reporting.  Supports multiple asset types with per-asset risk
parameters.

Supports per-instance databases: set the active DB with :func:`use_db` or
call :func:`set_active_db` before using any other function.  If no active
DB has been set, falls back to the default ``config.DB_PATH``.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from stockio import config
from stockio.config import AssetType, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-instance database routing
# ---------------------------------------------------------------------------

_active_db: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_db", default=None,
)


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
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float  # average cost per unit
    opened_at: str
    direction: str = "long"  # "long" or "short"
    asset_type: str = "equity"  # "equity", "forex", "commodity", "crypto"


@dataclass
class TradeRecord:
    id: int | None
    ticker: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    total: float
    timestamp: str
    reason: str


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            ticker    TEXT PRIMARY KEY,
            shares    REAL NOT NULL DEFAULT 0,
            avg_cost  REAL NOT NULL DEFAULT 0,
            opened_at TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'long'
        );

        CREATE TABLE IF NOT EXISTS trades (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL,
            side      TEXT NOT NULL,
            shares    REAL NOT NULL,
            price     REAL NOT NULL,
            total     REAL NOT NULL,
            timestamp TEXT NOT NULL,
            reason    TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS account (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            cash            REAL NOT NULL,
            holdings_value  REAL NOT NULL,
            total_value     REAL NOT NULL,
            pnl             REAL NOT NULL,
            pnl_pct         REAL NOT NULL,
            num_positions   INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS bot_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            entries   TEXT NOT NULL
        );
        """
    )
    # Add direction column if it doesn't exist (migration for existing DBs)
    try:
        conn.execute("SELECT direction FROM portfolio LIMIT 0")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE portfolio ADD COLUMN direction TEXT NOT NULL DEFAULT 'long'")

    # Add asset_type column if it doesn't exist
    try:
        conn.execute("SELECT asset_type FROM portfolio LIMIT 0")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE portfolio ADD COLUMN asset_type TEXT NOT NULL DEFAULT 'equity'")

    # Seed initial cash balance if not yet set
    row = conn.execute("SELECT value FROM account WHERE key = 'cash'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO account (key, value) VALUES ('cash', ?)",
            (str(config.INITIAL_BUDGET_GBP),),
        )
        conn.execute(
            "INSERT INTO account (key, value) VALUES ('initial_budget', ?)",
            (str(config.INITIAL_BUDGET_GBP),),
        )
        conn.commit()
        log.info("Portfolio initialised with £%.2f", config.INITIAL_BUDGET_GBP)


@contextmanager
def _get_conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = _active_db.get()
    path = db if db else str(config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cash() -> float:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM account WHERE key = 'cash'").fetchone()
        return float(row["value"])


def set_cash(amount: float) -> None:
    with _get_conn() as conn:
        conn.execute("UPDATE account SET value = ? WHERE key = 'cash'", (str(amount),))


def get_setting(key: str, default: str = "") -> str:
    """Read a setting from the account table."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM account WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Write a setting to the account table (upsert)."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO account (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_initial_budget() -> float:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM account WHERE key = 'initial_budget'"
        ).fetchone()
        return float(row["value"]) if row else config.INITIAL_BUDGET_GBP


def get_positions() -> list[Position]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost, opened_at, direction, asset_type "
            "FROM portfolio WHERE shares > 0"
        ).fetchall()
        return [
            Position(
                ticker=r["ticker"],
                shares=r["shares"],
                avg_cost=r["avg_cost"],
                opened_at=r["opened_at"],
                direction=r["direction"],
                asset_type=r["asset_type"] if r["asset_type"] else "equity",
            )
            for r in rows
        ]


def get_position(ticker: str) -> Position | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT ticker, shares, avg_cost, opened_at, direction, asset_type "
            "FROM portfolio WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        if row and row["shares"] > 0:
            return Position(
                ticker=row["ticker"],
                shares=row["shares"],
                avg_cost=row["avg_cost"],
                opened_at=row["opened_at"],
                direction=row["direction"],
                asset_type=row["asset_type"] if row["asset_type"] else "equity",
            )
        return None


def remove_position(ticker: str) -> None:
    """Remove a position from the local DB (used during Alpaca sync)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))


def record_buy(ticker: str, shares: float, price: float, reason: str = "") -> TradeRecord:
    """Record a purchase, update portfolio and cash."""
    total = shares * price
    cash = get_cash()
    if total > cash:
        raise ValueError(
            f"Insufficient cash: need £{total:.2f} but only £{cash:.2f} available"
        )

    now = dt.datetime.utcnow().isoformat()
    asset_type = config.get_asset_type(ticker).value

    with _get_conn() as conn:
        # Update cash
        new_cash = cash - total
        conn.execute("UPDATE account SET value = ? WHERE key = 'cash'", (str(new_cash),))

        # Update or insert position
        existing = conn.execute(
            "SELECT shares, avg_cost FROM portfolio WHERE ticker = ?", (ticker,)
        ).fetchone()

        if existing and existing["shares"] > 0:
            old_shares = existing["shares"]
            old_cost = existing["avg_cost"]
            new_shares = old_shares + shares
            new_avg = ((old_shares * old_cost) + (shares * price)) / new_shares
            conn.execute(
                "UPDATE portfolio SET shares = ?, avg_cost = ? WHERE ticker = ?",
                (new_shares, new_avg, ticker),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio "
                "(ticker, shares, avg_cost, opened_at, asset_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, shares, price, now, asset_type),
            )

        # Record trade
        conn.execute(
            "INSERT INTO trades (ticker, side, shares, price, total, timestamp, reason) "
            "VALUES (?, 'BUY', ?, ?, ?, ?, ?)",
            (ticker, shares, price, total, now, reason),
        )

    log.info("BUY  %s x%.4f @ £%.2f = £%.2f  (cash: £%.2f → £%.2f)",
             ticker, shares, price, total, cash, new_cash)

    return TradeRecord(
        id=None, ticker=ticker, side="BUY", shares=shares,
        price=price, total=total, timestamp=now, reason=reason,
    )


def record_sell(ticker: str, shares: float, price: float, reason: str = "") -> TradeRecord:
    """Record a sale, update portfolio and cash."""
    pos = get_position(ticker)
    if pos is None or pos.shares < shares:
        avail = pos.shares if pos else 0
        raise ValueError(
            f"Cannot sell {shares} shares of {ticker} — only {avail} held"
        )

    total = shares * price
    cash = get_cash()
    now = dt.datetime.utcnow().isoformat()

    with _get_conn() as conn:
        new_cash = cash + total
        conn.execute("UPDATE account SET value = ? WHERE key = 'cash'", (str(new_cash),))

        new_shares = pos.shares - shares
        if new_shares < 1e-9:
            conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
        else:
            conn.execute(
                "UPDATE portfolio SET shares = ? WHERE ticker = ?",
                (new_shares, ticker),
            )

        conn.execute(
            "INSERT INTO trades (ticker, side, shares, price, total, timestamp, reason) "
            "VALUES (?, 'SELL', ?, ?, ?, ?, ?)",
            (ticker, shares, price, total, now, reason),
        )

    pnl = (price - pos.avg_cost) * shares
    log.info("SELL %s x%.4f @ £%.2f = £%.2f  P&L: £%.2f  (cash: £%.2f → £%.2f)",
             ticker, shares, price, total, pnl, cash, new_cash)

    return TradeRecord(
        id=None, ticker=ticker, side="SELL", shares=shares,
        price=price, total=total, timestamp=now, reason=reason,
    )


def record_short(ticker: str, shares: float, price: float, reason: str = "") -> TradeRecord:
    """Open a short position — borrow shares and sell them at *price*.

    Cash increases by the sale proceeds.  The position is recorded with
    direction='short' and avg_cost = the entry price.  We need enough
    free margin (cash) to cover potential losses.
    """
    total = shares * price
    now = dt.datetime.utcnow().isoformat()
    asset_type = config.get_asset_type(ticker).value

    with _get_conn() as conn:
        cash = float(conn.execute(
            "SELECT value FROM account WHERE key = 'cash'"
        ).fetchone()["value"])

        # Credit proceeds to cash
        new_cash = cash + total
        conn.execute("UPDATE account SET value = ? WHERE key = 'cash'", (str(new_cash),))

        # Record or add to short position
        existing = conn.execute(
            "SELECT shares, avg_cost, direction FROM portfolio WHERE ticker = ?",
            (ticker,),
        ).fetchone()

        if existing and existing["shares"] > 0 and existing["direction"] == "short":
            old_shares = existing["shares"]
            old_cost = existing["avg_cost"]
            new_shares = old_shares + shares
            new_avg = ((old_shares * old_cost) + (shares * price)) / new_shares
            conn.execute(
                "UPDATE portfolio SET shares = ?, avg_cost = ? WHERE ticker = ?",
                (new_shares, new_avg, ticker),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio "
                "(ticker, shares, avg_cost, opened_at, direction, asset_type) "
                "VALUES (?, ?, ?, ?, 'short', ?)",
                (ticker, shares, price, now, asset_type),
            )

        conn.execute(
            "INSERT INTO trades (ticker, side, shares, price, total, timestamp, reason) "
            "VALUES (?, 'SHORT', ?, ?, ?, ?, ?)",
            (ticker, shares, price, total, now, reason),
        )

    log.info("SHORT %s x%.4f @ £%.2f = £%.2f  (cash: £%.2f → £%.2f)",
             ticker, shares, price, total, cash, new_cash)

    return TradeRecord(
        id=None, ticker=ticker, side="SHORT", shares=shares,
        price=price, total=total, timestamp=now, reason=reason,
    )


def record_cover(ticker: str, shares: float, price: float, reason: str = "") -> TradeRecord:
    """Close (cover) a short position — buy back *shares* at *price*.

    Cash decreases by the purchase cost.  P&L = (entry_price - cover_price) * shares.
    """
    pos = get_position(ticker)
    if pos is None or pos.direction != "short" or pos.shares < shares:
        avail = pos.shares if pos and pos.direction == "short" else 0
        raise ValueError(
            f"Cannot cover {shares} shares of {ticker} — only {avail} shorted"
        )

    total = shares * price
    cash = get_cash()
    if total > cash:
        raise ValueError(
            f"Insufficient cash to cover: need £{total:.2f} but only £{cash:.2f} available"
        )

    now = dt.datetime.utcnow().isoformat()

    with _get_conn() as conn:
        new_cash = cash - total
        conn.execute("UPDATE account SET value = ? WHERE key = 'cash'", (str(new_cash),))

        new_shares = pos.shares - shares
        if new_shares < 1e-9:
            conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
        else:
            conn.execute(
                "UPDATE portfolio SET shares = ? WHERE ticker = ?",
                (new_shares, ticker),
            )

        conn.execute(
            "INSERT INTO trades (ticker, side, shares, price, total, timestamp, reason) "
            "VALUES (?, 'COVER', ?, ?, ?, ?, ?)",
            (ticker, shares, price, total, now, reason),
        )

    pnl = (pos.avg_cost - price) * shares
    log.info("COVER %s x%.4f @ £%.2f = £%.2f  P&L: £%.2f  (cash: £%.2f → £%.2f)",
             ticker, shares, price, total, pnl, cash, new_cash)

    return TradeRecord(
        id=None, ticker=ticker, side="COVER", shares=shares,
        price=price, total=total, timestamp=now, reason=reason,
    )


def record_bot_log(entries: list[dict]) -> None:
    """Store the bot's reasoning log for the most recent cycle."""
    now = dt.datetime.utcnow().isoformat()
    import json as _json
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO bot_log (timestamp, entries) VALUES (?, ?)",
            (now, _json.dumps(entries)),
        )
        # Keep only the last 50 cycles
        conn.execute(
            "DELETE FROM bot_log WHERE id NOT IN "
            "(SELECT id FROM bot_log ORDER BY id DESC LIMIT 50)"
        )


def get_bot_logs(limit: int = 10) -> list[dict]:
    """Return recent bot reasoning logs."""
    import json as _json
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, entries FROM bot_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"timestamp": r["timestamp"], "entries": _json.loads(r["entries"])}
            for r in rows
        ]


def record_snapshot(current_prices: dict[str, float]) -> None:
    """Save a point-in-time snapshot of portfolio value for charting."""
    summary = portfolio_summary(current_prices)
    now = dt.datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO snapshots "
            "(timestamp, cash, holdings_value, total_value, pnl, pnl_pct, num_positions) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                summary["cash"],
                summary["holdings_value"],
                summary["total_value"],
                summary["total_pnl"],
                summary["total_pnl_pct"],
                summary["num_positions"],
            ),
        )


def get_snapshots(limit: int = 500) -> list[dict]:
    """Return recent portfolio snapshots for charting."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, cash, holdings_value, total_value, pnl, pnl_pct, num_positions "
            "FROM snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "cash": r["cash"],
                "holdings_value": r["holdings_value"],
                "total_value": r["total_value"],
                "pnl": r["pnl"],
                "pnl_pct": r["pnl_pct"],
                "num_positions": r["num_positions"],
            }
            for r in reversed(rows)  # chronological order
        ]


def reset_all_data() -> None:
    """Wipe all portfolio data and reset cash to the configured initial budget.

    Clears: positions, trades, snapshots, bot logs, and account balances.
    """
    with _get_conn() as conn:
        conn.execute("DELETE FROM portfolio")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM snapshots")
        conn.execute("DELETE FROM bot_log")
        conn.execute("DELETE FROM account")
        conn.execute(
            "INSERT INTO account (key, value) VALUES ('cash', ?)",
            (str(config.INITIAL_BUDGET_GBP),),
        )
        conn.execute(
            "INSERT INTO account (key, value) VALUES ('initial_budget', ?)",
            (str(config.INITIAL_BUDGET_GBP),),
        )
    log.info("All data reset — cash restored to £%.2f", config.INITIAL_BUDGET_GBP)


def get_trade_history(limit: int = 50) -> list[TradeRecord]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, ticker, side, shares, price, total, timestamp, reason "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            TradeRecord(
                id=r["id"], ticker=r["ticker"], side=r["side"], shares=r["shares"],
                price=r["price"], total=r["total"], timestamp=r["timestamp"],
                reason=r["reason"],
            )
            for r in rows
        ]


def get_trade_history_with_pnl(limit: int = 50) -> list[dict]:
    """Return recent trades annotated with per-trade P&L for exits.

    Walks all trades chronologically to build a running cost basis per
    ticker, then returns the most recent *limit* trades with a ``pnl``
    field set for SELL/COVER trades (None for BUY/SHORT).
    """
    with _get_conn() as conn:
        all_rows = conn.execute(
            "SELECT id, ticker, side, shares, price, total, timestamp, reason "
            "FROM trades ORDER BY id ASC"
        ).fetchall()

    # Running cost basis per ticker
    basis: dict[str, dict] = {}  # ticker -> {long_shares, long_cost, short_shares, short_cost}

    annotated: list[dict] = []
    for r in all_rows:
        ticker = r["ticker"]
        side = r["side"]
        shares = r["shares"]
        price = r["price"]

        if ticker not in basis:
            basis[ticker] = {"long_shares": 0.0, "long_cost": 0.0,
                             "short_shares": 0.0, "short_cost": 0.0}
        b = basis[ticker]
        pnl = None

        if side == "BUY":
            b["long_shares"] += shares
            b["long_cost"] += shares * price
        elif side == "SELL":
            if b["long_shares"] > 0:
                avg_cost = b["long_cost"] / b["long_shares"]
                sold = min(shares, b["long_shares"])
                pnl = round((price - avg_cost) * sold, 2)
                frac = sold / b["long_shares"] if b["long_shares"] else 0
                b["long_cost"] -= b["long_cost"] * frac
                b["long_shares"] -= sold
        elif side == "SHORT":
            b["short_shares"] += shares
            b["short_cost"] += shares * price
        elif side == "COVER":
            if b["short_shares"] > 0:
                avg_entry = b["short_cost"] / b["short_shares"]
                covered = min(shares, b["short_shares"])
                pnl = round((avg_entry - price) * covered, 2)
                frac = covered / b["short_shares"] if b["short_shares"] else 0
                b["short_cost"] -= b["short_cost"] * frac
                b["short_shares"] -= covered

        annotated.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "display_name": config.get_asset_display_name(r["ticker"]),
            "side": side,
            "shares": shares,
            "price": round(price, 2),
            "total": round(abs(r["total"]), 2),
            "pnl": pnl,
            "timestamp": r["timestamp"],
            "reason": r["reason"],
        })

    # Return most recent first, limited
    return annotated[-limit:][::-1]


def get_market_stats(current_prices: dict[str, float]) -> dict[str, dict]:
    """Return per-market-type aggregated stats.

    Groups open positions and completed trades by asset type and returns
    counts, unrealised P&L (open), realised P&L (closed trades), and
    total volume.
    """
    # Unrealised P&L from open positions
    positions = get_positions()
    market_stats: dict[str, dict] = {}

    for pos in positions:
        at = pos.asset_type or "equity"
        price = current_prices.get(pos.ticker, pos.avg_cost)
        if pos.direction == "short":
            pnl = (pos.avg_cost - price) * pos.shares
            value = pos.shares * price
        else:
            pnl = (price - pos.avg_cost) * pos.shares
            value = pos.shares * price

        stats = market_stats.setdefault(at, {
            "asset_type": at,
            "open_positions": 0,
            "unrealised_pnl": 0.0,
            "open_value": 0.0,
            "total_trades": 0,
            "total_buys": 0,
            "total_sells": 0,
            "total_volume": 0.0,
        })
        stats["open_positions"] += 1
        stats["unrealised_pnl"] += pnl
        stats["open_value"] += value

    # Trade history stats per market
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, side, shares, price, total FROM trades"
        ).fetchall()

    for r in rows:
        at = config.get_asset_type(r["ticker"]).value
        stats = market_stats.setdefault(at, {
            "asset_type": at,
            "open_positions": 0,
            "unrealised_pnl": 0.0,
            "open_value": 0.0,
            "total_trades": 0,
            "total_buys": 0,
            "total_sells": 0,
            "total_volume": 0.0,
        })
        stats["total_trades"] += 1
        if r["side"] in ("BUY", "COVER"):
            stats["total_buys"] += 1
        elif r["side"] in ("SELL", "SHORT"):
            stats["total_sells"] += 1
        stats["total_volume"] += abs(r["total"])

    # Round values
    for stats in market_stats.values():
        stats["unrealised_pnl"] = round(stats["unrealised_pnl"], 2)
        stats["open_value"] = round(stats["open_value"], 2)
        stats["total_volume"] = round(stats["total_volume"], 2)

    return market_stats


# ---------------------------------------------------------------------------
# Profit & Loss breakdown
# ---------------------------------------------------------------------------


def get_pnl_summary(current_prices: dict[str, float]) -> list[dict]:
    """Return per-ticker P&L breakdown combining realised and unrealised.

    For each ticker that has ever been traded, the returned dict contains:
    - ``ticker``, ``display_name``, ``asset_type``
    - ``realised_pnl``: sum of closed round-trips (BUY→SELL, SHORT→COVER)
    - ``unrealised_pnl``: current open-position P&L (0 if flat)
    - ``total_pnl``: realised + unrealised
    - ``total_bought``, ``total_sold``: cumulative volumes
    - ``num_trades``: total trade count
    - ``direction``: ``"long"`` / ``"short"`` / ``"flat"`` (current state)
    - ``open_shares``: shares currently held (0 if flat)
    """
    # 1. Fetch all trades ordered chronologically
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, side, shares, price, total FROM trades ORDER BY id ASC"
        ).fetchall()

    # 2. Build per-ticker running cost basis and realised P&L.
    #    We use a FIFO-style accumulator.
    class _Acc:  # noqa: N801
        __slots__ = (
            "realised", "total_bought", "total_sold", "num_trades",
            "long_shares", "long_cost",   # running totals for long
            "short_shares", "short_cost", # running totals for short
        )

        def __init__(self) -> None:
            self.realised = 0.0
            self.total_bought = 0.0
            self.total_sold = 0.0
            self.num_trades = 0
            self.long_shares = 0.0
            self.long_cost = 0.0
            self.short_shares = 0.0
            self.short_cost = 0.0

    accs: dict[str, _Acc] = {}

    for r in rows:
        ticker = r["ticker"]
        acc = accs.setdefault(ticker, _Acc())
        acc.num_trades += 1
        side = r["side"]
        shares = r["shares"]
        price = r["price"]
        total = abs(r["total"])

        if side == "BUY":
            acc.total_bought += total
            acc.long_shares += shares
            acc.long_cost += shares * price
        elif side == "SELL":
            acc.total_sold += total
            if acc.long_shares > 0:
                avg_cost = acc.long_cost / acc.long_shares
                sold = min(shares, acc.long_shares)
                acc.realised += (price - avg_cost) * sold
                # Reduce basis proportionally
                frac = sold / acc.long_shares if acc.long_shares else 0
                acc.long_cost -= acc.long_cost * frac
                acc.long_shares -= sold
        elif side == "SHORT":
            acc.total_sold += total
            acc.short_shares += shares
            acc.short_cost += shares * price
        elif side == "COVER":
            acc.total_bought += total
            if acc.short_shares > 0:
                avg_entry = acc.short_cost / acc.short_shares
                covered = min(shares, acc.short_shares)
                acc.realised += (avg_entry - price) * covered
                frac = covered / acc.short_shares if acc.short_shares else 0
                acc.short_cost -= acc.short_cost * frac
                acc.short_shares -= covered

    # 3. Combine with unrealised P&L from open positions
    positions = {p.ticker: p for p in get_positions()}
    all_tickers = sorted(set(list(accs.keys()) + list(positions.keys())))

    result: list[dict] = []
    for ticker in all_tickers:
        acc = accs.get(ticker, _Acc())
        pos = positions.get(ticker)
        price = current_prices.get(ticker, pos.avg_cost if pos else 0.0)

        if pos and pos.direction == "short":
            unrealised = (pos.avg_cost - price) * pos.shares
            direction = "short"
            open_shares = pos.shares
        elif pos and pos.direction == "long":
            unrealised = (price - pos.avg_cost) * pos.shares
            direction = "long"
            open_shares = pos.shares
        else:
            unrealised = 0.0
            direction = "flat"
            open_shares = 0.0

        total_pnl = acc.realised + unrealised

        result.append({
            "ticker": ticker,
            "display_name": config.get_asset_display_name(ticker),
            "asset_type": config.get_asset_type(ticker).value,
            "realised_pnl": round(acc.realised, 2),
            "unrealised_pnl": round(unrealised, 2),
            "total_pnl": round(total_pnl, 2),
            "total_bought": round(acc.total_bought, 2),
            "total_sold": round(acc.total_sold, 2),
            "num_trades": acc.num_trades,
            "direction": direction,
            "open_shares": round(open_shares, 4),
            "avg_cost": round(pos.avg_cost, 4) if pos else None,
            "current_price": round(price, 4) if pos else None,
        })

    # Sort by absolute total P&L descending (biggest movers first)
    result.sort(key=lambda r: abs(r["total_pnl"]), reverse=True)
    return result


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------


def check_position_limit(ticker: str, buy_total: float) -> bool:
    """Return True if buying *buy_total* of *ticker* is within risk limits.

    Uses per-asset-type position limits.
    """
    cash = get_cash()
    positions = get_positions()
    long_value = sum(p.shares * p.avg_cost for p in positions if p.direction == "long")
    portfolio_value = cash + long_value
    asset_type = config.get_asset_type(ticker)
    risk = config.get_risk_params(asset_type)
    max_allowed = portfolio_value * (risk["max_position_pct"] / 100.0)

    current_value = 0.0
    pos = get_position(ticker)
    if pos and pos.direction == "long":
        current_value = pos.shares * pos.avg_cost

    return (current_value + buy_total) <= max_allowed


def check_short_limit(ticker: str, short_total: float) -> bool:
    """Return True if shorting *short_total* GBP of *ticker* is within risk limits.

    Enforces two caps:
      1. Single-position limit: MAX_SHORT_POSITION_PCT of portfolio
      2. Total short exposure: MAX_TOTAL_SHORT_PCT of portfolio
    """
    cash = get_cash()
    positions = get_positions()
    long_value = sum(p.shares * p.avg_cost for p in positions if p.direction == "long")
    portfolio_value = cash + long_value
    if portfolio_value <= 0:
        return False

    # Single position limit
    max_single = portfolio_value * (config.MAX_SHORT_POSITION_PCT / 100.0)
    current_short = 0.0
    pos = get_position(ticker)
    if pos and pos.direction == "short":
        current_short = pos.shares * pos.avg_cost
    if (current_short + short_total) > max_single:
        return False

    # Total short exposure limit
    total_short = sum(
        p.shares * p.avg_cost for p in positions if p.direction == "short"
    )
    max_total = portfolio_value * (config.MAX_TOTAL_SHORT_PCT / 100.0)
    if (total_short + short_total) > max_total:
        return False

    return True


def check_stop_loss(ticker: str, current_price: float) -> bool:
    """Return True if the position should be exited (stop-loss triggered).

    Uses per-asset-type thresholds (crypto has wider thresholds than equities).
    For long positions: price dropped below entry by stop_loss_pct.
    For short positions: price rose above entry by stop_loss_pct.
    """
    pos = get_position(ticker)
    if pos is None:
        return False
    asset_type = config.get_asset_type(ticker)
    risk = config.get_risk_params(asset_type)
    stop_pct = risk["stop_loss_pct"]
    if pos.direction == "short":
        # Short: we lose when price goes UP
        loss_pct = ((current_price - pos.avg_cost) / pos.avg_cost) * 100
        return loss_pct >= stop_pct
    # Long: we lose when price goes DOWN
    loss_pct = ((pos.avg_cost - current_price) / pos.avg_cost) * 100
    return loss_pct >= stop_pct


def check_take_profit(ticker: str, current_price: float) -> bool:
    """Return True if the position should be exited (take-profit triggered).

    Uses per-asset-type thresholds.
    For long positions: price rose above entry by take_profit_pct.
    For short positions: price dropped below entry by take_profit_pct.
    """
    pos = get_position(ticker)
    if pos is None:
        return False
    asset_type = config.get_asset_type(ticker)
    risk = config.get_risk_params(asset_type)
    tp_pct = risk["take_profit_pct"]
    if pos.direction == "short":
        # Short: we profit when price goes DOWN
        gain_pct = ((pos.avg_cost - current_price) / pos.avg_cost) * 100
        return gain_pct >= tp_pct
    # Long: we profit when price goes UP
    gain_pct = ((current_price - pos.avg_cost) / pos.avg_cost) * 100
    return gain_pct >= tp_pct


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def portfolio_summary(current_prices: dict[str, float]) -> dict:
    """Return a summary of the current portfolio state.

    For **long** positions the market value is ``shares * price``.
    For **short** positions the *liability* is ``shares * price`` and the
    unrealised P&L is ``(avg_cost - price) * shares`` (profit when price
    drops below entry).  Total portfolio value = cash + long_value - short_liability.
    """
    cash = get_cash()
    initial = get_initial_budget()
    positions = get_positions()

    holdings = []
    long_value = 0.0
    short_liability = 0.0

    for pos in positions:
        price = current_prices.get(pos.ticker, pos.avg_cost)

        if pos.direction == "short":
            # Short: liability = what it'd cost to buy back
            liability = pos.shares * price
            short_liability += liability
            pnl = (pos.avg_cost - price) * pos.shares
            pnl_pct = ((pos.avg_cost - price) / pos.avg_cost * 100) if pos.avg_cost else 0
            market_val = -liability  # negative to show it's a liability
        else:
            market_val = pos.shares * price
            long_value += market_val
            pnl = (price - pos.avg_cost) * pos.shares
            pnl_pct = ((price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost else 0

        holdings.append({
            "ticker": pos.ticker,
            "shares": pos.shares,
            "avg_cost": round(pos.avg_cost, 2),
            "current_price": round(price, 2),
            "market_value": round(market_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "direction": pos.direction,
            "asset_type": pos.asset_type,
            "display_name": config.get_asset_display_name(pos.ticker),
        })

    holdings_value = long_value - short_liability
    total_value = cash + holdings_value
    total_pnl = total_value - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial else 0

    # Unrealised = sum of P&L on open positions
    unrealised_pnl = sum(h["pnl"] for h in holdings)
    # Realised = total P&L minus unrealised (what's already been locked in)
    realised_pnl = total_pnl - unrealised_pnl

    # Count long vs short positions
    num_long = sum(1 for h in holdings if h["direction"] != "short")
    num_short = sum(1 for h in holdings if h["direction"] == "short")

    return {
        "cash": round(cash, 2),
        "holdings_value": round(holdings_value, 2),
        "long_value": round(long_value, 2),
        "short_value": round(short_liability, 2),
        "num_long": num_long,
        "num_short": num_short,
        "total_value": round(total_value, 2),
        "initial_budget": round(initial, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "realised_pnl": round(realised_pnl, 2),
        "unrealised_pnl": round(unrealised_pnl, 2),
        "num_positions": len(holdings),
        "holdings": holdings,
    }
