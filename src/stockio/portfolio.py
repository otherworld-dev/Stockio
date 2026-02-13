"""Portfolio manager — tracks holdings, cash, and trade history in SQLite.

Enforces risk management rules (position sizing, stop-loss, take-profit) and
provides P&L reporting.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass

from stockio import config
from stockio.config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float  # average cost per share (GBP)
    opened_at: str


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
            opened_at TEXT NOT NULL
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
        """
    )
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
    conn = sqlite3.connect(str(config.DB_PATH))
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


def get_initial_budget() -> float:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM account WHERE key = 'initial_budget'"
        ).fetchone()
        return float(row["value"]) if row else config.INITIAL_BUDGET_GBP


def get_positions() -> list[Position]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost, opened_at FROM portfolio WHERE shares > 0"
        ).fetchall()
        return [
            Position(
                ticker=r["ticker"],
                shares=r["shares"],
                avg_cost=r["avg_cost"],
                opened_at=r["opened_at"],
            )
            for r in rows
        ]


def get_position(ticker: str) -> Position | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT ticker, shares, avg_cost, opened_at FROM portfolio WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        if row and row["shares"] > 0:
            return Position(
                ticker=row["ticker"],
                shares=row["shares"],
                avg_cost=row["avg_cost"],
                opened_at=row["opened_at"],
            )
        return None


def record_buy(ticker: str, shares: float, price: float, reason: str = "") -> TradeRecord:
    """Record a purchase, update portfolio and cash."""
    total = shares * price
    cash = get_cash()
    if total > cash:
        raise ValueError(
            f"Insufficient cash: need £{total:.2f} but only £{cash:.2f} available"
        )

    now = dt.datetime.utcnow().isoformat()

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
                "INSERT OR REPLACE INTO portfolio (ticker, shares, avg_cost, opened_at) "
                "VALUES (?, ?, ?, ?)",
                (ticker, shares, price, now),
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


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------


def check_position_limit(ticker: str, buy_total: float) -> bool:
    """Return True if buying *buy_total* GBP of *ticker* is within risk limits."""
    cash = get_cash()
    positions = get_positions()
    portfolio_value = cash + sum(p.shares * p.avg_cost for p in positions)
    max_allowed = portfolio_value * (config.MAX_POSITION_PCT / 100.0)

    current_value = 0.0
    pos = get_position(ticker)
    if pos:
        current_value = pos.shares * pos.avg_cost

    return (current_value + buy_total) <= max_allowed


def check_stop_loss(ticker: str, current_price: float) -> bool:
    """Return True if the position should be sold (stop-loss triggered)."""
    pos = get_position(ticker)
    if pos is None:
        return False
    loss_pct = ((pos.avg_cost - current_price) / pos.avg_cost) * 100
    return loss_pct >= config.STOP_LOSS_PCT


def check_take_profit(ticker: str, current_price: float) -> bool:
    """Return True if the position should be sold (take-profit triggered)."""
    pos = get_position(ticker)
    if pos is None:
        return False
    gain_pct = ((current_price - pos.avg_cost) / pos.avg_cost) * 100
    return gain_pct >= config.TAKE_PROFIT_PCT


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def portfolio_summary(current_prices: dict[str, float]) -> dict:
    """Return a summary of the current portfolio state."""
    cash = get_cash()
    initial = get_initial_budget()
    positions = get_positions()

    holdings = []
    holdings_value = 0.0
    for pos in positions:
        price = current_prices.get(pos.ticker, pos.avg_cost)
        market_val = pos.shares * price
        pnl = (price - pos.avg_cost) * pos.shares
        pnl_pct = ((price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost else 0
        holdings_value += market_val
        holdings.append({
            "ticker": pos.ticker,
            "shares": pos.shares,
            "avg_cost": round(pos.avg_cost, 2),
            "current_price": round(price, 2),
            "market_value": round(market_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    total_value = cash + holdings_value
    total_pnl = total_value - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial else 0

    return {
        "cash": round(cash, 2),
        "holdings_value": round(holdings_value, 2),
        "total_value": round(total_value, 2),
        "initial_budget": round(initial, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "num_positions": len(holdings),
        "holdings": holdings,
    }
