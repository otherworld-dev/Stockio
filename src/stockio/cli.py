"""CLI interface for Stockio."""

from __future__ import annotations

import click

from stockio import __version__


@click.group()
@click.version_option(__version__)
def main():
    """Stockio — AI-powered trading bot for forex and commodities."""


@main.command()
@click.option("--cycle-seconds", type=int, default=None, help="Override cycle interval")
def run(cycle_seconds: int | None):
    """Start the headless trading bot."""
    from stockio.main import run_bot

    run_bot(cycle_seconds_override=cycle_seconds)


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=5000, type=int, help="Port number")
@click.option("--debug", is_flag=True, help="Enable Flask debug mode")
def web(host: str, port: int, debug: bool):
    """Start the web dashboard."""
    from stockio.web.app import create_app

    app = create_app()
    click.echo(f"Starting Stockio dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


@main.command()
def status():
    """Show portfolio status."""
    from stockio import db
    from stockio.config import load_settings

    settings = load_settings()
    db.set_default_db(settings.get_db_path())

    snapshots = db.get_snapshots(limit=1)
    if not snapshots:
        click.echo("No data yet. Run the bot first.")
        return

    s = snapshots[-1]
    click.echo(f"Balance:        {s['balance']:.2f}")
    click.echo(f"Equity:         {s['equity']:.2f}")
    click.echo(f"Unrealized P&L: {s['unrealized_pnl']:.2f}")
    click.echo(f"Open Positions: {s['open_positions']}")
    click.echo(f"Last Cycle:     {s['cycle']}")
    click.echo(f"Timestamp:      {s['timestamp']}")


@main.command()
@click.option("--limit", default=20, type=int, help="Number of trades to show")
def history(limit: int):
    """Show recent trade history."""
    from stockio import db
    from stockio.config import load_settings

    settings = load_settings()
    db.set_default_db(settings.get_db_path())

    trades = db.get_trade_history(limit=limit)
    if not trades:
        click.echo("No trades yet.")
        return

    click.echo(f"{'Time':<22} {'Instrument':<10} {'Dir':<5} {'Units':>7} {'Price':>10} {'Conf':>6}")
    click.echo("-" * 65)
    for t in trades:
        ts = t["timestamp"][:19]
        click.echo(
            f"{ts:<22} {t['instrument']:<10} {t['direction']:<5} "
            f"{t['units']:>7} {t['price']:>10.5f} "
            f"{t['confidence'] or 0:>5.1%}"
        )


@main.command()
def pnl():
    """Show per-instrument P&L breakdown."""
    from stockio import db
    from stockio.config import load_settings

    settings = load_settings()
    db.set_default_db(settings.get_db_path())

    summary = db.get_pnl_summary()
    if summary["total_trades"] == 0:
        click.echo("No trades yet.")
        return

    click.echo(f"Total trades: {summary['total_trades']}")
    click.echo()
    click.echo(f"{'Instrument':<12} {'Trades':>7} {'Last Dir':<6} {'Last Price':>12}")
    click.echo("-" * 40)
    for inst, data in summary["instruments"].items():
        click.echo(
            f"{inst:<12} {data['trades']:>7} {data['last_direction']:<6} "
            f"{data['last_price']:>12.5f}"
        )


@main.command()
def signals():
    """Dry-run one cycle and show signals (no trading)."""

    from stockio import db
    from stockio.broker import OandaBroker
    from stockio.config import load_instruments, load_settings
    from stockio.engine import TradingEngine

    settings = load_settings()
    db.set_default_db(settings.get_db_path())

    click.echo("Fetching data and computing signals...")
    instruments = load_instruments()
    broker = OandaBroker(settings)
    engine = TradingEngine(broker=broker, instruments=instruments, settings=settings)

    # Run one cycle (data fetch + indicators only, no trading)
    engine.run_cycle()

    click.echo()
    click.echo(f"{'Instrument':<12} {'Direction':<6} {'Confidence':>10} {'RSI':>6} {'ADX':>6}")
    click.echo("-" * 45)
    for name, features in engine._latest_features.items():
        sentiment = engine._sentiment.get(name, 0.0)
        sig = engine.scorer.score_instrument(name, features, sentiment)
        click.echo(
            f"{sig.instrument:<12} {sig.direction.value:<6} "
            f"{sig.confidence:>9.1%} "
            f"{features.get('rsi_14', 0):>6.1f} "
            f"{features.get('adx', 0):>6.1f}"
        )


if __name__ == "__main__":
    main()
