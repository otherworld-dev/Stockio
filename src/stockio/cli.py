"""Command-line interface for Stockio."""

from __future__ import annotations

import json

import click

from stockio import __version__
from stockio.config import get_logger

log = get_logger("stockio.cli")


@click.group()
@click.version_option(__version__, prog_name="stockio")
def main() -> None:
    """Stockio — AI-powered stock trading bot."""


# ------------------------------------------------------------------
# run: start the trading bot
# ------------------------------------------------------------------


@main.command()
def run() -> None:
    """Start the trading bot loop."""
    from stockio.bot import StockioBot

    bot = StockioBot()
    bot.start()


# ------------------------------------------------------------------
# train: manually retrain the ML model
# ------------------------------------------------------------------


@main.command()
@click.option("--period", default="2y", help="Historical data period (e.g. 1y, 2y, 5y)")
def train(period: str) -> None:
    """Retrain the ML prediction model."""
    from stockio import config
    from stockio.market_discovery import get_cached_tickers, maybe_refresh
    from stockio.strategy import train_model

    # Ensure market cache is populated
    if config.MARKETS:
        click.echo("Checking market ticker cache ...")
        maybe_refresh()

    tickers = get_cached_tickers()

    # Limit training set for speed
    max_training = 100
    if len(tickers) > max_training:
        click.echo(f"Universe has {len(tickers)} tickers — training on top {max_training} by market cap")
        tickers = tickers[:max_training]

    click.echo(f"Training model on {len(tickers)} tickers (period={period}) ...")
    _, _, features, accuracy = train_model(tickers, period=period)
    click.echo(f"Model trained — CV accuracy: {accuracy:.4f}")
    click.echo(f"Features used: {len(features)}")


# ------------------------------------------------------------------
# status: show portfolio summary
# ------------------------------------------------------------------


@main.command()
def status() -> None:
    """Show current portfolio status."""
    from stockio import config
    from stockio.market_data import get_current_prices
    from stockio.market_discovery import get_ticker_count
    from stockio.portfolio import get_positions, portfolio_summary

    held_tickers = [p.ticker for p in get_positions()]
    if held_tickers:
        click.echo(f"Fetching prices for {len(held_tickers)} held positions ...")
        prices = get_current_prices(held_tickers)
    else:
        prices = {}

    summary = portfolio_summary(prices)

    click.echo()
    click.echo(f"{'=' * 50}")
    click.echo(f"  STOCKIO PORTFOLIO STATUS")
    click.echo(f"{'=' * 50}")
    click.echo(f"  Markets:        {', '.join(config.MARKETS) or 'none (watchlist only)'}")
    click.echo(f"  Universe:       {get_ticker_count()} tickers")
    click.echo(f"  Cash:           £{summary['cash']:.2f}")
    click.echo(f"  Holdings value: £{summary['holdings_value']:.2f}")
    click.echo(f"  Total value:    £{summary['total_value']:.2f}")
    click.echo(f"  Initial budget: £{summary['initial_budget']:.2f}")
    click.echo(
        f"  P&L:            £{summary['total_pnl']:+.2f} "
        f"({summary['total_pnl_pct']:+.2f}%)"
    )
    click.echo(f"{'=' * 50}")

    if summary["holdings"]:
        click.echo(f"  {'Ticker':<8} {'Shares':>10} {'Avg Cost':>10} {'Price':>10} {'Value':>10} {'P&L':>10}")
        click.echo(f"  {'-' * 58}")
        for h in summary["holdings"]:
            click.echo(
                f"  {h['ticker']:<8} {h['shares']:>10.4f} "
                f"£{h['avg_cost']:>8.2f} £{h['current_price']:>8.2f} "
                f"£{h['market_value']:>8.2f} £{h['pnl']:>+8.2f}"
            )
    else:
        click.echo("  No holdings.")
    click.echo()


# ------------------------------------------------------------------
# history: show trade history
# ------------------------------------------------------------------


@main.command()
@click.option("--limit", default=20, help="Number of trades to show")
def history(limit: int) -> None:
    """Show recent trade history."""
    from stockio.portfolio import get_trade_history

    trades = get_trade_history(limit=limit)
    if not trades:
        click.echo("No trades recorded yet.")
        return

    click.echo(f"{'=' * 70}")
    click.echo(f"  RECENT TRADES (last {limit})")
    click.echo(f"{'=' * 70}")
    click.echo(f"  {'ID':>4} {'Side':<5} {'Ticker':<8} {'Shares':>10} {'Price':>10} {'Total':>10} {'Time'}")
    click.echo(f"  {'-' * 68}")
    for t in trades:
        click.echo(
            f"  {t.id or 0:>4} {t.side:<5} {t.ticker:<8} {t.shares:>10.4f} "
            f"£{t.price:>8.2f} £{t.total:>8.2f} {t.timestamp[:19]}"
        )
    click.echo()


# ------------------------------------------------------------------
# signals: show current signals without trading
# ------------------------------------------------------------------


@main.command()
def signals() -> None:
    """Analyse and display current trade signals (dry run)."""
    from stockio import config
    from stockio.market_data import get_current_prices
    from stockio.market_discovery import get_cached_tickers
    from stockio.portfolio import get_positions
    from stockio.sentiment import get_sentiment_scores
    from stockio.strategy import generate_signals

    # Show signals for held positions + top tickers
    held = [p.ticker for p in get_positions()]
    all_tickers = get_cached_tickers()
    sample = all_tickers[:20]
    combined = list(dict.fromkeys(held + sample))

    click.echo(f"Fetching prices for {len(combined)} tickers ...")
    prices = get_current_prices(combined)
    tickers = list(prices.keys())

    click.echo("Analysing sentiment ...")
    try:
        sentiments = get_sentiment_scores(tickers)
    except Exception:
        click.echo("(Sentiment analysis unavailable — showing ML signals only)")
        sentiments = {}

    sigs = generate_signals(tickers, sentiments=sentiments)

    click.echo()
    click.echo(f"{'=' * 60}")
    click.echo(f"  TRADE SIGNALS")
    click.echo(f"{'=' * 60}")
    for s in sigs:
        icon = {"BUY": "+", "SELL": "-", "HOLD": " "}[s.signal.value]
        click.echo(f"  [{icon}] {s.ticker:<12} {s.signal.value:<5} conf={s.confidence:.2f}")
        for r in s.reasons:
            click.echo(f"        {r}")
    click.echo()


# ------------------------------------------------------------------
# markets: manage market discovery
# ------------------------------------------------------------------


@main.command()
@click.option("--refresh", is_flag=True, help="Force-refresh all market caches")
@click.option("--list-supported", is_flag=True, help="Show all supported markets")
def markets(refresh: bool, list_supported: bool) -> None:
    """Show or refresh market ticker caches."""
    from stockio import config
    from stockio.market_discovery import (
        SUPPORTED_MARKETS,
        get_market_summary,
        get_ticker_count,
        refresh_all_markets,
    )

    if list_supported:
        click.echo()
        click.echo(f"{'=' * 60}")
        click.echo(f"  SUPPORTED MARKETS")
        click.echo(f"{'=' * 60}")
        for key, info in SUPPORTED_MARKETS.items():
            active = " [ACTIVE]" if key in config.MARKETS else ""
            click.echo(f"  {key:<10} {info['name']:<40} {info['currency']}{active}")
        click.echo()
        click.echo(f"  Set STOCKIO_MARKETS in .env to activate markets.")
        click.echo(f"  e.g. STOCKIO_MARKETS=LSE,AIM,NASDAQ")
        click.echo()
        return

    if refresh:
        click.echo("Refreshing all configured market caches ...")
        results = refresh_all_markets()
        for market, count in results.items():
            click.echo(f"  {market}: {count} tickers")
        click.echo(f"Total: {get_ticker_count()} tickers")
        click.echo()
        return

    # Default: show current state
    click.echo()
    click.echo(f"{'=' * 60}")
    click.echo(f"  MARKET DISCOVERY STATUS")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Configured markets: {', '.join(config.MARKETS) or 'none'}")
    click.echo(f"  Total tickers:      {get_ticker_count()}")
    click.echo(f"  Batch size:         {config.BATCH_SIZE} per cycle")
    click.echo(f"  Refresh interval:   every {config.MARKET_REFRESH_HOURS}h")
    click.echo()

    summary = get_market_summary()
    if summary:
        click.echo(f"  {'Market':<10} {'Tickers':>10} {'Last Refresh'}")
        click.echo(f"  {'-' * 50}")
        for m in summary:
            click.echo(f"  {m['market']:<10} {m['ticker_count']:>10} {m['last_refresh'][:19]}")
    else:
        click.echo("  No market data cached yet. Run: stockio markets --refresh")

    if config.WATCHLIST:
        click.echo(f"\n  Extra watchlist: {', '.join(config.WATCHLIST)}")
    click.echo()


# ------------------------------------------------------------------
# web: start the web dashboard
# ------------------------------------------------------------------


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=5000, type=int, help="Port number")
@click.option("--debug", is_flag=True, help="Enable Flask debug mode")
def web(host: str, port: int, debug: bool) -> None:
    """Start the web dashboard."""
    from stockio.webapp import run_webapp

    click.echo(f"Starting Stockio web dashboard on http://{host}:{port}")
    run_webapp(host=host, port=port, debug=debug)


# ------------------------------------------------------------------
# backtest: simple historical backtest
# ------------------------------------------------------------------


@main.command()
@click.option("--ticker", required=True, help="Ticker to backtest")
@click.option("--period", default="1y", help="Historical period")
def backtest(ticker: str, period: str) -> None:
    """Run a simple backtest on historical data for a single ticker."""
    import numpy as np

    from stockio.market_data import add_technical_indicators, fetch_history
    from stockio.strategy import _load_model

    model, scaler, meta = _load_model()
    if model is None:
        click.echo("No trained model found. Run 'stockio train' first.")
        return

    click.echo(f"Backtesting {ticker} over {period} ...")
    df = fetch_history(ticker, period=period)
    if df.empty:
        click.echo("No data available.")
        return

    df = add_technical_indicators(df)
    feature_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume")]

    cash = 500.0
    shares = 0.0
    buy_price = 0.0
    trades = 0

    for i in range(len(df)):
        row = df.iloc[[i]]
        price = float(row["Close"].iloc[0])
        features = row[feature_cols].values

        try:
            features_scaled = scaler.transform(features)
            prob = model.predict_proba(features_scaled)[0]
            prob_up = float(prob[1]) if len(prob) > 1 else 0.5
        except Exception:
            continue

        if prob_up > 0.6 and shares == 0 and cash > 0:
            shares = cash / price
            buy_price = price
            cash = 0
            trades += 1
        elif prob_up < 0.4 and shares > 0:
            cash = shares * price
            shares = 0
            trades += 1

    # Final value
    final_price = float(df["Close"].iloc[-1])
    total = cash + shares * final_price
    pnl = total - 500.0
    pnl_pct = (pnl / 500.0) * 100

    # Buy-and-hold comparison
    bh_shares = 500.0 / float(df["Close"].iloc[0])
    bh_total = bh_shares * final_price
    bh_pnl = bh_total - 500.0
    bh_pct = (bh_pnl / 500.0) * 100

    click.echo()
    click.echo(f"{'=' * 50}")
    click.echo(f"  BACKTEST RESULTS — {ticker} ({period})")
    click.echo(f"{'=' * 50}")
    click.echo(f"  Strategy P&L:     £{pnl:+.2f} ({pnl_pct:+.2f}%)")
    click.echo(f"  Buy & Hold P&L:   £{bh_pnl:+.2f} ({bh_pct:+.2f}%)")
    click.echo(f"  Total trades:     {trades}")
    click.echo(f"  Final value:      £{total:.2f}")
    click.echo(f"{'=' * 50}")
    click.echo()


if __name__ == "__main__":
    main()
