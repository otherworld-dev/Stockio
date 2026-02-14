"""Tests for the portfolio manager."""

import os
import sqlite3
import tempfile
from unittest import mock

import pytest

# Patch config before importing portfolio
_tmp = tempfile.mkdtemp()
_patches = {
    "stockio.config.DATA_DIR": mock.patch("stockio.config.DATA_DIR", new=__import__("pathlib").Path(_tmp)),
    "stockio.config.DB_PATH": mock.patch("stockio.config.DB_PATH", new=__import__("pathlib").Path(_tmp) / "test.db"),
    "stockio.config.INITIAL_BUDGET_GBP": mock.patch("stockio.config.INITIAL_BUDGET_GBP", new=500.0),
    "stockio.config.MAX_POSITION_PCT": mock.patch("stockio.config.MAX_POSITION_PCT", new=20.0),
    "stockio.config.STOP_LOSS_PCT": mock.patch("stockio.config.STOP_LOSS_PCT", new=5.0),
    "stockio.config.TAKE_PROFIT_PCT": mock.patch("stockio.config.TAKE_PROFIT_PCT", new=15.0),
}
for p in _patches.values():
    p.start()

from stockio.portfolio import (
    check_position_limit,
    check_stop_loss,
    check_take_profit,
    get_cash,
    get_position,
    get_positions,
    get_trade_history,
    portfolio_summary,
    record_buy,
    record_sell,
    set_cash,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure each test starts with a clean database."""
    import stockio.config as cfg
    db_path = cfg.DB_PATH
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


class TestCashManagement:
    def test_initial_cash(self):
        assert get_cash() == 500.0

    def test_set_cash(self):
        set_cash(123.45)
        assert get_cash() == 123.45


class TestBuySell:
    def test_buy_reduces_cash(self):
        record_buy("AAPL", shares=2.0, price=50.0, reason="test")
        assert get_cash() == pytest.approx(400.0)

    def test_buy_creates_position(self):
        record_buy("AAPL", shares=2.0, price=50.0)
        pos = get_position("AAPL")
        assert pos is not None
        assert pos.shares == 2.0
        assert pos.avg_cost == 50.0

    def test_buy_averages_cost(self):
        record_buy("AAPL", shares=2.0, price=50.0)
        record_buy("AAPL", shares=2.0, price=60.0)
        pos = get_position("AAPL")
        assert pos.shares == 4.0
        assert pos.avg_cost == pytest.approx(55.0)

    def test_sell_increases_cash(self):
        record_buy("AAPL", shares=2.0, price=50.0)
        record_sell("AAPL", shares=2.0, price=60.0)
        # started with 500, spent 100, got back 120
        assert get_cash() == pytest.approx(520.0)

    def test_sell_removes_position(self):
        record_buy("AAPL", shares=2.0, price=50.0)
        record_sell("AAPL", shares=2.0, price=60.0)
        assert get_position("AAPL") is None

    def test_partial_sell(self):
        record_buy("AAPL", shares=4.0, price=50.0)
        record_sell("AAPL", shares=2.0, price=60.0)
        pos = get_position("AAPL")
        assert pos is not None
        assert pos.shares == pytest.approx(2.0)

    def test_buy_insufficient_cash_raises(self):
        with pytest.raises(ValueError, match="Insufficient cash"):
            record_buy("AAPL", shares=100.0, price=100.0)

    def test_sell_more_than_held_raises(self):
        record_buy("AAPL", shares=1.0, price=10.0)
        with pytest.raises(ValueError, match="Cannot sell"):
            record_sell("AAPL", shares=5.0, price=10.0)


class TestTradeHistory:
    def test_trades_recorded(self):
        record_buy("AAPL", shares=1.0, price=100.0)
        record_sell("AAPL", shares=1.0, price=110.0)
        trades = get_trade_history()
        assert len(trades) == 2
        assert trades[0].side == "SELL"  # most recent first
        assert trades[1].side == "BUY"


class TestRiskManagement:
    def test_stop_loss_triggers(self):
        record_buy("AAPL", shares=1.0, price=100.0)
        # 5% loss → price = 95
        assert check_stop_loss("AAPL", 94.0) is True
        assert check_stop_loss("AAPL", 96.0) is False

    def test_take_profit_triggers(self):
        record_buy("AAPL", shares=1.0, price=100.0)
        # 15% gain → price = 115
        assert check_take_profit("AAPL", 116.0) is True
        assert check_take_profit("AAPL", 114.0) is False

    def test_position_limit(self):
        # Portfolio value = 500 (all cash), 20% max = 100
        assert check_position_limit("AAPL", 99.0) is True
        assert check_position_limit("AAPL", 101.0) is False


class TestMultiAssetRiskManagement:
    """Test per-asset-type risk parameters for forex, commodities, and crypto."""

    def test_crypto_stop_loss_uses_wider_threshold(self):
        """Crypto has 8% stop-loss, not the equity default 5%."""
        record_buy("BTC-USD", shares=0.01, price=100.0)
        # 6% drop → within crypto threshold (8%), should NOT trigger
        assert check_stop_loss("BTC-USD", 94.0) is False
        # 9% drop → beyond crypto threshold, should trigger
        assert check_stop_loss("BTC-USD", 91.0) is True

    def test_forex_stop_loss_uses_tighter_threshold(self):
        """Forex has 2% stop-loss."""
        record_buy("EURUSD=X", shares=100.0, price=1.10)
        # 1.5% drop → within forex threshold, should NOT trigger
        assert check_stop_loss("EURUSD=X", 1.0835) is False
        # 2.5% drop → beyond forex threshold, should trigger
        assert check_stop_loss("EURUSD=X", 1.072) is True

    def test_commodity_take_profit(self):
        """Commodities have 10% take-profit."""
        record_buy("GC=F", shares=1.0, price=100.0)
        assert check_take_profit("GC=F", 109.0) is False
        assert check_take_profit("GC=F", 111.0) is True

    def test_crypto_take_profit_uses_wider_threshold(self):
        """Crypto has 20% take-profit."""
        record_buy("ETH-USD", shares=0.1, price=100.0)
        assert check_take_profit("ETH-USD", 119.0) is False
        assert check_take_profit("ETH-USD", 121.0) is True

    def test_forex_position_limit(self):
        """Forex max position is 10% (vs equity 20%)."""
        # Portfolio = 500, forex limit = 50
        assert check_position_limit("EURUSD=X", 49.0) is True
        assert check_position_limit("EURUSD=X", 51.0) is False

    def test_position_asset_type_stored(self):
        """Positions should store asset_type."""
        record_buy("BTC-USD", shares=0.01, price=50000.0)
        pos = get_position("BTC-USD")
        assert pos is not None
        assert pos.asset_type == "crypto"


class TestPortfolioSummary:
    def test_summary_no_positions(self):
        summary = portfolio_summary({})
        assert summary["cash"] == 500.0
        assert summary["total_value"] == 500.0
        assert summary["total_pnl"] == 0.0
        assert summary["num_positions"] == 0

    def test_summary_with_positions(self):
        record_buy("AAPL", shares=2.0, price=50.0)
        # Price went up to 60
        summary = portfolio_summary({"AAPL": 60.0})
        assert summary["cash"] == pytest.approx(400.0)
        assert summary["holdings_value"] == pytest.approx(120.0)
        assert summary["total_value"] == pytest.approx(520.0)
        assert summary["total_pnl"] == pytest.approx(20.0)
        # Holdings should include asset_type
        assert summary["holdings"][0]["asset_type"] == "equity"

    def test_summary_with_crypto_position(self):
        record_buy("BTC-USD", shares=0.001, price=50000.0)
        summary = portfolio_summary({"BTC-USD": 55000.0})
        assert summary["holdings"][0]["asset_type"] == "crypto"
        assert summary["holdings"][0]["display_name"] == "Bitcoin"
