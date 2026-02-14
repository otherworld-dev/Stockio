"""Tests for the paper executor."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Patch DB to temp dir before imports
_tmp = tempfile.mkdtemp()
_patches = [
    mock.patch("stockio.config.DATA_DIR", new=Path(_tmp)),
    mock.patch("stockio.config.DB_PATH", new=Path(_tmp) / "test_exec.db"),
    mock.patch("stockio.config.INITIAL_BUDGET_GBP", new=500.0),
    mock.patch("stockio.config.MAX_POSITION_PCT", new=20.0),
    mock.patch("stockio.config.STOP_LOSS_PCT", new=5.0),
    mock.patch("stockio.config.TAKE_PROFIT_PCT", new=15.0),
]
for p in _patches:
    p.start()

from stockio.executor import PaperExecutor
from stockio.portfolio import get_cash, get_position, record_buy
from stockio.strategy import Signal, TradeSignal


@pytest.fixture(autouse=True)
def _fresh_db():
    import stockio.config as cfg
    db_path = cfg.DB_PATH
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


class TestPaperExecutor:
    def test_hold_does_nothing(self):
        executor = PaperExecutor()
        sig = TradeSignal(ticker="AAPL", signal=Signal.HOLD, confidence=0.5)
        result = executor.execute(sig, current_price=150.0)
        assert result is None
        assert get_cash() == 500.0

    def test_buy_signal(self):
        executor = PaperExecutor()
        sig = TradeSignal(
            ticker="AAPL", signal=Signal.BUY, confidence=0.8,
            reasons=["test buy"],
        )
        result = executor.execute(sig, current_price=50.0)
        assert result is not None
        assert result.side == "BUY"
        assert get_cash() < 500.0
        assert get_position("AAPL") is not None

    def test_sell_signal_with_position(self):
        executor = PaperExecutor()
        record_buy("AAPL", shares=2.0, price=50.0)
        sig = TradeSignal(
            ticker="AAPL", signal=Signal.SELL, confidence=0.7,
            reasons=["test sell"],
        )
        result = executor.execute(sig, current_price=60.0)
        assert result is not None
        assert result.side == "SELL"
        assert get_position("AAPL") is None

    def test_sell_signal_no_position(self):
        executor = PaperExecutor()
        sig = TradeSignal(ticker="AAPL", signal=Signal.SELL, confidence=0.7)
        result = executor.execute(sig, current_price=60.0)
        assert result is None

    def test_stop_loss_exit(self):
        executor = PaperExecutor()
        record_buy("AAPL", shares=2.0, price=100.0)
        # Price dropped 6% — should trigger 5% stop-loss
        result = executor.check_exits("AAPL", current_price=93.0)
        assert result is not None
        assert result.side == "SELL"
        assert "Stop-loss" in result.reason

    def test_take_profit_exit(self):
        executor = PaperExecutor()
        record_buy("AAPL", shares=2.0, price=100.0)
        # Price up 16% — should trigger 15% take-profit
        result = executor.check_exits("AAPL", current_price=116.0)
        assert result is not None
        assert result.side == "SELL"
        assert "Take-profit" in result.reason

    def test_no_exit_in_range(self):
        executor = PaperExecutor()
        record_buy("AAPL", shares=2.0, price=100.0)
        result = executor.check_exits("AAPL", current_price=105.0)
        assert result is None

    def test_buy_crypto_uses_smaller_position(self):
        """Crypto max position is 10% vs equity 20%, so smaller buy."""
        executor = PaperExecutor()
        sig_equity = TradeSignal(
            ticker="AAPL", signal=Signal.BUY, confidence=1.0,
            reasons=["test"],
        )
        result_eq = executor.execute(sig_equity, current_price=50.0)
        assert result_eq is not None
        equity_spend = result_eq.total

        # Reset
        from stockio.portfolio import record_sell
        record_sell("AAPL", result_eq.shares, 50.0)

        sig_crypto = TradeSignal(
            ticker="BTC-USD", signal=Signal.BUY, confidence=1.0,
            reasons=["test"],
        )
        result_cr = executor.execute(sig_crypto, current_price=50000.0)
        assert result_cr is not None
        # Crypto position should be smaller (10% vs 20% of portfolio)
        assert result_cr.total < equity_spend

    def test_crypto_stop_loss_wider(self):
        """Crypto uses 8% stop-loss, so 6% drop shouldn't trigger."""
        executor = PaperExecutor()
        record_buy("BTC-USD", shares=0.001, price=50000.0)
        # 6% drop is within crypto threshold
        result = executor.check_exits("BTC-USD", current_price=47000.0)
        assert result is None

    def test_crypto_stop_loss_triggers(self):
        """Crypto 8% stop-loss should trigger at >8% drop."""
        executor = PaperExecutor()
        record_buy("BTC-USD", shares=0.001, price=50000.0)
        # 9% drop exceeds crypto 8% threshold
        result = executor.check_exits("BTC-USD", current_price=45500.0)
        assert result is not None
        assert result.side == "SELL"
