"""Tests for the Flask web application."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Patch DB to temp dir before imports
_tmp = tempfile.mkdtemp()
_patches = [
    mock.patch("stockio.config.DATA_DIR", new=Path(_tmp)),
    mock.patch("stockio.config.DB_PATH", new=Path(_tmp) / "test_web.db"),
    mock.patch("stockio.config.INITIAL_BUDGET_GBP", new=500.0),
    mock.patch("stockio.config.MAX_POSITION_PCT", new=20.0),
    mock.patch("stockio.config.STOP_LOSS_PCT", new=5.0),
    mock.patch("stockio.config.TAKE_PROFIT_PCT", new=15.0),
]
for p in _patches:
    p.start()

from stockio.webapp import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    # Clean DB for each test
    import stockio.config as cfg
    if cfg.DB_PATH.exists():
        cfg.DB_PATH.unlink()
    with app.test_client() as client:
        yield client
    if cfg.DB_PATH.exists():
        cfg.DB_PATH.unlink()


class TestDashboard:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Stockio" in resp.data
        assert b"Dashboard" in resp.data

    def test_index_has_key_elements(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "total-value" in html
        assert "cash-value" in html
        assert "pnl-value" in html
        assert "/api/status" in html


class TestAPI:
    def test_config_endpoint(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["budget"] == 500.0
        assert data["mode"] == "paper"
        assert isinstance(data["watchlist"], list)
        # Multi-asset config fields
        assert "forex_enabled" in data
        assert "commodities_enabled" in data
        assert "crypto_enabled" in data
        assert "forex_risk" in data
        assert "commodity_risk" in data
        assert "crypto_risk" in data

    def test_trades_endpoint_empty(self, client):
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == []

    def test_trades_with_limit(self, client):
        resp = client.get("/api/trades?limit=5")
        assert resp.status_code == 200

    def test_bot_start(self, client):
        # Mock the bot so it doesn't actually run
        with mock.patch("stockio.bot.StockioBot") as mock_bot_cls:
            mock_bot_cls.return_value = mock.MagicMock()
            with mock.patch("stockio.webapp.threading.Thread") as mock_thread:
                mock_thread.return_value = mock.MagicMock()
                resp = client.post("/api/bot/start")
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["status"] in ("started", "already_running")

    def test_bot_stop(self, client):
        resp = client.post("/api/bot/stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "stopped"
