"""Tests for the strategy module."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from stockio.strategy import Signal, TradeSignal


class TestSignalEnum:
    def test_values(self):
        assert Signal.BUY == "BUY"
        assert Signal.SELL == "SELL"
        assert Signal.HOLD == "HOLD"


class TestTradeSignal:
    def test_creation(self):
        sig = TradeSignal(
            ticker="AAPL",
            signal=Signal.BUY,
            confidence=0.75,
            reasons=["test reason"],
        )
        assert sig.ticker == "AAPL"
        assert sig.signal == Signal.BUY
        assert sig.confidence == 0.75


class TestModelPersistence:
    def test_save_and_load(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with (
                mock.patch("stockio.strategy._MODEL_PATH", tmpdir / "model.joblib"),
                mock.patch("stockio.strategy._SCALER_PATH", tmpdir / "scaler.joblib"),
                mock.patch("stockio.strategy._META_PATH", tmpdir / "meta.json"),
                mock.patch("stockio.strategy.config.MODEL_DIR", tmpdir),
            ):
                from stockio.strategy import _load_model, _save_model

                model = GradientBoostingClassifier(n_estimators=10)
                scaler = StandardScaler()

                # Fit on dummy data
                X = np.random.randn(50, 5)
                y = (X[:, 0] > 0).astype(int)
                scaler.fit(X)
                model.fit(scaler.transform(X), y)

                _save_model(model, scaler, ["f1", "f2", "f3", "f4", "f5"], 0.85)

                loaded_model, loaded_scaler, meta = _load_model()
                assert loaded_model is not None
                assert loaded_scaler is not None
                assert meta["accuracy"] == 0.85
                assert meta["features"] == ["f1", "f2", "f3", "f4", "f5"]

                # Check prediction works
                pred = loaded_model.predict(loaded_scaler.transform(X[:1]))
                assert pred[0] in (0, 1)
