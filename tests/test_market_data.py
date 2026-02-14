"""Tests for market_data module — uses synthetic data to avoid network calls."""

import numpy as np
import pandas as pd
import pytest

from stockio.market_data import add_technical_indicators, build_feature_matrix


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    n = len(dates)  # use actual number of business days
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestTechnicalIndicators:
    def test_adds_expected_columns(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        expected = {"sma_10", "sma_30", "ema_10", "ema_30", "rsi", "macd", "bb_upper", "atr", "obv"}
        assert expected.issubset(set(df.columns))

    def test_no_nans_after_indicators(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        assert not df.isnull().any().any()

    def test_indicator_values_reasonable(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        assert df["rsi"].between(0, 100).all()
        assert (df["bb_upper"] >= df["bb_lower"]).all()


class TestTechnicalIndicatorsWithoutVolume:
    """Test that indicators work for assets without volume data (forex, some commodities)."""

    def test_no_volume_column(self):
        df = _make_ohlcv()
        df["Volume"] = 0  # simulate no volume data
        df = add_technical_indicators(df)
        assert "obv" in df.columns
        assert not df.isnull().any().any()

    def test_zero_volume_uses_close_for_vwap(self):
        df = _make_ohlcv()
        df["Volume"] = 0
        df = add_technical_indicators(df)
        # vwap should be equal to close when no volume
        assert (df["vwap"] == df["Close"]).all()
        assert (df["volume_pct"] == 0.0).all()

    def test_normal_volume_computes_obv(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        # OBV should not be all zeros when volume exists
        assert df["obv"].abs().sum() > 0


class TestFeatureMatrix:
    def test_shape(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        X, y, features = build_feature_matrix(df, forecast_horizon=5)
        assert X.ndim == 2
        assert len(y) == len(X)
        assert len(features) == X.shape[1]

    def test_target_is_binary(self):
        df = _make_ohlcv()
        df = add_technical_indicators(df)
        _, y, _ = build_feature_matrix(df)
        assert set(np.unique(y)).issubset({0, 1})
