"""Configuration loading from .env and TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_toml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Instrument config (loaded from instruments.toml)
# ---------------------------------------------------------------------------

class InstrumentConfig:
    """Per-instrument configuration loaded from instruments.toml."""

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.display_name: str = data["display_name"]
        self.pip_size: float = data["pip_size"]
        self.min_units: int = data["min_units"]
        self.type: str = data["type"]
        self.news_keywords: list[str] = data.get("news_keywords", [])
        self.trading_hours: dict[str, str] = data.get("trading_hours", {})
        self.spread_typical: float = data.get("spread_typical", 0.0)

    def __repr__(self) -> str:
        return f"InstrumentConfig({self.name!r})"


def load_instruments() -> dict[str, InstrumentConfig]:
    """Load all instrument configs from instruments.toml."""
    raw = _load_toml("instruments.toml")
    return {name: InstrumentConfig(name, data) for name, data in raw.items()}


# ---------------------------------------------------------------------------
# Main settings (loaded from .env + settings.toml)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings from .env (secrets) and settings.toml (parameters)."""

    # --- Secrets from .env ---
    oanda_environment: str = "practice"
    oanda_account_id: str = ""
    oanda_api_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    news_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Strategy (from settings.toml) ---
    granularity: str = "M15"
    min_confidence: float = 0.55
    lookback_bars: int = 200

    # --- Risk ---
    risk_per_trade: float = 0.01
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.0
    max_positions: int = 3
    max_leverage: float = 5.0
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.05
    max_drawdown: float = 0.15

    # --- Indicators ---
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_periods: list[int] = Field(default_factory=lambda: [7, 14])
    stochastic_k: int = 14
    stochastic_d: int = 3
    stochastic_smooth: int = 3
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14

    # --- Model ---
    algorithm: str = "lightgbm"
    label_atr_mult: float = 1.0
    label_horizon_bars: int = 4
    n_splits: int = 5
    gap_bars: int = 48
    degradation_window: int = 50
    degradation_threshold: float = 0.40

    # --- Scheduler ---
    cycle_seconds: int = 900
    sentiment_refresh_seconds: int = 3600

    # --- Sentiment / LLM ---
    llm_model: str = "claude-haiku-4-5-20251001"
    max_headlines: int = 10
    news_lookback_hours: int = 24
    rss_feeds: list[str] = Field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/topNews",
    ])

    # --- Monitoring ---
    log_level: str = "INFO"
    daily_summary_hour: int = 22
    heartbeat_seconds: int = 300

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("oanda_environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        if v not in ("practice", "live"):
            raise ValueError("oanda_environment must be 'practice' or 'live'")
        return v

    @model_validator(mode="before")
    @classmethod
    def load_toml_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Merge settings.toml values as defaults (env vars take precedence)."""
        try:
            toml_data = _load_toml("settings.toml")
        except FileNotFoundError:
            return values

        # Flatten nested TOML sections into flat keys
        flat: dict[str, Any] = {}
        for section_data in toml_data.values():
            if isinstance(section_data, dict):
                flat.update(section_data)

        # TOML values are defaults — only set if not already provided
        for key, val in flat.items():
            if key not in values or values[key] is None:
                values[key] = val

        return values

    @property
    def oanda_api_url(self) -> str:
        if self.oanda_environment == "practice":
            return "https://api-fxpractice.oanda.com"
        return "https://api-fxtrade.oanda.com"

    @property
    def oanda_stream_url(self) -> str:
        if self.oanda_environment == "practice":
            return "https://stream-fxpractice.oanda.com"
        return "https://stream-fxtrade.oanda.com"

    @property
    def data_dir(self) -> Path:
        d = _PROJECT_ROOT / "data"
        d.mkdir(exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        d = _PROJECT_ROOT / "models"
        d.mkdir(exist_ok=True)
        return d


def load_settings() -> Settings:
    """Load and return the application settings."""
    return Settings()
