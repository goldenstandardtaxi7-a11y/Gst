import os
from dataclasses import dataclass, field

LIVE_CONFIRMATION = "I_UNDERSTAND_THE_RISK"

# Liquid ETFs across asset classes: US/intl equities, bonds, gold, real estate, commodities.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]


@dataclass
class StrategyConfig:
    universe: list = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    sma_window: int = 200            # trend filter: price must be above its 200-day average
    lookbacks: tuple = (21, 63, 126, 252)  # 1, 3, 6, 12 month momentum
    vol_window: int = 60
    target_vol_per_asset: float = 0.10     # annualised volatility budget per position
    max_weight: float = 0.20               # never more than 20% of equity in one ETF
    max_gross: float = 1.0                 # no leverage


@dataclass
class RiskConfig:
    max_drawdown: float = 0.20        # halt and go to cash if equity falls 20% from peak
    min_trade_weight: float = 0.02    # ignore rebalances smaller than 2% of equity
    max_order_fraction: float = 0.25  # no single order larger than 25% of equity
    cash_buffer: float = 0.02         # keep 2% cash for slippage/fees


@dataclass
class BrokerConfig:
    api_key: str = ""
    secret_key: str = ""
    mode: str = "paper"

    @property
    def trading_url(self):
        return ("https://api.alpaca.markets" if self.mode == "live"
                else "https://paper-api.alpaca.markets")

    data_url: str = "https://data.alpaca.markets"


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def broker_config_from_env():
    mode = os.environ.get("TRADING_MODE", "paper").strip().lower()
    if mode not in ("paper", "live"):
        raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {mode!r}")
    if mode == "live" and os.environ.get("CONFIRM_LIVE", "") != LIVE_CONFIRMATION:
        raise RuntimeError(
            f"Live trading requires CONFIRM_LIVE={LIVE_CONFIRMATION} in the environment.")
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set (see .env.example).")
    return BrokerConfig(api_key=key, secret_key=secret, mode=mode)
