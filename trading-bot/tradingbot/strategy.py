"""Target portfolio weights from daily closes.

Rules (based on published research, parameters fixed up front, not optimised):
- Time-series momentum: average sign of 1/3/6/12-month returns
  (Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum").
- Trend filter: only hold an asset while it trades above its 200-day SMA
  (Faber 2007, "A Quantitative Approach to Tactical Asset Allocation").
- Inverse-volatility sizing so a calm bond ETF and a volatile equity ETF
  contribute similar risk.
"""
import numpy as np
import pandas as pd

from .config import StrategyConfig

TRADING_DAYS = 252


def momentum_score(closes: pd.DataFrame, lookbacks) -> pd.DataFrame:
    signs = [np.sign(closes.pct_change(lb, fill_method=None)) for lb in lookbacks]
    return sum(signs) / len(signs)


def target_weights(closes: pd.DataFrame, cfg: StrategyConfig = None) -> pd.DataFrame:
    """Weights per day; row t uses only data up to and including day t."""
    cfg = cfg or StrategyConfig()
    closes = closes.sort_index()
    score = momentum_score(closes, cfg.lookbacks)
    sma = closes.rolling(cfg.sma_window).mean()
    vol = (closes.pct_change(fill_method=None).rolling(cfg.vol_window).std()
           * np.sqrt(TRADING_DAYS))

    in_trend = (score > 0) & (closes > sma)
    raw = (cfg.target_vol_per_asset / vol).clip(upper=cfg.max_weight)
    # Conviction: full size when all lookbacks agree, smaller when mixed.
    weights = (raw * score.clip(lower=0)).where(in_trend, 0.0).fillna(0.0)

    gross = weights.sum(axis=1)
    scale = (cfg.max_gross / gross).where(gross > cfg.max_gross, 1.0)
    return weights.mul(scale, axis=0)


def latest_target(closes: pd.DataFrame, cfg: StrategyConfig = None) -> pd.Series:
    cfg = cfg or StrategyConfig()
    needed = max(max(cfg.lookbacks), cfg.sma_window) + 1
    if len(closes.dropna(how="all")) < needed:
        raise ValueError(f"Need at least {needed} daily bars, got {len(closes)}")
    return target_weights(closes, cfg).iloc[-1]
