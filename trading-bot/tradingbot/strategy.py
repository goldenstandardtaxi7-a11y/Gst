"""Target portfolio weights from daily closes.

Rules (based on published research, parameters fixed up front, not optimised):
- Time-series momentum: average sign of 1/3/6/12-month returns
  (Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum").
- Trend filter: only hold an asset while it trades above its 200-day SMA
  (Faber 2007, "A Quantitative Approach to Tactical Asset Allocation").
- Short side: the same rules mirrored. An asset whose momentum is negative and
  which trades below its 200-day SMA is sold short (smaller size cap).
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

    raw = cfg.target_vol_per_asset / vol
    # Conviction: full size when all lookbacks agree, smaller when mixed.
    up = (score > 0) & (closes > sma)
    longs = (raw.clip(upper=cfg.max_weight) * score).where(up, 0.0).fillna(0.0)

    if cfg.allow_shorts:
        down = (score < 0) & (closes < sma)
        shorts = (raw.clip(upper=cfg.max_short_weight) * score).where(down, 0.0).fillna(0.0)
        short_gross = -shorts.sum(axis=1)
        cap = (cfg.max_short_gross / short_gross).where(short_gross > cfg.max_short_gross, 1.0)
        shorts = shorts.mul(cap, axis=0)
    else:
        shorts = longs * 0.0

    weights = longs + shorts
    gross = weights.abs().sum(axis=1)
    scale = (cfg.max_gross / gross).where(gross > cfg.max_gross, 1.0)
    return weights.mul(scale, axis=0)


def latest_target(closes: pd.DataFrame, cfg: StrategyConfig = None) -> pd.Series:
    cfg = cfg or StrategyConfig()
    needed = max(max(cfg.lookbacks), cfg.sma_window) + 1
    if len(closes.dropna(how="all")) < needed:
        raise ValueError(f"Need at least {needed} daily bars, got {len(closes)}")
    return target_weights(closes, cfg).iloc[-1]
