"""Daily backtest: weights decided at close of day t are traded at close of day t+1."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskConfig, StrategyConfig
from .strategy import TRADING_DAYS, target_weights


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    stats: dict


def compute_stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    if years <= 0:
        return {}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()
    return {"years": round(years, 1), "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
            "cagr": cagr, "volatility": vol, "sharpe": sharpe, "max_drawdown": max_dd}


def run_backtest(closes: pd.DataFrame, strategy: StrategyConfig = None,
                 risk: RiskConfig = None, cost_bps: float = 10.0,
                 rebalance_every: int = 5) -> BacktestResult:
    strategy = strategy or StrategyConfig()
    risk = risk or RiskConfig()
    closes = closes.sort_index().ffill()
    targets = target_weights(closes, strategy)
    rets = closes.pct_change(fill_method=None).fillna(0.0)

    held = pd.Series(0.0, index=closes.columns)
    equity, eq_path, w_path, turn_path = 1.0, [], [], []
    pending = None
    for i, day in enumerate(closes.index):
        # Mark to market with the positions held overnight.
        day_ret = float((held * rets.loc[day]).sum())
        equity *= 1 + day_ret
        if equity > 0:
            held = held * (1 + rets.loc[day]) / (1 + day_ret)

        turnover = 0.0
        if pending is not None:
            diff = pending - held
            diff = diff.where(diff.abs() >= risk.min_trade_weight, 0.0)
            turnover = float(diff.abs().sum())
            equity *= 1 - turnover * cost_bps / 1e4
            held = held + diff
            pending = None
        if i % rebalance_every == 0:
            pending = targets.loc[day] * (1 - risk.cash_buffer)

        eq_path.append(equity)
        w_path.append(held.copy())
        turn_path.append(turnover)

    equity_s = pd.Series(eq_path, index=closes.index)
    return BacktestResult(equity=equity_s, weights=pd.DataFrame(w_path, index=closes.index),
                          turnover=pd.Series(turn_path, index=closes.index),
                          stats=compute_stats(equity_s))


def buy_and_hold(closes: pd.Series) -> pd.Series:
    closes = closes.dropna()
    return closes / closes.iloc[0]
