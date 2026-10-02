"""One trading cycle: risk checks, target weights, orders."""
import json
import logging
import os
from datetime import date, timedelta

import pandas as pd

from .config import RiskConfig, StrategyConfig
from .strategy import latest_target

log = logging.getLogger("tradingbot")


class State:
    def __init__(self, path):
        self.path = path
        self.data = {"peak_equity": 0.0, "halted": False, "halt_reason": ""}
        if os.path.exists(path):
            with open(path) as f:
                self.data.update(json.load(f))

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def plan_orders(target: pd.Series, positions: dict, equity: float, cash: float,
                risk: RiskConfig):
    """Return [(symbol, side, notional, close_all)] with sells before buys."""
    symbols = sorted(set(target.index) | set(positions))
    sells, buys = [], []
    for sym in symbols:
        want = float(target.get(sym, 0.0)) * equity
        have = float(positions.get(sym, 0.0))
        diff = want - have
        if want <= 0 and have > 0:
            sells.append((sym, "sell", have, True))
        elif abs(diff) >= risk.min_trade_weight * equity:
            notional = min(abs(diff), risk.max_order_fraction * equity)
            (buys if diff > 0 else sells).append((sym, "buy" if diff > 0 else "sell", notional, False))

    budget = cash + sum(n for _, _, n, _ in sells) - risk.cash_buffer * equity
    capped = []
    for sym, side, notional, close in sorted(buys, key=lambda o: -o[2]):
        notional = min(notional, budget)
        if notional >= 1.0:
            capped.append((sym, side, round(notional, 2), close))
            budget -= notional
    return sells + capped


def run_cycle(broker, state: State, strategy: StrategyConfig = None, risk: RiskConfig = None,
              dry_run: bool = False, today: date = None):
    strategy = strategy or StrategyConfig()
    risk = risk or RiskConfig()
    today = today or date.today()

    if state.data["halted"]:
        log.warning("Bot is halted (%s). Run 'reset-halt' after reviewing.", state.data["halt_reason"])
        return []

    account = broker.account()
    if account.get("trading_blocked") or account.get("account_blocked"):
        log.error("Account is blocked by the broker; doing nothing.")
        return []
    equity, cash = float(account["equity"]), float(account["cash"])
    state.data["peak_equity"] = max(state.data["peak_equity"], equity)
    drawdown = 1 - equity / state.data["peak_equity"] if state.data["peak_equity"] else 0.0
    log.info("Equity %.2f, cash %.2f, drawdown %.1f%%", equity, cash, drawdown * 100)

    if drawdown >= risk.max_drawdown:
        reason = f"drawdown {drawdown:.1%} >= limit {risk.max_drawdown:.0%}"
        log.error("KILL SWITCH: %s. Closing all positions.", reason)
        if not dry_run:
            broker.close_all_positions()
            state.data.update(halted=True, halt_reason=reason)
            state.save()
        return [("ALL", "close", equity, True)]

    if not dry_run and not broker.clock().get("is_open"):
        log.info("Market is closed; nothing to do.")
        state.save()
        return []

    start = (today - timedelta(days=600)).isoformat()
    closes = broker.daily_closes(strategy.universe, start)
    closes = closes[closes.index.date < today]  # only completed daily bars
    target = latest_target(closes, strategy) * (1 - risk.cash_buffer)
    log.info("Target weights: %s", {k: round(v, 3) for k, v in target.items() if v > 0} or "all cash")

    orders = plan_orders(target, broker.positions(), equity, cash, risk)
    for sym, side, notional, close in orders:
        log.info("%s %s %s $%.2f", "DRY-RUN" if dry_run else "ORDER", side.upper(), sym, notional)
        if dry_run:
            continue
        if close:
            broker.close_position(sym)
        else:
            broker.submit_notional_order(sym, side, notional)
    state.save()
    return orders
