"""One trading cycle: risk checks, target weights, orders."""
import json
import logging
import math
import os
from datetime import date, timedelta
from typing import NamedTuple, Optional

import pandas as pd

from .config import RiskConfig, StrategyConfig
from .strategy import latest_target

log = logging.getLogger("tradingbot")


class Order(NamedTuple):
    symbol: str
    side: str                  # "buy" / "sell" / "close"
    notional: float            # approximate dollar size
    qty: Optional[int] = None  # whole shares (short side); None = notional order


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


def plan_orders(target: pd.Series, positions: dict, prices: pd.Series, equity: float,
                risk: RiskConfig):
    """Orders moving signed exposures (shorts negative) toward target weights.

    Reductions go first. Increases are capped so long + |short| stays within
    equity minus the cash buffer, i.e. no leverage. A position that has to flip
    from long to short (or back) is closed now and reopened on the next cycle,
    because the broker rejects opening the other side while the close is pending.
    """
    reduce, increase = [], []
    exposure = {s: float(v) for s, v in positions.items()}
    min_trade = risk.min_trade_weight * equity
    max_order = risk.max_order_fraction * equity

    for sym in sorted(set(target.index) | set(positions)):
        want = float(target.get(sym, 0.0)) * equity
        have = exposure.get(sym, 0.0)
        if have != 0 and (want == 0 or (want > 0) != (have > 0)):
            reduce.append(Order(sym, "close", abs(have)))
            exposure[sym] = 0.0
            if want != 0:
                log.info("%s flips side: closing now, reopening next cycle", sym)
            continue
        diff = want - have
        if abs(diff) < min_trade:
            continue
        size = min(abs(diff), max_order)
        shrinking = abs(want) < abs(have)
        if want > 0:
            order = Order(sym, "buy" if diff > 0 else "sell", round(size, 2))
        else:
            qty = math.floor(size / float(prices[sym]))
            if qty < 1:
                continue
            order = Order(sym, "buy" if diff > 0 else "sell", round(qty * float(prices[sym]), 2), qty)
        if shrinking:
            reduce.append(order)
            exposure[sym] = have + (order.notional if diff > 0 else -order.notional)
        else:
            increase.append(order)

    budget = equity * (1 - risk.cash_buffer) - sum(abs(v) for v in exposure.values())
    capped = []
    for o in sorted(increase, key=lambda o: -o.notional):
        if o.qty is None:
            notional = min(o.notional, budget)
            if notional >= 1.0:
                capped.append(o._replace(notional=round(notional, 2)))
                budget -= notional
        else:
            price = o.notional / o.qty
            qty = min(o.qty, math.floor(max(budget, 0) / price))
            if qty >= 1:
                capped.append(o._replace(qty=qty, notional=round(qty * price, 2)))
                budget -= qty * price
    return reduce + capped


def drop_unshortable(target: pd.Series, broker, account) -> pd.Series:
    target = target.copy()
    shorts = target[target < 0].index
    if len(shorts) and not account.get("shorting_enabled", False):
        log.warning("Shorting is not enabled on this account (needs a margin account with "
                    ">= $2,000); skipping shorts %s", list(shorts))
        target[shorts] = 0.0
        return target
    for sym in shorts:
        if not broker.can_short(sym):
            log.warning("%s is not easy to borrow; skipping short", sym)
            target[sym] = 0.0
    return target


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
    equity = float(account["equity"])
    state.data["peak_equity"] = max(state.data["peak_equity"], equity)
    drawdown = 1 - equity / state.data["peak_equity"] if state.data["peak_equity"] else 0.0
    log.info("Equity %.2f, drawdown %.1f%%", equity, drawdown * 100)

    if drawdown >= risk.max_drawdown:
        reason = f"drawdown {drawdown:.1%} >= limit {risk.max_drawdown:.0%}"
        log.error("KILL SWITCH: %s. Closing all positions.", reason)
        if not dry_run:
            broker.close_all_positions()
            state.data.update(halted=True, halt_reason=reason)
            state.save()
        return [Order("ALL", "close", equity)]

    if not dry_run and not broker.clock().get("is_open"):
        log.info("Market is closed; nothing to do.")
        state.save()
        return []

    start = (today - timedelta(days=600)).isoformat()
    closes = broker.daily_closes(strategy.universe, start)
    closes = closes[closes.index.date < today]  # only completed daily bars
    target = latest_target(closes, strategy) * (1 - risk.cash_buffer)
    target = drop_unshortable(target, broker, account)
    log.info("Target weights: %s", {k: round(v, 3) for k, v in target.items() if v != 0} or "all cash")

    orders = plan_orders(target, broker.positions(), closes.iloc[-1], equity, risk)
    for o in orders:
        size = f"{o.qty} sh (~${o.notional:.2f})" if o.qty else f"${o.notional:.2f}"
        log.info("%s %s %s %s", "DRY-RUN" if dry_run else "ORDER", o.side.upper(), o.symbol, size)
        if dry_run:
            continue
        if o.side == "close":
            broker.close_position(o.symbol)
        elif o.qty:
            broker.submit_qty_order(o.symbol, o.side, o.qty)
        else:
            broker.submit_notional_order(o.symbol, o.side, o.notional)
    state.save()
    return orders
