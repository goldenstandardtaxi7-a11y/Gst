from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest import run_backtest
from tradingbot.config import LIVE_CONFIRMATION, RiskConfig, StrategyConfig, broker_config_from_env
from tradingbot.data import synthetic_closes
from tradingbot.runner import State, plan_orders, run_cycle
from tradingbot.strategy import target_weights

CFG = StrategyConfig()


@pytest.fixture
def closes():
    return synthetic_closes(CFG.universe, days=800)


def test_weights_respect_limits(closes):
    w = target_weights(closes, CFG)
    assert (w >= 0).all().all()
    assert (w <= CFG.max_weight + 1e-9).all().all()
    assert (w.sum(axis=1) <= CFG.max_gross + 1e-9).all()


def test_no_lookahead(closes):
    full = target_weights(closes, CFG)
    cut = 600
    shocked = closes.copy()
    shocked.iloc[cut + 1:] *= 3  # change the future only
    pd.testing.assert_series_equal(full.iloc[cut], target_weights(shocked, CFG).iloc[cut])


def test_downtrend_goes_to_cash():
    idx = pd.bdate_range("2020-01-01", periods=400)
    falling = pd.DataFrame({"SPY": 100 * np.exp(-0.001 * np.arange(400))
                            + np.random.default_rng(0).normal(0, 0.1, 400)}, index=idx)
    assert target_weights(falling, StrategyConfig(universe=["SPY"])).iloc[-1].sum() == 0


def test_costs_reduce_returns(closes):
    cheap = run_backtest(closes, CFG, cost_bps=0).equity.iloc[-1]
    expensive = run_backtest(closes, CFG, cost_bps=50).equity.iloc[-1]
    assert expensive < cheap


def test_plan_orders_sells_first_and_respects_cash():
    risk = RiskConfig()
    target = pd.Series({"SPY": 0.5, "TLT": 0.4, "GLD": 0.0})
    orders = plan_orders(target, {"GLD": 3000.0}, equity=10000, cash=7000, risk=risk)
    assert orders[0] == ("GLD", "sell", 3000.0, True)
    buys = [o for o in orders if o[1] == "buy"]
    assert all(o[2] <= risk.max_order_fraction * 10000 for o in buys)
    assert sum(o[2] for o in buys) <= 7000 + 3000 - risk.cash_buffer * 10000


def test_plan_orders_skips_tiny_rebalances():
    orders = plan_orders(pd.Series({"SPY": 0.30}), {"SPY": 2950.0}, 10000, 7000, RiskConfig())
    assert orders == []


class FakeBroker:
    def __init__(self, equity, closes, is_open=True):
        self.equity, self.closes, self.is_open = equity, closes, is_open
        self.sent, self.closed_all = [], False

    def account(self):
        return {"equity": str(self.equity), "cash": str(self.equity)}

    def clock(self):
        return {"is_open": self.is_open}

    def positions(self):
        return {}

    def daily_closes(self, symbols, start):
        return self.closes

    def submit_notional_order(self, sym, side, notional):
        self.sent.append((sym, side, notional))

    def close_position(self, sym):
        self.sent.append((sym, "close", None))

    def close_all_positions(self):
        self.closed_all = True


def test_kill_switch_liquidates_and_halts(tmp_path, closes):
    state = State(str(tmp_path / "s.json"))
    state.data["peak_equity"] = 10000
    broker = FakeBroker(7500, closes)
    run_cycle(broker, state)
    assert broker.closed_all and state.data["halted"]
    broker2 = FakeBroker(10000, closes)
    assert run_cycle(broker2, State(str(tmp_path / "s.json"))) == []
    assert broker2.sent == []


def test_cycle_places_orders_when_open(tmp_path, closes):
    broker = FakeBroker(10000, closes)
    run_cycle(broker, State(str(tmp_path / "s.json")), today=date(2030, 1, 1))
    assert broker.sent and all(side == "buy" for _, side, _ in broker.sent)


def test_cycle_does_nothing_when_market_closed(tmp_path, closes):
    broker = FakeBroker(10000, closes, is_open=False)
    run_cycle(broker, State(str(tmp_path / "s.json")))
    assert broker.sent == []


def test_live_requires_confirmation(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.delenv("CONFIRM_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        broker_config_from_env()
    monkeypatch.setenv("CONFIRM_LIVE", LIVE_CONFIRMATION)
    assert broker_config_from_env().trading_url == "https://api.alpaca.markets"
