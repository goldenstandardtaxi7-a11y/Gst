from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest import run_backtest
from tradingbot.config import LIVE_CONFIRMATION, RiskConfig, StrategyConfig, broker_config_from_env
from tradingbot.data import synthetic_closes
from tradingbot.runner import Order, State, plan_orders, run_cycle
from tradingbot.strategy import target_weights

CFG = StrategyConfig()


@pytest.fixture
def closes():
    return synthetic_closes(CFG.universe, days=800)


def test_weights_respect_limits(closes):
    w = target_weights(closes, CFG)
    assert (w < 0).any().any() and (w > 0).any().any()
    assert (w <= CFG.max_weight + 1e-9).all().all()
    assert (w >= -CFG.max_short_weight - 1e-9).all().all()
    assert (-w.clip(upper=0).sum(axis=1) <= CFG.max_short_gross + 1e-9).all()
    assert (w.abs().sum(axis=1) <= CFG.max_gross + 1e-9).all()


def test_long_only_has_no_shorts(closes):
    assert (target_weights(closes, StrategyConfig(allow_shorts=False)) >= 0).all().all()


def test_no_lookahead(closes):
    full = target_weights(closes, CFG)
    cut = 600
    shocked = closes.copy()
    shocked.iloc[cut + 1:] *= 3  # change the future only
    pd.testing.assert_series_equal(full.iloc[cut], target_weights(shocked, CFG).iloc[cut])


def _falling():
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame({"SPY": 100 * np.exp(-0.001 * np.arange(400))
                         + np.random.default_rng(0).normal(0, 0.1, 400)}, index=idx)


def test_downtrend_goes_short():
    w = target_weights(_falling(), StrategyConfig(universe=["SPY"])).iloc[-1]["SPY"]
    assert -CFG.max_short_weight - 1e-9 <= w < 0


def test_downtrend_long_only_goes_to_cash():
    cfg = StrategyConfig(universe=["SPY"], allow_shorts=False)
    assert target_weights(_falling(), cfg).iloc[-1].sum() == 0


def test_short_profits_in_downtrend_and_pays_borrow():
    closes = _falling()
    cfg = StrategyConfig(universe=["SPY"])
    with_fee = run_backtest(closes, cfg, RiskConfig(short_borrow_fee=0.05), cost_bps=0)
    no_fee = run_backtest(closes, cfg, RiskConfig(short_borrow_fee=0.0), cost_bps=0)
    assert no_fee.equity.iloc[-1] > 1.0
    assert with_fee.equity.iloc[-1] < no_fee.equity.iloc[-1]


def test_costs_reduce_returns(closes):
    cheap = run_backtest(closes, CFG, cost_bps=0).equity.iloc[-1]
    expensive = run_backtest(closes, CFG, cost_bps=50).equity.iloc[-1]
    assert expensive < cheap


PRICES = pd.Series({"SPY": 500.0, "TLT": 90.0, "GLD": 200.0, "EEM": 40.0})


def test_plan_orders_reduces_first_and_never_levers():
    risk = RiskConfig()
    target = pd.Series({"SPY": 0.5, "TLT": 0.4, "GLD": 0.0, "EEM": -0.1})
    orders = plan_orders(target, {"GLD": 3000.0}, PRICES, equity=10000, risk=risk)
    assert orders[0] == Order("GLD", "close", 3000.0)
    assert all(o.notional <= risk.max_order_fraction * 10000 for o in orders[1:])
    gross_after = sum(o.notional for o in orders[1:])
    assert gross_after <= 10000 * (1 - risk.cash_buffer) + 1e-6


def test_short_orders_use_whole_shares():
    orders = plan_orders(pd.Series({"EEM": -0.1}), {}, PRICES, 10000, RiskConfig())
    assert orders == [Order("EEM", "sell", 1000.0, 25)]


def test_cover_short_buys_back_shares():
    orders = plan_orders(pd.Series({"EEM": -0.05}), {"EEM": -1000.0}, PRICES, 10000, RiskConfig())
    assert orders == [Order("EEM", "buy", 480.0, 12)]


def test_flip_closes_first():
    orders = plan_orders(pd.Series({"SPY": -0.1}), {"SPY": 2000.0}, PRICES, 10000, RiskConfig())
    assert orders == [Order("SPY", "close", 2000.0)]


def test_plan_orders_skips_tiny_rebalances():
    orders = plan_orders(pd.Series({"SPY": 0.30}), {"SPY": 2950.0}, PRICES, 10000, RiskConfig())
    assert orders == []


class FakeBroker:
    def __init__(self, equity, closes, is_open=True, shorting=True, borrowable=True):
        self.equity, self.closes, self.is_open = equity, closes, is_open
        self.shorting, self.borrowable = shorting, borrowable
        self.sent, self.closed_all = [], False

    def account(self):
        return {"equity": str(self.equity), "shorting_enabled": self.shorting}

    def can_short(self, sym):
        return self.borrowable

    def submit_qty_order(self, sym, side, qty):
        self.sent.append((sym, side, qty))

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


def test_cycle_places_long_and_short_orders(tmp_path, closes):
    broker = FakeBroker(10000, closes)
    run_cycle(broker, State(str(tmp_path / "s.json")), today=date(2030, 1, 1))
    sides = {side for _, side, _ in broker.sent}
    assert sides == {"buy", "sell"}


@pytest.mark.parametrize("kwargs", [{"shorting": False}, {"borrowable": False}])
def test_cycle_skips_shorts_when_not_allowed(tmp_path, closes, kwargs):
    broker = FakeBroker(10000, closes, **kwargs)
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
