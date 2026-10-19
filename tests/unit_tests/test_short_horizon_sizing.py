"""Business rule (owner, 2026-10-19): volume in integer units from rel_volume of the current balance.

units_max = floor(balance_USD * rel_volume * leverage / price_USD_per_EUR)   (basis "margin", default)
units_max = floor(balance_USD * rel_volume / price)                         (basis "notional")
units     = floor(f * units_max), f in (0, 1]; below the broker minimum -> no trade (counted).
Written before execution.Sizing and the order-record checks existed.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import strategies as S
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars


def test_units_max_currency_conversion_and_bases():
    s = X.Sizing(rel_volume=0.10, leverage=100.0, basis="margin")
    assert s.units_max(10_000.0, 1.10) == 90_909           # floor(10 000 * 0.10 * 100 / 1.10) EUR units
    assert X.Sizing(rel_volume=0.10, leverage=100.0, basis="notional").units_max(10_000.0, 1.10) == 909
    assert s.units(10_000.0, 1.10, fraction=0.5) == 45_454
    assert s.units(10_000.0, 1.10, fraction=1.0) <= s.units_max(10_000.0, 1.10)
    with pytest.raises(ValueError):
        X.Sizing(rel_volume=0.30)                          # declared bounds (0, 0.20]
    with pytest.raises(ValueError):
        s.units(10_000.0, 1.10, fraction=1.5)
    assert X.Sizing.manifest()["rel_volume"]["bounds"] == [0.0, 0.2]


def market_orders(bl, tp_off=0.003, sl_off=0.003, fraction=1.0):
    return [X.BlockOrder(1, "MARKET", None, float(bl.ref[k]) + tp_off, float(bl.ref[k]) - sl_off, None, fraction)
            if bl.status[k] == B.OK else None for k in range(len(bl))]


def test_sizing_compounds_on_current_balance_and_records_every_field():
    bars = synthetic_bars(seed=80, segments=4)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sz = X.Sizing(rel_volume=0.10, leverage=100.0)
    res = X.run_backtest(bars, bl, market_orders(bl), X.Costs(), cal, 10_000.0, split, sizing=sz)
    bal = 10_000.0
    for t in sorted(res.trades, key=lambda r: r["entry_time"]):
        expected = sz.units(bal, float(bl.ref[t["block"]]), 1.0)
        assert t["units"] == expected and float(t["units"]).is_integer()
        for key in ("units", "direction", "entry_price", "stop_loss", "take_profit"):
            assert key in t and np.isfinite(t[key])
        bal += t["net_usd"]
    m = res.metrics
    assert m["rel_volume"] == 0.10 and m["leverage"] == 100.0 and m["rel_volume_basis"] == "margin"
    assert m["max_margin_used_pct"] == pytest.approx(10.0, rel=0.02)
    assert m["max_exposure_x_balance"] == pytest.approx(10.0, rel=0.02)
    assert "margin_call_events" in m


def test_below_broker_minimum_is_not_traded_and_counted():
    bars = synthetic_bars(seed=81, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sz = X.Sizing(rel_volume=0.01, leverage=1.0, basis="notional", min_units=1_000)  # 100 USD -> ~90 units
    res = X.run_backtest(bars, bl, market_orders(bl), X.Costs(), cal, 10_000.0, split, sizing=sz)
    assert res.metrics["trades"] == 0 and res.metrics["below_min_units"] > 0


@pytest.mark.parametrize("bad", [dict(direction=0), dict(units=None), dict(units=0), dict(units=10.5)])
def test_order_missing_units_or_side_is_refused_without_sizing(bad):
    base = dict(direction=1, entry_type="MARKET", entry_price=None, take_profit=1.1030, stop_loss=1.0980,
                units=10_000)
    base.update(bad)
    with pytest.raises(X.OrderRuleError):
        X.check_order(X.BlockOrder(**base), 1.1000, 0.0050, X.OrderRules(), require_units=True)


def test_margin_call_events_are_reported():
    from tests.unit_tests.test_short_horizon_ruin import falling_market
    bars = falling_market()   # every long loses ~0.15 % per hour: a 100x notional reaches closeout
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sz = X.Sizing(rel_volume=0.20, leverage=500.0, basis="margin", leverage_bound=1000.0)  # 100x notional
    res = X.run_backtest(bars, bl, market_orders(bl, 0.02, 0.02), X.Costs(), cal, 10_000.0, split, sizing=sz)
    assert res.metrics["margin_call_events"] >= 1
