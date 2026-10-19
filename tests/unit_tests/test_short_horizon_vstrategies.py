"""Direction-free strategies V1 (OCO breakout + trailing SL) and V2 (range reversion), owner 2026-10-19.

Written before app/short_horizon/vstrategies.py existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import vstrategies as V
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars

PIP = 1e-4


def sim(rows, order, **kw):
    a = np.array(rows, float)
    return V.simulate_oco(order, a[:, 0], a[:, 1], a[:, 2], a[:, 3], **kw)


def breakout(tp_off=0.0030, sl_off=0.0020, trail=None):
    up = V.Leg(1, "STOP", 1.1010, tp_off, sl_off, trail, relative=True)
    dn = V.Leg(-1, "STOP", 1.0990, tp_off, sl_off, trail, relative=True)
    return V.OcoOrder((up, dn), units=10_000)


# ---------------------------------------------------------------- OCO stop entries
def test_stop_entry_fills_on_touch_and_cancels_the_other_leg():
    f = sim([[1.1000, 1.1012, 1.0995, 1.1011], [1.1011, 1.1015, 1.0980, 1.0985]], breakout())
    assert f.direction == 1 and f.entry_price == 1.1010 and f.entry_bar == 0
    assert f.exit_reason == "TIME" and f.exit_price == 1.0985          # the sell-stop was cancelled


def test_gap_through_a_stop_fills_at_the_open_and_relative_levels_shift():
    f = sim([[1.1020, 1.1025, 1.1015, 1.1022]], breakout())
    assert f.entry_price == 1.1020 and f.leg_tp == pytest.approx(1.1050) and f.leg_sl == pytest.approx(1.1000)


def test_both_legs_in_one_bar_take_the_worse_and_flag_ambiguity():
    f = sim([[1.1000, 1.1012, 1.0988, 1.1000]], breakout(sl_off=0.0010))
    # long at 1.1010 would be stopped at 1.1000 (bar low 1.0988); short at 1.0990 would be stopped at 1.1000
    assert f.ambiguous_entry and f.exit_reason == "SL"
    pnl = f.direction * (f.exit_price - f.entry_price)
    assert pnl == pytest.approx(-0.0010)


def test_trailing_stop_only_tightens_from_the_next_bar_and_is_never_removed():
    order = breakout(tp_off=0.0100, sl_off=0.0020, trail=0.0015)
    rows = [[1.1000, 1.1012, 1.0998, 1.1011],    # fill long at 1.1010, SL 1.0990
            [1.1011, 1.1040, 1.1010, 1.1038],    # close 1.1038 -> SL moves to 1.1023 (effective next bar)
            [1.1038, 1.1039, 1.1020, 1.1025],    # low 1.1020 <= 1.1023 -> trailed stop
            [1.1025, 1.1030, 1.1000, 1.1005]]
    f = sim(rows, order)
    assert f.exit_reason == "SL" and f.exit_price == pytest.approx(1.1023) and f.trailed
    assert f.leg_sl == pytest.approx(1.1023) and f.leg_sl_initial == pytest.approx(1.0990)


# ---------------------------------------------------------------- V2 limits
def test_reversion_limits_need_trade_through_and_take_profit_at_the_anchor():
    buy = V.Leg(1, "LIMIT", 1.0980, 1.1000, 1.0950, None, relative=False)
    sell = V.Leg(-1, "LIMIT", 1.1020, 1.1000, 1.1050, None, relative=False)
    order = V.OcoOrder((buy, sell), units=10_000)
    assert sim([[1.1000, 1.1005, 1.0980, 1.1000]], order, limit_through=PIP) is None   # touch only
    f = sim([[1.1000, 1.1005, 1.0975, 1.0990], [1.0990, 1.1003, 1.0985, 1.1001]], order, limit_through=PIP)
    assert f.direction == 1 and f.entry_price == 1.0980
    assert f.exit_reason == "TP" and f.exit_price == 1.1000


# ---------------------------------------------------------------- order builders
def market(seed=90, segments=8, hours=24):
    bars = synthetic_bars(seed=seed, segments=segments, start="2024-01-07 17:00")
    cal = calendar_for(bars)
    return bars, cal, B.build_blocks(bars, cal, hours)


def test_v1_trades_only_on_predicted_expansion_and_meets_the_rules():
    bars, cal, bl = market()
    sigma = np.full(len(bl), 0.0060)
    p = V.V1Params()
    k = next(i for i in range(len(bl)) if bl.status[i] == B.OK)
    assert V.v1_order(bl, k, 0.0035, 0.0035, sigma[k], p) is not None          # 0.007 / 0.006 = 1.17 >= 1.10
    assert V.v1_order(bl, k, 0.0030, 0.0030, sigma[k], p) is None              # 0.006 / 0.006 = 1.00 < 1.10
    o = V.v1_order(bl, k, 0.0040, 0.0035, sigma[k], p)
    up, dn = o.legs
    r = float(bl.ref[k])
    assert up.entry_price == pytest.approx(r + p.k_entry * 0.0040)
    assert dn.entry_price == pytest.approx(r - p.k_entry * 0.0035)
    assert up.tp_offset == pytest.approx(p.k_tp * 0.0075) and up.trail == pytest.approx(p.k_trail * 0.0075)
    V.check_oco(o, sigma[k], X.OrderRules())


def test_v2_anchor_is_causal_and_trades_only_on_contraction():
    bars, cal, bl = market()
    k = next(i for i in range(len(bl)) if bl.status[i] == B.OK and i > 2)
    a1 = V.causal_anchor(bars, bl, k)
    later = bars.copy()
    later.loc[later.index >= bl.start[k], "CLOSE"] += 0.01
    assert V.causal_anchor(later, bl, k) == a1
    sigma = 0.0060
    p = V.V2Params()
    assert V.v2_order(bars, bl, k, 0.0040, 0.0040, sigma, p) is None           # 0.008/0.006 not a contraction
    o = V.v2_order(bars, bl, k, 0.0020, 0.0020, sigma, p)
    if o is not None:
        for leg in o.legs:
            assert leg.take_profit == pytest.approx(a1)
            band = leg.entry_price
            assert leg.stop_loss == pytest.approx(band - leg.direction * p.m_sl * 0.0020)
        V.check_oco(o, sigma, X.OrderRules())


def test_oco_runs_through_the_backtest_with_sizing_rules_and_full_records():
    bars, cal, bl = market(seed=91, segments=10, hours=6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sigma = np.full(len(bl), 0.0040)
    orders = [V.v1_order(bl, k, 0.0030, 0.0025, sigma[k], V.V1Params()) if bl.status[k] == B.OK else None
              for k in range(len(bl))]
    res = X.run_backtest(bars, bl, orders, X.RealisticProfile(), cal, 10_000.0, split,
                         profile=X.RealisticProfile(), max_orders_per_day=4, ruin_floor=0.0,
                         order_rules=X.OrderRules(), sigma_block=sigma, sizing=X.Sizing())
    assert res.trades
    for t in res.trades:
        for key in ("units", "direction", "entry_price", "stop_loss", "take_profit"):
            assert np.isfinite(t[key])
        assert float(t["units"]).is_integer() and t["units"] > 0


# ---------------------------------------------------------------- range error families
def test_range_error_families():
    U = np.array([0.003, 0.002, 0.004])
    nU = np.array([0.0025, 0.0025, 0.0025])
    z = np.zeros(3)
    np.testing.assert_allclose(V.blend_excursion(U, nU, 0.0, 0.0, z), nU)
    np.testing.assert_allclose(V.blend_excursion(U, nU, 1.0, 0.0, z), U)
    np.testing.assert_allclose(V.blend_excursion(U, nU, 0.5, 0.0, z), (U + nU) / 2)
