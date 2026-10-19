"""Owner realism pass 2026-10-19: model-like path errors and realistic execution.

Written before execution.RealisticProfile, the realistic simulate_block options,
noise.blend_path / noise.random_walk_error and the ``pathr`` grid existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars
import tools.short_horizon_oracle as H

PIP = 1e-4


# ---------------------------------------------------------------- spread schedule
def test_spread_schedule_by_hours_since_rollover_and_weekly_open():
    prof = X.RealisticProfile()
    o = pd.Timestamp("2024-01-07 17:00", tz="UTC").value
    at = lambda hours: prof.spread_at(o + int(hours * HOUR.value), o)  # noqa: E731
    assert at(0.0) == prof.weekly_open_spread            # first hour after the weekly open
    assert at(24.0) == prof.spread_for_hour(0)            # first hour after a daily rollover
    assert at(24 + 23) == prof.spread_for_hour(23)        # last hour before the next rollover
    assert at(24 + 12) == prof.spread_for_hour(12)        # core hours
    assert prof.spread_for_hour(12) < prof.spread_for_hour(23) < prof.spread_for_hour(0)
    assert X.RealisticProfile().scaled(2.0).spread_for_hour(12) == pytest.approx(2 * prof.spread_for_hour(12))


# ---------------------------------------------------------------- limit trade-through and latency
def run(rows, order, **kw):
    a = np.array(rows, float)
    return X.simulate_block(order, a[:, 0], a[:, 1], a[:, 2], a[:, 3], **kw)


def test_limit_entry_needs_trade_through():
    order = X.BlockOrder(1, "LIMIT", 1.0980, 1.1100, 1.0900, 10000)
    touch = [[1.1000, 1.1010, 1.0980, 1.1000], [1.1000, 1.1005, 1.0990, 1.1000]]
    assert run(touch, order) is not None                               # legacy: a touch fills
    assert run(touch, order, limit_through=PIP) is None                # realistic: a touch does not
    through = [[1.1000, 1.1010, 1.0979, 1.1000]]
    assert run(through, order, limit_through=PIP) is None              # 0.1 pip through is not enough
    through = [[1.1000, 1.1010, 1.0970, 1.1000]]
    f = run(through, order, limit_through=PIP)
    assert f.entry_price == 1.0980 and f.entry_kind == "LIMIT"


def test_take_profit_needs_trade_through():
    order = X.BlockOrder(1, "MARKET", None, 1.1050, 1.0900, 10000)
    rows = [[1.1000, 1.1050, 1.0990, 1.1000], [1.1000, 1.1005, 1.0995, 1.1001]]
    assert run(rows, order).exit_reason == "TP"
    f = run(rows, order, limit_through=PIP)
    assert f.exit_reason == "TIME"
    rows[0][1] = 1.1060
    assert run(rows, order, limit_through=PIP).exit_reason == "TP"


def test_one_bar_latency_enters_at_second_open():
    order = X.BlockOrder(1, "MARKET", None, 1.2000, 1.0000, 10000)
    rows = [[1.1000, 1.1010, 1.0990, 1.1005], [1.1005, 1.1010, 1.1000, 1.1008]]
    assert run(rows, order).entry_price == 1.1000
    f = run(rows, order, entry_delay=1)
    assert f.entry_bar == 1 and f.entry_price == 1.1005
    assert run(rows[:1], order, entry_delay=1) is None


# ---------------------------------------------------------------- swap and a hand-calculated round trip
def test_realistic_round_trip_costs_and_triple_swap_by_side():
    prof = X.RealisticProfile()
    bars = synthetic_bars(seed=1, segments=2, hours=120)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 24)
    seg = cal.segments[0]
    # the business-day block that ends on the configured triple rollover
    rolls = cal.rollover_ns(0)
    triple = rolls[prof.triple_swap_rollover_index - 1]
    k = next(i for i in range(len(bl)) if bl.end[i].value == triple and bl.segment[i] == 0)
    units = 10_000
    split = (bars.index[0], bars.index[-1] + HOUR)
    out = {}
    for d in (1, -1):
        orders = [None] * len(bl)
        orders[k] = X.BlockOrder(d, "MARKET", None, d * 50.0 if d > 0 else 0.0001, 0.0001 if d > 0 else 50.0, units)
        res = X.run_backtest(bars, bl, orders, prof, cal, 10_000.0, split, profile=prof)
        out[d] = res.trades[0]
    t = out[1]
    lots = units / 1e5
    entry_h = ((t["entry_time"].value - seg.market_open.value) // HOUR.value) % 24
    exit_label = bars.index[t["exit_bar_abs"]].value
    exit_h = ((exit_label - seg.market_open.value) // HOUR.value) % 24
    assert t["entry_time"] == bl.start[k] + HOUR                       # one-bar latency
    assert t["spread_usd"] == pytest.approx(units * (prof.spread_for_hour(entry_h) + prof.spread_for_hour(exit_h)) / 2)
    assert t["slippage_usd"] == pytest.approx(units * 2 * prof.slippage_market)
    assert t["commission_usd"] == pytest.approx(2 * prof.commission_per_lot_side * lots)
    assert t["financing_usd"] == pytest.approx(-3 * prof.swap_long_per_lot * lots)    # triple, long pays
    assert out[-1]["financing_usd"] == pytest.approx(-3 * prof.swap_short_per_lot * lots)  # short receives
    assert t["net_usd"] == pytest.approx(t["gross_usd"] - t["spread_usd"] - t["slippage_usd"]
                                         - t["commission_usd"] - t["financing_usd"])


def test_profile_round_trip_estimate_is_explicit():
    prof = X.RealisticProfile()
    assert 0 < prof.round_trip_price() < X.Costs().round_trip_price()


# ---------------------------------------------------------------- model-like error families
def test_blend_path_and_random_walk_error():
    T = np.array([[1.1, 1.2, np.nan], [1.0, 0.9, 0.8]])
    ref = np.array([1.0, 1.0])
    z = np.zeros_like(T)
    np.testing.assert_allclose(N.blend_path(T, ref, 0.0, 0.0, z)[1], [1.0, 1.0, 1.0])     # w = 0 is persistence
    np.testing.assert_allclose(N.blend_path(T, ref, 1.0, 0.0, z)[1], T[1])                # w = 1 is the truth
    assert np.isnan(N.blend_path(T, ref, 0.5, 0.0, z)[0, 2])
    rng = np.random.default_rng(0)
    zz = rng.standard_normal((20000, 24))
    rw = N.random_walk_error(zz, 1.0)
    np.testing.assert_allclose(rw.std(axis=0)[[0, 3, 15]], np.sqrt([1, 4, 16]), rtol=0.05)
    assert np.corrcoef(rw[:, 10], rw[:, 11])[0, 1] > 0.9                                # autocorrelated along h
    a, b = N.path_draws(42, 5, 24, stream=1), N.path_draws(42, 5, 24, stream=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, N.path_draws(42, 5, 24))                               # separate stream
    np.testing.assert_array_equal(N.path_draws(42, 5, 24), N.path_draws(42, 5, 24, stream=0))


def test_pathr_grid():
    cells = H.cells("pathr")
    fams = {c["family"] for c in cells if c["strategy"] in ("RF-M", "RF-L")}
    assert fams == {"iid", "a1", "a2", "a3"}
    assert {c["sweep"] for c in cells} == {"legacy", "realistic"}
    legacy = [c for c in cells if c["sweep"] == "legacy" and c["strategy"] != "PERSISTENCE"]
    assert {c["family"] for c in legacy} == {"iid"} and {c["cost_mult"] for c in legacy} == {1.0}
    real = [c for c in cells if c["sweep"] == "realistic"]
    assert {c["cost_mult"] for c in real} == {1.0, 2.0}
    assert len({c["seed"] for c in real if c["strategy"] != "PERSISTENCE"}) >= 5
    assert {c["block_hours"] for c in cells} == {6, 24}
    keys = [H.cell_key(c) + (c["sweep"],) for c in cells]
    assert len(keys) == len(set(keys))


def test_pathr_a1_at_zero_weight_is_the_naive_and_rarely_trades():
    bars = synthetic_bars(seed=33, segments=8, start="2023-12-03 17:00")
    ctx = H.Context(bars, calendar_for(bars), 24)
    cell = dict(grid="pathr", sweep="realistic", block_hours=24, strategy="RF-M", family="a1", level=0.0,
                p_flip=0.0, seed=42, cost_mult=1.0)
    rows = {r["split"]: r for r, _ in H.run_cell(ctx, cell)}
    v = rows["VALIDATION_2024"]
    assert v["path_mae_ratio_to_naive"] == pytest.approx(1.0, abs=0.05)
    assert v["trades"] <= 0.1 * v["eligible_blocks"]
    ideal = {r["split"]: r for r, _ in H.run_cell(ctx, dict(cell, level=1.0))}["VALIDATION_2024"]
    assert ideal["path_mae"] < 0.1 * v["path_mae"]
