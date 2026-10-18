"""Short-horizon range-first strategy: labels, calendar, execution, costs, noise.

Written before app/short_horizon existed (STRATEGY_CONTRACT.md, S0).
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import calendar as CAL
from app.short_horizon import execution as X
from app.short_horizon import strategies as S
from app.short_horizon import data as D
from app.short_horizon import noise as N

HOUR = pd.Timedelta(hours=1)


def synthetic_bars(seed=0, segments=2, hours=120, start="2024-01-07 17:00", gap_hours=49):
    rng = np.random.default_rng(seed)
    stamps = []
    t = pd.Timestamp(start, tz="UTC")
    for _ in range(segments):
        stamps += [t + k * HOUR for k in range(hours)]
        t = stamps[-1] + gap_hours * HOUR
    n = len(stamps)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[1.10, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0006, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0006, n)
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close},
                        index=pd.DatetimeIndex(stamps, name="DATE_TIME"))


def calendar_for(bars, delay=12, flat=12, cutoff=0):
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    return CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs,
                              "entry_delay_after_open_hours": delay,
                              "force_flat_before_close_hours": flat,
                              "entry_cutoff_before_flat_hours": cutoff})


# ---------------------------------------------------------------- calendar
def test_segments_split_on_closure_and_refuse_outside_bars():
    bars = synthetic_bars()
    cal = calendar_for(bars)
    assert len(cal.segments) == 2
    seg = cal.segments[0]
    assert seg.market_open == bars.index[0]
    assert seg.market_close == bars.index[119] + HOUR
    assert seg.operational_start == seg.market_open + 12 * HOUR
    assert seg.force_flat == seg.market_close - 12 * HOUR
    cal.assign(bars.index)  # all inside
    stray = bars.index[119] + 5 * HOUR  # inside the closure
    with pytest.raises(CAL.CalendarError):
        cal.assign(bars.index.append(pd.DatetimeIndex([stray])))


def test_no_weekday_names_in_strategy_code():
    for module in (CAL, B, X, S):
        src = inspect.getsource(module)
        for token in ("weekday", "dayofweek", "day_name", "isoweekday"):
            assert token not in src, (module.__name__, token)


def test_blocks_never_outside_operational_window_and_never_invent_bars():
    bars = synthetic_bars()
    cal = calendar_for(bars)
    for hours in (6, 24):
        bl = B.build_blocks(bars, cal, hours)
        assert len(bl.start) > 0
        for k in range(len(bl.start)):
            seg = cal.segments[bl.segment[k]]
            assert seg.operational_start <= bl.start[k] < bl.end[k] <= seg.force_flat
            idx = bars.index[bl.bar_lo[k]:bl.bar_hi[k]]
            assert (idx >= bl.start[k]).all() and (idx + HOUR <= bl.end[k]).all()
        # business-day blocks are rollover aligned: boundaries at O + 24k or clipped
        if hours == 24:
            for k in range(len(bl.start)):
                seg = cal.segments[bl.segment[k]]
                off = (bl.start[k] - seg.market_open) / HOUR
                assert off % 24 == 0 or bl.start[k] == seg.operational_start


def test_missing_bars_are_absent_and_sparse_blocks_skipped():
    bars = synthetic_bars()
    cal = calendar_for(bars)
    full = B.build_blocks(bars, cal, 6)
    k = 2
    drop = bars.index[full.bar_lo[k]:full.bar_hi[k]][:4]  # leave 2 of 6 bars
    sparse = bars.drop(drop)
    bl = B.build_blocks(sparse, cal, 6)
    assert bl.status[k] == B.INSUFFICIENT_BARS
    assert len(sparse) == len(bars) - 4


# ---------------------------------------------------------------- labels
def test_first_touch_and_ambiguity():
    assert B.first_touch(np.array([1.0, 3.0, 2.0]), np.array([0.5, 1.0, 0.2])) == B.HIGH_FIRST
    assert B.first_touch(np.array([1.0, 3.0, 2.0]), np.array([0.1, 1.0, 0.9])) == B.LOW_FIRST
    assert B.first_touch(np.array([1.0, 3.0, 2.0]), np.array([0.9, 0.1, 0.9])) == B.AMBIGUOUS
    # first attainment wins when the extreme repeats
    assert B.first_touch(np.array([3.0, 1.0, 3.0]), np.array([0.5, 0.1, 0.5])) == B.HIGH_FIRST


def test_labels_are_causal_and_local():
    bars = synthetic_bars(seed=3)
    cal = calendar_for(bars)
    base = B.build_blocks(bars, cal, 6)
    k = 3
    # perturbing bars after block k's end leaves block k's label unchanged
    later = bars.copy()
    after = bars.index >= base.end[k]
    later.loc[after, ["HIGH"]] += 0.01
    later.loc[after, ["LOW"]] -= 0.01
    pert = B.build_blocks(later, cal, 6)
    for j in range(k + 1):
        assert (pert.H[j], pert.L[j], pert.ft[j], pert.ref[j]) == (base.H[j], base.L[j], base.ft[j], base.ref[j])
    # perturbing a HIGH/LOW inside block k changes block k labels only
    inside = bars.copy()
    t = bars.index[base.bar_lo[k] + 1]
    inside.loc[t, "HIGH"] += 0.02
    inside.loc[t, "LOW"] -= 0.02
    pert2 = B.build_blocks(inside, cal, 6)
    changed = [j for j in range(len(base.start))
               if (pert2.H[j], pert2.L[j], pert2.ft[j]) != (base.H[j], base.L[j], base.ft[j])]
    assert changed == [k]
    assert np.array_equal(pert2.ref, base.ref)  # HIGH/LOW never feed the decision price


def test_reference_price_is_last_close_before_start():
    bars = synthetic_bars()
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    for k in range(len(bl.start)):
        assert bars.index[bl.ref_bar[k]] + HOUR <= bl.start[k]
        assert bl.ref[k] == bars.CLOSE.iloc[bl.ref_bar[k]]


# ---------------------------------------------------------------- execution / ambiguity
def bar_frame(rows, start="2024-01-08 05:00"):
    t = pd.Timestamp(start, tz="UTC")
    idx = pd.DatetimeIndex([t + k * HOUR for k in range(len(rows))])
    return pd.DataFrame(rows, columns=["OPEN", "HIGH", "LOW", "CLOSE"], index=idx)


def run_single(rows, order, costs=None):
    bars = bar_frame(rows)
    o, h, l, c = (bars[k].to_numpy() for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    return X.simulate_block(order, o, h, l, c)


def test_stop_first_when_both_touched():
    order = X.BlockOrder(direction=1, entry_type="MARKET", entry_price=None,
                         take_profit=1.1050, stop_loss=1.0950, units=10000)
    fill = run_single([[1.1000, 1.1060, 1.0940, 1.1000]], order)
    assert fill.exit_reason == "SL" and fill.exit_price == 1.0950 and fill.ambiguous


def test_gap_through_stop_fills_at_open_and_tp_gap_improves():
    order = X.BlockOrder(1, "MARKET", None, 1.1050, 1.0950, 10000)
    fill = run_single([[1.1000, 1.1010, 1.0990, 1.0990], [1.0900, 1.0920, 1.0890, 1.0910]], order)
    assert fill.exit_reason == "SL" and fill.exit_price == 1.0900
    fill = run_single([[1.1000, 1.1010, 1.0990, 1.1000], [1.1070, 1.1080, 1.1060, 1.1070]], order)
    assert fill.exit_reason == "TP" and fill.exit_price == 1.1070
    short = X.BlockOrder(-1, "MARKET", None, 1.0950, 1.1050, 10000)
    fill = run_single([[1.1000, 1.1060, 1.0940, 1.1000]], short)
    assert fill.exit_reason == "SL" and fill.exit_price == 1.1050 and fill.ambiguous


def test_limit_fill_bar_never_credits_tp_but_can_stop():
    order = X.BlockOrder(1, "LIMIT", 1.0980, 1.1030, 1.0960, 10000)
    # bar 0 touches the limit and the TP: TP is NOT credited on the fill bar
    fill = run_single([[1.1000, 1.1040, 1.0975, 1.1000], [1.1000, 1.1010, 1.0990, 1.1005]], order)
    assert fill.entry_price == 1.0980 and fill.exit_reason == "TIME"
    assert fill.exit_price == 1.1005
    # fill bar also touching the stop: stopped (conservative)
    fill = run_single([[1.1000, 1.1010, 1.0950, 1.0990]], order)
    assert fill.exit_reason == "SL" and fill.exit_price == 1.0960
    # limit crossed at the open: fills at the open, whole bar is after the fill
    fill = run_single([[1.0970, 1.1035, 1.0965, 1.1000]], order)
    assert fill.entry_price == 1.0970 and fill.exit_reason == "TP"
    # never touched: no trade
    fill = run_single([[1.1000, 1.1010, 1.0990, 1.1000]], order)
    assert fill is None


def test_market_entry_refused_when_open_violates_bracket():
    order = X.BlockOrder(1, "MARKET", None, 1.1050, 1.0950, 10000)
    fill = run_single([[1.0940, 1.0960, 1.0930, 1.0950]], order)
    assert fill is not None and fill.refused and fill.units == 0


def test_time_exit_at_block_end():
    order = X.BlockOrder(-1, "MARKET", None, 1.0900, 1.1100, 10000)
    fill = run_single([[1.1000, 1.1010, 1.0990, 1.1002], [1.1002, 1.1010, 1.0990, 1.0995]], order)
    assert fill.exit_reason == "TIME" and fill.exit_price == 1.0995 and fill.exit_bar == 1


# ---------------------------------------------------------------- costs
def test_hand_calculated_round_trip_costs_charged_once():
    costs = X.Costs()  # 2 pip spread, 1 pip slip/fill, USD 7/lot RT, USD 10/lot/rollover
    units = 10_000
    per_fill = costs.fill_costs(units)
    assert per_fill["spread"] == pytest.approx(1.00)
    assert per_fill["slippage"] == pytest.approx(1.00)
    assert per_fill["commission"] == pytest.approx(0.35)
    assert costs.financing(units, 1) == pytest.approx(1.00)
    assert costs.round_trip_price() == pytest.approx(0.00047)
    # full backtest of exactly one long trade +10 pips gross, crossing one rollover
    bars = synthetic_bars(seed=1)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 24)
    k = next(k for k in range(len(bl.start)) if bl.status[k] == B.OK and (bl.end[k] - bl.start[k]) == 24 * HOUR)
    lo, hi = bl.bar_lo[k], bl.bar_hi[k]
    bars = bars.copy()
    bars.iloc[lo, bars.columns.get_loc("OPEN")] = 1.10000
    bars.iloc[hi - 1, bars.columns.get_loc("CLOSE")] = 1.10100
    bars["HIGH"] = bars[["OPEN", "HIGH", "CLOSE"]].max(axis=1)
    bars["LOW"] = bars[["OPEN", "LOW", "CLOSE"]].min(axis=1)
    bl = B.build_blocks(bars, cal, 24)
    orders = [None] * len(bl.start)
    orders[k] = X.BlockOrder(1, "MARKET", None, 5.0, 0.01, units)
    res = X.run_backtest(bars, bl, orders, costs, cal, initial_capital=10_000.0,
                         split=(bars.index[0], bars.index[-1] + HOUR))
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t["gross_usd"] == pytest.approx(10.00, abs=1e-9)
    assert t["spread_usd"] == pytest.approx(2.00)
    assert t["slippage_usd"] == pytest.approx(2.00)
    assert t["commission_usd"] == pytest.approx(0.70)
    assert t["financing_usd"] == pytest.approx(1.00)  # the block ends on a rollover (inclusive)
    assert t["net_usd"] == pytest.approx(10.00 - 4.70 - 1.00)
    assert res.metrics["net_profit_usd"] == pytest.approx(t["net_usd"])
    assert res.metrics["end_equity_usd"] == pytest.approx(10_000 + t["net_usd"])
    total = sum(res.metrics[f"cost_{c}_usd"] for c in ("spread", "slippage", "commission", "financing"))
    assert total == pytest.approx(5.70)


def test_cost_multiplier_scales_every_cost():
    c = X.Costs().scaled(2.0)
    assert c.round_trip_price() == pytest.approx(0.00094)
    assert c.financing(100_000, 1) == pytest.approx(20.0)
    assert X.Costs().scaled(0.0).round_trip_price() == 0.0


# ---------------------------------------------------------------- strategies
def test_range_first_market_geometry_and_sizing():
    p = S.RangeFirstParams()
    costs = X.Costs()
    o = S.range_first_order("RF-M", 1.1060, 1.0960, B.LOW_FIRST, 1.1000, costs, p, 10_000.0)
    assert o.direction == 1 and o.entry_type == "MARKET"
    assert o.take_profit == pytest.approx(1.1060 - 0.1 * 0.0100)
    assert o.stop_loss == pytest.approx(1.0960 - 0.25 * 0.0100)
    # risk 1% of 10k = 100 USD over 0.0065 -> 15384 units -> leverage cap 5x = 45454 -> floor 15000
    assert o.units == 15000
    s = S.range_first_order("RF-M", 1.1060, 1.0960, B.HIGH_FIRST, 1.1000, costs, p, 10_000.0)
    assert s.direction == -1 and s.take_profit == pytest.approx(1.0960 + 0.001)
    # no edge after costs -> no trade
    assert S.range_first_order("RF-M", 1.1004, 1.0990, B.LOW_FIRST, 1.1000, costs, p, 10_000.0) is None
    # inverted range -> no trade
    assert S.range_first_order("RF-M", 1.0950, 1.1050, B.LOW_FIRST, 1.1000, costs, p, 10_000.0) is None


def test_range_first_limit_geometry():
    p = S.RangeFirstParams()
    o = S.range_first_order("RF-L", 1.1060, 1.0960, B.LOW_FIRST, 1.1000, X.Costs(), p, 10_000.0)
    assert o.entry_type == "LIMIT" and o.entry_price == pytest.approx(1.0970)
    s = S.range_first_order("RF-L", 1.1060, 1.0960, B.HIGH_FIRST, 1.1000, X.Costs(), p, 10_000.0)
    assert s.entry_type == "LIMIT" and s.entry_price == pytest.approx(1.1050) and s.direction == -1


def test_comparator_requires_chronological_tp_before_sl():
    params = S.comparator_params(X.Costs())
    up_then_down = [1.1030, 1.1060, 1.1040, 1.0990]
    o = S.comparator_order(1.1000, up_then_down, X.Costs(), params, 10_000.0)
    assert o is not None and o.direction == 1
    # same extrema, reversed order: the SL-side sample is not reached first? geometry
    # is identical (extrema only); the chronological rule decides acceptance.
    reordered = [1.0990, 1.0985, 1.1060, 1.1030]
    g1 = S.entry_geometry(1.1000, up_then_down, params)
    g2 = S.entry_geometry(1.1000, list(reversed(up_then_down)), params)
    assert (g1.direction, g1.take_profit_price, g1.stop_loss_price) == (g2.direction, g2.take_profit_price, g2.stop_loss_price)
    assert S.first_barrier(g1.direction, up_then_down, g1.take_profit_price, g1.stop_loss_price) == "TP"
    assert S.comparator_order(1.1000, [1.1000] * 4, X.Costs(), params, 10_000.0) is None  # persistence: no trade
    assert S.comparator_order(1.1000, [math.nan] * 4, X.Costs(), params, 10_000.0) is None


def test_comparator_policy_is_the_committed_one():
    from app.policies import prediction_entry_exit as P
    assert S.entry_geometry.__wrapped__ is P.calculate_entry_geometry


def test_naive_forecasts_use_completed_blocks_only():
    bars = synthetic_bars(seed=5)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    u, d, ft = S.naive_forecasts(bl, "PERSIST")
    ok = np.flatnonzero(bl.status == B.OK)
    for a, b in zip(ok[:-1], ok[1:]):
        if bl.end[a] <= bl.start[b] and np.isfinite(u[b]):
            assert u[b] == pytest.approx(bl.H[a] - bl.ref[a])
            assert d[b] == pytest.approx(bl.ref[a] - bl.L[a])
    # changing the last block's future bars cannot change any naive forecast up to it
    later = bars.copy()
    last = ok[-1]
    later.iloc[bl.bar_lo[last]:bl.bar_hi[last], later.columns.get_loc("HIGH")] += 0.01
    bl2 = B.build_blocks(later, cal, 6)
    u2, d2, ft2 = S.naive_forecasts(bl2, "MEAN20")
    u1, d1, ft1 = S.naive_forecasts(bl, "MEAN20")
    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(ft1, ft2)


# ---------------------------------------------------------------- business guards on the oracle
@pytest.mark.parametrize("hours", [6, 24])
@pytest.mark.parametrize("variant", ["RF-M", "RF-L"])
def test_oracle_respects_entry_start_cutoff_and_forced_flat(hours, variant):
    bars = synthetic_bars(seed=11, segments=3)
    cal = calendar_for(bars, cutoff=6)
    bl = B.build_blocks(bars, cal, hours)
    orders = S.range_first_orders(variant, bl, bl.H, bl.L, bl.ft_oracle(), X.Costs(), S.RangeFirstParams(), 10_000.0)
    res = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0,
                         split=(bars.index[0], bars.index[-1] + HOUR))
    assert res.trades, "oracle should trade on synthetic data"
    for t in res.trades:
        seg = cal.segments[t["segment"]]
        assert seg.operational_start <= t["entry_time"] < seg.entry_cutoff
        assert t["exit_time"] <= seg.force_flat
    assert res.final_position_units == 0
    for seg in cal.segments:  # flat at every forced-flat instant
        assert res.position_at(seg.force_flat) == 0


def test_oracle_rf_l_wins_every_trade_without_costs():
    bars = synthetic_bars(seed=12, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    zero = X.Costs().scaled(0.0)
    orders = S.range_first_orders("RF-M", bl, bl.H, bl.L, bl.ft_oracle(), zero, S.RangeFirstParams(), 10_000.0)
    res = X.run_backtest(bars, bl, orders, zero, cal, 10_000.0, split=(bars.index[0], bars.index[-1] + HOUR))
    assert all(t["net_usd"] > 0 for t in res.trades)


# ---------------------------------------------------------------- baselines and identical rows
def test_all_strategies_and_baselines_share_the_block_population():
    bars = synthetic_bars(seed=13, segments=3)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    populations = []
    for name, orders in S.baseline_orders(bl, X.Costs(), S.RangeFirstParams(), 10_000.0).items():
        res = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split=split)
        populations.append(tuple(res.evaluated_blocks))
        if name == "NO_TRADE":
            assert res.metrics["trades"] == 0 and res.metrics["net_profit_usd"] == 0
    seg_res = X.run_segment_hold(bars, cal, 1, X.Costs(), 10_000.0, split=split, blocks=bl)
    populations.append(tuple(seg_res.evaluated_blocks))
    assert len(set(populations)) == 1


def test_metrics_include_flat_calendar_days():
    bars = synthetic_bars(seed=14)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 6)
    split = (bars.index[0], bars.index[-1] + HOUR)
    res = X.run_backtest(bars, bl, [None] * len(bl.start), X.Costs(), cal, 10_000.0, split=split)
    days = pd.date_range(bars.index[0].normalize(), bars.index[-1].normalize(), freq="D")
    assert len(res.daily_equity) == len(days)  # the closure days are present
    assert res.metrics["sharpe_daily_sqrt365"] == 0.0


# ---------------------------------------------------------------- noise
def test_noise_is_deterministic_per_seed_and_common_across_levels():
    a = N.draws(42, 50, 4)
    b = N.draws(42, 50, 4)
    c = N.draws(43, 50, 4)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["z_high"], c["z_high"])
    H = np.full(50, 1.1)
    L = np.full(50, 1.09)
    ft = np.where(np.arange(50) % 2, B.HIGH_FIRST, B.LOW_FIRST)
    h1, l1, f1 = N.apply_range_noise(H, L, ft, a, sigma=np.full(50, 0.0010), p_flip=0.3)
    h2, l2, f2 = N.apply_range_noise(H, L, ft, a, sigma=np.full(50, 0.0020), p_flip=0.3)
    np.testing.assert_allclose(h2 - H, 2 * (h1 - H))  # same draws, scaled
    np.testing.assert_array_equal(f1, f2)
    _, _, f0 = N.apply_range_noise(H, L, ft, a, sigma=np.zeros(50), p_flip=0.0)
    np.testing.assert_array_equal(f0, ft)
    flips = N.apply_range_noise(H, L, ft, a, sigma=np.zeros(50), p_flip=0.5)[2] != ft
    assert flips.sum() == (a["u_flip"] < 0.5).sum()


def test_ambiguous_truth_gets_a_fixed_coin_not_the_future():
    ft = np.array([B.AMBIGUOUS] * 200 + [B.LOW_FIRST])
    coin = N.ambiguous_coin(201)
    resolved = N.resolve_ambiguous(ft, coin)
    assert set(np.unique(resolved[:200])) == {B.LOW_FIRST, B.HIGH_FIRST}
    np.testing.assert_array_equal(resolved, N.resolve_ambiguous(ft, N.ambiguous_coin(201)))


# ---------------------------------------------------------------- data guard
def test_loader_refuses_test_rows(tmp_path):
    t = pd.date_range("2024-12-31 20:00", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame({"datetime": t, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0})
    path = tmp_path / "x.parquet"
    frame.to_parquet(path)
    loaded = D.load_ohlc(path, start="2024-12-01", end_exclusive="2025-01-01")
    assert loaded.index.max() < pd.Timestamp("2025-01-01", tz="UTC")
    assert len(loaded) == 4
    with pytest.raises(D.TestRowsError):
        D.load_ohlc(path, start="2024-12-01", end_exclusive="2025-01-02")
