"""A06: the pinned strategy adapter's execution contract as the timing evaluator uses it (Ctx.run path), on a
synthetic one-segment calendar.  These tests pin observed behaviour of the EXISTING adapter; they do not
reimplement it.  AC01-AC09."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

S = pytest.importorskip("tools.short_horizon_zigzag_study")
from app.short_horizon import calendar as CAL  # noqa: E402
from app.short_horizon import execution as X  # noqa: E402
from app.short_horizon.execution import BlockOrder, simulate_block  # noqa: E402

TE = pytest.importorskip("tools.short_horizon_timing_eval")


def _ctx(n=120, start="2023-01-08 17:00"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")           # Sunday 17:00 weekly open (bar clock)
    c = 1.10 + 0.0005 * np.sin(np.arange(n) / 3.0)
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"OPEN": o, "HIGH": np.maximum(o, c) + 2e-4, "LOW": np.minimum(o, c) - 2e-4, "CLOSE": c}, index=idx)
    cfg = {"calendar_id": "T", "version": "t", "entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12,
           "entry_cutoff_before_flat_hours": 0,
           "segments": [{"segment_id": "S", "market_open": str(idx[0]), "market_close": str(idx[-1] + pd.Timedelta(hours=1))}]}
    ctx = S.Ctx.__new__(S.Ctx)
    ctx.bars, ctx.cal, ctx.digest = bars, CAL.load_calendar(cfg), "synthetic"
    ctx.h, ctx.l = bars.HIGH.to_numpy(float), bars.LOW.to_numpy(float)
    ctx.sigma, ctx.sigma_rule = np.full(n, 5e-4), np.full(n, 2e-3)
    ctx.prof, ctx._piv = X.RealisticProfile(), {}
    S.PERIODS["SYNTH_ALL"] = (str(idx[0]), str(idx[-1] + pd.Timedelta(hours=2)))
    return ctx


def test_ac01_fill_is_the_open_two_bars_after_the_signal_bar_never_the_pivot_extreme():
    ctx = _ctx()
    _, res = ctx.run([(90, 1)], [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    t = res.trades[0]
    assert t["entry_bar_abs"] == 92 and t["entry_price"] == pytest.approx(ctx.bars.OPEN.iloc[92])
    assert t["entry_price"] != pytest.approx(ctx.bars.LOW.iloc[90])


def test_ac02_reopen_rule_is_not_applied_by_signal_blocks_and_the_evaluator_guard_drops_it():
    ctx = _ctx()
    seg = ctx.cal.segments[0]
    _, res = ctx.run([(5, 1)], [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    assert res.trades and pd.Timestamp(res.trades[0]["entry_time"]) < seg.operational_start   # documented adapter gap
    kept, dropped = TE.reopen_guard([(5, 1), (20, 1)], ctx)
    assert kept == [(20, 1)] and dropped == 1


def test_ac03_forced_flat_twelve_hours_before_close():
    ctx = _ctx()
    seg = ctx.cal.segments[0]
    _, res = ctx.run([(20, 1)], [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    t = res.trades[0]
    assert t["exit_reason"] == "TIME" and pd.Timestamp(t["exit_time"]) == seg.force_flat
    assert seg.force_flat == seg.market_close - pd.Timedelta(hours=12)
    assert TE.trade_breakdown(res, ctx)["closes_forced_flat"] == 1


def test_ac04_stop_before_take_profit_inside_one_bar_is_flagged_ambiguous():
    o, h, l, c = (np.array(v, float) for v in ([1.0, 1.0], [1.0, 1.02], [1.0, 0.98], [1.0, 1.0]))
    f = simulate_block(BlockOrder(1, "MARKET", None, 1.01, 0.99, 1, 1.0), o, h, l, c)
    assert f.exit_reason == "SL" and f.exit_price == pytest.approx(0.99) and f.ambiguous is True


def test_ac05_gap_through_the_stop_fills_at_the_open():
    o, h, l, c = (np.array(v, float) for v in ([1.0, 0.97], [1.0, 0.975], [1.0, 0.96], [1.0, 0.97]))
    f = simulate_block(BlockOrder(1, "MARKET", None, 1.01, 0.99, 1, 1.0), o, h, l, c)
    assert f.exit_reason == "SL" and f.exit_price == pytest.approx(0.97) and f.exit_at_open is True


def test_ac06_close_and_reverse_is_two_costed_fills_and_two_entries():
    ctx = _ctx()
    _, res = ctx.run([(20, 1), (22, -1)], [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    assert len(res.trades) == 2 and [t["direction"] for t in res.trades] == [1, -1]
    assert all(t["entry_cost_usd"] > 0 and t["exit_cost_usd"] > 0 for t in res.trades)
    br = TE.trade_breakdown(res, ctx)
    assert br["entries_filled"] == 2 and br["close_and_reverse"] == 1


def test_ac07_at_most_four_orders_per_business_day():
    ctx = _ctx()
    sig = [(b, 1 if i % 2 == 0 else -1) for i, b in enumerate(range(20, 40, 2))]   # 10 orders inside one rollover day
    m, _ = ctx.run(sig, [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    assert m["max_orders_per_day"] <= 4 and m["capped_orders"] >= 1


def test_ac08_units_are_integer_and_sized_from_the_balance():
    ctx = _ctx()
    _, res = ctx.run([(90, 1)], [], "SYNTH_ALL", mode="causal", k_tp=2.0)
    u = res.trades[0]["units"]
    assert float(u).is_integer() and u >= 1
    expect = math.floor(0.10 * S.CAPITAL * 100.0 / res.trades[0]["entry_price"])
    assert abs(u - expect) <= expect * 0.01          # margin basis rel_volume 0.10 at 1:100 (refused above free margin)


def test_ac09_the_frequency_rule_marks_validity_and_never_adds_an_entry():
    ctx = _ctx()
    P = np.tile([0.9, 0.05, 0.05], (120, 1))                          # nothing crosses any threshold
    m, res, sig = TE.strategy(ctx, [], P, np.arange(120), 0.4, 2, 2.0, "SYNTH_ALL")
    assert sig == [] and m["entries_filled"] == 0 and m["valid"] is False and m["sharpe_daily_sqrt365"] == 0.0


def test_ac10_executable_path_fills_at_the_next_open_and_refuses_before_operational_start():
    ZO = pytest.importorskip("app.short_horizon.zigzag_opt")
    ctx = _ctx()
    ZO.PERIODS["SYNTH_ALL"] = S.PERIODS["SYNTH_ALL"]
    zo = ZO.Evaluator(ctx.bars, ctx.cal)
    zo.sigma, zo.sigma_rule = ctx.sigma, ctx.sigma_rule                # the synthetic series is shorter than its rolling windows
    TE._EXEC.update(executable=True, cap=4, zo=zo)
    try:
        P = np.tile([0.05, 0.9, 0.05], (120, 1))
        origin = np.array([5, 90])                                     # 5: before operational start; 90: inside
        m, res, sig = TE.strategy(ctx, [], P[origin], origin, 0.5, 2, 2.0, "SYNTH_ALL")
        assert m["fill_path"] == "EXECUTABLE_B_PLUS_1" and m["refused_before_operational_start"] == 1
        t = res.trades[0]
        assert len(res.trades) == 1 and t["entry_bar_abs"] == 91 and t["entry_price"] == pytest.approx(ctx.bars.OPEN.iloc[91])
    finally:
        TE._EXEC.update(executable=False, cap=4, zo=None)
