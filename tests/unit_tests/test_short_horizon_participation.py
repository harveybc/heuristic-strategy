"""Coordinator follow-up 2026-10-19: participation, RF-M range sweep, realistic sweeps.

Written before noise.flip_to_accuracy, the participation metrics and the val24x grid existed.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars
import tools.short_horizon_oracle as H


def test_flip_to_accuracy_is_exact_nested_and_local():
    rng = np.random.default_rng(1)
    ft = np.where(rng.uniform(size=400) < 0.5, B.LOW_FIRST, B.HIGH_FIRST)
    eligible = np.ones(400, bool)
    eligible[:50] = False  # e.g. ambiguous or outside the evaluated split
    u = rng.uniform(size=400)
    prev = None
    for target in (1.0, 0.9, 0.7, 0.6, 0.5):
        hat = N.flip_to_accuracy(ft, eligible, u, target)
        acc = np.mean(hat[eligible] == ft[eligible])
        assert acc == pytest.approx(target, abs=1 / eligible.sum())
        np.testing.assert_array_equal(hat[~eligible], ft[~eligible])
        flipped = hat != ft
        if prev is not None:
            assert (flipped | ~prev).all()  # lower accuracy flips a superset
        prev = flipped
    np.testing.assert_array_equal(N.flip_to_accuracy(ft, eligible, u, 0.7), N.flip_to_accuracy(ft, eligible, u, 0.7))


def test_participation_metrics_count_unfilled_limits():
    bars = synthetic_bars(seed=4)
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, 24)
    split = (bars.index[0], bars.index[-1] + HOUR)
    ok = np.flatnonzero(bl.in_split(split))
    orders = [None] * len(bl)
    k = ok[0]
    lo, hi = bl.bar_lo[k], bl.bar_hi[k]
    far = float(bars.LOW.iloc[lo:hi].min()) - 0.05  # never reached
    orders[k] = X.BlockOrder(1, "LIMIT", far, far + 0.2, far - 0.01, 10_000)
    res = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split)
    m = res.metrics
    assert m["orders_submitted"] == 1 and m["trades"] == 0 and m["unfilled_orders"] == 1
    assert m["fill_rate"] == 0.0 and m["eligible_blocks"] == len(ok)
    assert m["net_per_eligible_block_usd"] == 0.0
    orders[k] = X.BlockOrder(1, "MARKET", None, 5.0, 0.01, 10_000)
    m = X.run_backtest(bars, bl, orders, X.Costs(), cal, 10_000.0, split).metrics
    assert m["orders_submitted"] == 1 and m["trades"] == 1 and m["unfilled_orders"] == 0
    assert m["fill_rate"] == pytest.approx(1 / len(ok))
    assert m["net_per_eligible_block_usd"] == pytest.approx(m["net_profit_usd"] / len(ok))


def test_val24x_grid_contents():
    cells = H.cells("val24x")
    model = [c for c in cells if c["sweep"] != "baseline"]
    assert {c["block_hours"] for c in cells} == {24}
    assert {c["strategy"] for c in model} == {"RF-L", "RF-M"}
    assert len({c["seed"] for c in model}) >= 5
    assert {c["cost_mult"] for c in cells} == {1.0, 2.0}
    sweeps = {c["sweep"] for c in model}
    assert {"rfm_range", "rfm_firsttouch", "ft_at_naive_range", "ft_at_08_range",
            "range_abs_at_ft", "range_blend_at_ft"} <= sweeps
    rfm = [c for c in model if c["sweep"] == "rfm_range"]
    assert {c["strategy"] for c in rfm} == {"RF-M"} and {c["ft_target"] for c in rfm} == {1.0}
    naive = [c for c in model if c["sweep"] == "ft_at_naive_range"]
    assert {(c["family"], c["level"]) for c in naive} == {("blend", 0.0)}
    assert {c["ft_target"] for c in naive} == {round(1 - 0.05 * k, 2) for k in range(11)}
    assert {(c["family"], c["level"]) for c in model if c["sweep"] == "ft_at_08_range"} == {("blend", 0.2)}
    assert {c["ft_target"] for c in model if c["sweep"].startswith("range_")} == {0.6, 0.7}
    keys = [H.cell_key(c) + (c["sweep"], c["ft_target"]) for c in cells]
    assert len(keys) == len(set(keys))


def test_val24x_cell_hits_target_accuracy_on_validation_rows():
    bars = synthetic_bars(seed=22, segments=9, start="2024-01-07 17:00")
    ctx = H.Context(bars, calendar_for(bars), 24)
    cell = dict(grid="val24x", sweep="ft_at_naive_range", block_hours=24, strategy="RF-M", family="blend",
                level=0.0, p_flip=0.0, ft_target=0.6, seed=44, cost_mult=1.0)
    row = H.run_cell(ctx, cell)[0][0]
    n = row["eligible_blocks"] - row["ambiguous_blocks"]
    assert row["first_touch_accuracy"] == pytest.approx(0.6, abs=1 / n)
    assert row["range_mae_ratio_to_naive"] == pytest.approx(1.0)
    for key in ("fill_rate", "orders_submitted", "unfilled_orders", "net_per_eligible_block_usd"):
        assert key in row
