"""Owner follow-up 2026-10-19: business-day RF-L degradation on VALIDATION 2024.

Written before app/short_horizon/analysis.py and the ``val24`` grid existed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import analysis as A
from app.short_horizon import blocks as B
from tests.unit_tests.test_short_horizon import calendar_for, synthetic_bars
import tools.short_horizon_oracle as H


def test_zero_crossing_interpolates_first_sign_change():
    x = [0, 1, 2, 3]
    assert A.zero_crossing(x, [30, 10, -10, -20]) == pytest.approx(1.5)
    assert A.zero_crossing(x, [30, 10, 0, -20]) == pytest.approx(2.0)
    assert A.zero_crossing(x, [5, 4, 3, 2]) is None
    # descending x (accuracy 1.0 -> 0.5) works the same way
    assert A.zero_crossing([1.0, 0.9, 0.8], [20, 5, -5]) == pytest.approx(0.85)
    with pytest.raises(ValueError):
        A.zero_crossing([0, 1], [1.0])


def test_calibrated_brier():
    truth = np.array([B.LOW_FIRST, B.HIGH_FIRST, B.LOW_FIRST, B.AMBIGUOUS])
    assert A.calibrated_brier(truth.copy(), truth, 1.0) == 0.0
    wrong = np.array([B.HIGH_FIRST, B.LOW_FIRST, B.HIGH_FIRST, B.LOW_FIRST])
    assert A.calibrated_brier(wrong, truth, 1.0) == 1.0  # ambiguous truth excluded
    assert A.calibrated_brier(wrong, truth, 0.5) == pytest.approx(0.25)
    # calibrated forecaster with flip rate p has expected Brier p(1-p)
    rng = np.random.default_rng(0)
    t = np.where(rng.uniform(size=20000) < 0.5, B.LOW_FIRST, B.HIGH_FIRST)
    p = 0.2
    hat = np.where(rng.uniform(size=t.size) < p, -t, t)
    assert A.calibrated_brier(hat, t, 1 - p) == pytest.approx(p * (1 - p), abs=0.01)


def test_val24_grid_is_business_day_rf_l_validation_only_with_five_seeds():
    cells = H.cells("val24")
    model = [c for c in cells if c["strategy"] == "RF-L"]
    assert {c["block_hours"] for c in cells} == {24}
    assert {c["strategy"] for c in cells} == {"RF-L", "NO_TRADE", "NAIVE_MEAN20_RF-L"}
    assert len({c["seed"] for c in model}) >= 5
    assert {c["sweep"] for c in model} == {"range", "firsttouch", "joint"}
    rng = [c for c in model if c["sweep"] == "range"]
    assert {c["p_flip"] for c in rng} == {0.0} and max(c["level"] for c in rng) >= 40
    ft = [c for c in model if c["sweep"] == "firsttouch"]
    assert {c["level"] for c in ft} == {0.0}
    assert sorted({round(c["p_flip"], 2) for c in ft}) == [round(0.05 * k, 2) for k in range(11)]
    assert {c["cost_mult"] for c in cells} == {1.0, 2.0}
    keys = [H.cell_key(c) + (c["sweep"],) for c in cells]
    assert len(keys) == len(set(keys))


def test_val24_cells_report_only_validation_rows_with_achieved_metrics():
    bars = synthetic_bars(seed=21, segments=7, start="2024-01-07 17:00")
    cal = calendar_for(bars)
    ctx = H.Context(bars, cal, 24)
    cell = dict(grid="val24", sweep="joint", block_hours=24, strategy="RF-L", family="abs",
                level=5.0, p_flip=0.2, seed=45, cost_mult=1.0)
    rows = H.run_cell(ctx, cell)
    assert [r["split"] for r, _ in rows] == ["VALIDATION_2024"]
    row = rows[0][0]
    for key in ("mae_high_pips", "mae_low_pips", "range_mae_ratio_to_naive", "naive_mean20_mae_pips",
                "first_touch_accuracy", "brier_calibrated", "net_profit_usd", "end_equity_usd",
                "max_drawdown_usd", "sharpe_daily_sqrt365", "profit_factor", "trades", "win_rate"):
        assert key in row
    assert row["mae_high_pips"] > 0 and math.isfinite(row["range_mae_ratio_to_naive"])
    ideal = H.run_cell(ctx, dict(cell, level=0.0, p_flip=0.0))[0][0]
    assert ideal["mae_high_pips"] == 0 and ideal["first_touch_accuracy"] == 1.0 and ideal["brier_calibrated"] == 0


def test_original_seeds_draws_unchanged_by_extra_seeds():
    from app.short_horizon import noise as N
    bars = synthetic_bars(seed=2)
    ctx = H.Context(bars, calendar_for(bars), 6)
    ref = N.draws(42, len(ctx.blocks), len(ctx.horizons))
    np.testing.assert_array_equal(ctx.draws[42]["z_high"], ref["z_high"])
    assert set(H.VAL_SEEDS) <= set(ctx.draws)
