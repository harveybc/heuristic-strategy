"""Owner redesign 2026-10-19: the strategy consumes ONLY a predicted hourly price path.

Written before strategies.path_to_range, noise.path_draws, analysis.path_mae and the
``pathx`` grid existed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.short_horizon import analysis as A
from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import noise as N
from app.short_horizon import strategies as S
from tests.unit_tests.test_short_horizon import calendar_for, synthetic_bars
import tools.short_horizon_oracle as H


def test_path_to_range_takes_max_min_and_their_order():
    paths = np.array([[1.10, 1.12, 1.09, 1.11],      # max at 1, min at 2 -> HIGH_FIRST
                      [1.10, 1.08, 1.13, np.nan],    # min at 1, max at 2 -> LOW_FIRST; NaN ignored
                      [1.10, 1.10, 1.10, 1.10],      # flat -> zero range, ambiguous
                      [np.nan] * 4])                 # no forecast
    Hh, Lh, ft = S.path_to_range(paths)
    np.testing.assert_allclose(Hh[:3], [1.12, 1.13, 1.10])
    np.testing.assert_allclose(Lh[:3], [1.09, 1.08, 1.10])
    assert list(ft) == [B.HIGH_FIRST, B.LOW_FIRST, B.AMBIGUOUS, B.AMBIGUOUS]
    assert math.isnan(Hh[3]) and math.isnan(Lh[3])
    # first occurrence wins on ties
    Hh, Lh, ft = S.path_to_range(np.array([[1.0, 2.0, 0.5, 2.0, 0.5]]))
    assert ft[0] == B.HIGH_FIRST


def test_flat_or_missing_path_never_trades():
    Hh, Lh, ft = S.path_to_range(np.array([[1.1, 1.1, 1.1], [np.nan] * 3]))
    for k in range(2):
        assert S.range_first_order("RF-M", Hh[k], Lh[k], int(ft[k]), 1.1, X.Costs(), S.RangeFirstParams(), 1e4) is None


def test_path_mae_overall_per_horizon_and_naive():
    target = np.array([[1.0, 2.0, np.nan], [3.0, 5.0, 4.0]])
    pred = np.array([[1.5, 2.0, np.nan], [3.0, 4.0, 6.0]])
    ref = np.array([1.0, 3.0])
    out = A.path_mae(pred, target, np.array([True, True]))
    assert out["mae"] == pytest.approx((0.5 + 0 + 0 + 1 + 2) / 5)
    np.testing.assert_allclose(out["per_h"], [0.25, 0.5, 2.0])
    naive = A.path_mae(np.repeat(ref[:, None], 3, axis=1), target, np.array([True, True]))
    assert naive["mae"] == pytest.approx((0 + 1 + 0 + 2 + 1) / 5)
    only_second = A.path_mae(pred, target, np.array([False, True]))
    assert only_second["mae"] == pytest.approx(1.0)


def test_path_draws_deterministic_and_seed_specific():
    a, b, c = N.path_draws(42, 30, 24), N.path_draws(42, 30, 24), N.path_draws(43, 30, 24)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (30, 24) and not np.array_equal(a, c)


def test_pathx_grid():
    cells = H.cells("pathx")
    model = [c for c in cells if c["strategy"] in ("RF-L", "RF-M")]
    assert {c["block_hours"] for c in cells} == {6, 24}
    assert {c["strategy"] for c in cells} == {"RF-L", "RF-M", "PERSISTENCE"}
    assert len({c["seed"] for c in model}) >= 5
    assert {c["cost_mult"] for c in cells} == {1.0, 2.0}
    sig = sorted({c["level"] for c in model})
    assert sig[0] == 0.0 and sig[-1] >= 0.01  # well past the naive level (USD/EUR)
    keys = [H.cell_key(c) for c in cells]
    assert len(keys) == len(set(keys))


def test_pathx_cell_reports_price_mae_and_both_splits():
    bars = synthetic_bars(seed=31, segments=8, start="2023-12-03 17:00")
    ctx = H.Context(bars, calendar_for(bars), 24)
    cell = dict(grid="pathx", block_hours=24, strategy="RF-M", family="path", level=0.0, p_flip=0.0,
                seed=42, cost_mult=1.0)
    rows = {r["split"]: r for r, _ in H.run_cell(ctx, cell)}
    assert set(rows) == {"TRAIN_2020_2023", "VALIDATION_2024"}
    v = rows["VALIDATION_2024"]
    assert v["path_mae"] == 0.0 and v["path_mae_ratio_to_naive"] == 0.0
    for key in ("naive_path_mae", "naive_mae_h12", "naive_mae_h24", "path_mae_h1", "path_mae_h24",
                "net_profit_usd", "sharpe_daily_sqrt365", "trades", "fill_rate"):
        assert key in v
    noisy = {r["split"]: r for r, _ in H.run_cell(ctx, dict(cell, level=0.002))}["VALIDATION_2024"]
    assert noisy["path_mae"] == pytest.approx(0.002 * math.sqrt(2 / math.pi), rel=0.25)
    persist = {r["split"]: r for r, _ in H.run_cell(ctx, dict(cell, strategy="PERSISTENCE", seed=0))}
    assert persist["VALIDATION_2024"]["trades"] == 0
    assert persist["VALIDATION_2024"]["path_mae_ratio_to_naive"] == pytest.approx(1.0)
