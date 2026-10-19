"""Calendar-range experiment (owner, 2026-10-19): causal volatility features, small GBM, adapter range metrics.

Written before app/short_horizon/vol_features.py, predictors.gbm_predict / choose_gbm_config and the
adapter's range-MAE ratios and owner sizing existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import adapter as AD
from app.short_horizon import blocks as B
from app.short_horizon import execution as X
from app.short_horizon import predictors as PR
from app.short_horizon import vol_features as VF
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars


def setup(seed=100, segments=24, hours=24):
    bars = synthetic_bars(seed=seed, segments=segments, start="2023-06-04 17:00")
    cal = calendar_for(bars)
    bl = B.build_blocks(bars, cal, hours)
    return bars, cal, bl, PR.block_table(bl, cal, hours)


def test_features_use_only_bars_before_the_block_start():
    bars, cal, bl, t = setup()
    F = VF.block_features(bars, bl, t, 24)
    assert len(F) == len(t) and np.isfinite(F.iloc[-5:].to_numpy()).all()
    i = len(t) // 2
    k = int(t.k.iloc[i])
    later = bars.copy()
    sel = later.index >= bl.start[k]
    later.loc[sel, ["HIGH", "CLOSE"]] += 0.01
    later.loc[sel, "LOW"] -= 0.01
    bl2 = B.build_blocks(later, cal, 24)
    t2 = PR.block_table(bl2, cal, 24)
    F2 = VF.block_features(later, bl2, t2, 24)
    np.testing.assert_allclose(F.iloc[: i + 1].to_numpy(), F2.iloc[: i + 1].to_numpy(), equal_nan=True)


def test_parkinson_and_efficiency_ratio_definitions():
    h = np.array([1.2, 1.1, 1.3]); l = np.array([1.0, 1.0, 1.1])
    assert VF.parkinson(h, l) == pytest.approx(np.sqrt(np.mean(np.log(h / l) ** 2) / (4 * np.log(2))))
    c = np.array([1.0, 1.1, 1.0, 1.2])
    assert VF.efficiency_ratio(c) == pytest.approx(0.2 / 0.4)


def test_gbm_is_causal_and_config_choice_ignores_rows_after_the_decision_year():
    bars, cal, bl, t = setup(segments=40)
    F = VF.block_features(bars, bl, t, 24)
    fit_end = t.start.iloc[len(t) // 2]
    dec_end = t.start.iloc[3 * len(t) // 4]
    grid = ({"max_depth": 2, "max_iter": 30}, {"max_depth": 3, "max_iter": 30})
    cfg, scores = PR.choose_gbm_config(t, F, fit_end, dec_end, grid, embargo=pd.Timedelta(hours=24))
    t2 = t.copy()
    t2.loc[t2.start >= dec_end, ["U", "D"]] *= 5          # perturb rows after the decision period
    cfg2, scores2 = PR.choose_gbm_config(t2, F, fit_end, dec_end, grid, embargo=pd.Timedelta(hours=24))
    assert cfg == cfg2 and scores == scores2
    u = PR.gbm_predict(t, F, "U", "weekly", cfg, min_rows=20)
    t3 = t.copy()
    j = len(t) // 2
    t3.loc[t3.index > j, "U"] *= 3
    u3 = PR.gbm_predict(t3, F, "U", "weekly", cfg, min_rows=20)
    np.testing.assert_array_equal(u[: j + 1], u3[: j + 1])


def test_adapter_reports_range_ratios_and_uses_owner_sizing(tmp_path):
    bars, cal, bl, t = setup(segments=10)
    m = AD.Market(bars, cal, bl, 24)
    split = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    ok = np.flatnonzero(bl.in_split(split))
    df = pd.DataFrame({"block_start": [bl.start[k].isoformat() for k in ok],
                       "made_at": [bl.start[k].isoformat() for k in ok],
                       "u_hat": 0.004, "d_hat": 0.004, "p_low_first": 0.5})
    f = tmp_path / "p.csv"
    df.to_csv(f, index=False)
    res = AD.score(AD.load_predictions(f, m, split), m, split, "RF-M", sizing=X.Sizing(leverage=10.0))
    for key in ("mae_r_ratio_mean20", "mae_r_ratio_persist"):
        assert key in res and np.isfinite(res[key])
    assert res["leverage"] == 10.0 and res["rel_volume"] == 0.10
