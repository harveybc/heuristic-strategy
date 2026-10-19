"""Sealed-prediction adapter and cheap causal baseline predictors (owner, 2026-10-19).

Written before app/short_horizon/adapter.py and app/short_horizon/predictors.py existed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import adapter as AD
from app.short_horizon import blocks as B
from app.short_horizon import predictors as PR
from app.short_horizon import strategies as S
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars

SPLIT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))


def market(seed=50, segments=10, hours=24):
    bars = synthetic_bars(seed=seed, segments=segments, start="2024-01-07 17:00")
    cal = calendar_for(bars)
    return AD.Market(bars, cal, B.build_blocks(bars, cal, hours), hours)


def excursion_frame(m, k_rows=None):
    ok = np.flatnonzero(m.blocks.in_split(SPLIT))
    rows = ok if k_rows is None else ok[k_rows]
    return pd.DataFrame({"block_start": [m.blocks.start[k].isoformat() for k in rows],
                         "made_at": [m.blocks.start[k].isoformat() for k in rows],
                         "u_hat": 0.0010, "d_hat": 0.0008, "p_low_first": 0.6})


# ---------------------------------------------------------------- adapter: sealing and refusals
def test_seal_then_tamper_is_refused(tmp_path):
    m = market()
    f = tmp_path / "pred.csv"
    excursion_frame(m).to_csv(f, index=False)
    seal = AD.seal(f, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "SEAL.json").read_text())["sha256"] == seal["sha256"]
    AD.verify_seal(f, seal)
    f.write_text(f.read_text() + "\n")
    with pytest.raises(AD.PredictionError):
        AD.verify_seal(f, seal)


@pytest.mark.parametrize("mutation", ["outside_split", "duplicate", "future_made_at", "unknown_key", "nan"])
def test_adapter_refuses_bad_rows(tmp_path, mutation):
    m = market()
    df = excursion_frame(m)
    if mutation == "outside_split":
        df.loc[0, ["block_start", "made_at"]] = "2023-06-05T05:00:00+00:00"
    elif mutation == "duplicate":
        df = pd.concat([df, df.iloc[[0]]])
    elif mutation == "future_made_at":
        df.loc[1, "made_at"] = (pd.Timestamp(df.loc[1, "block_start"]) + HOUR).isoformat()
    elif mutation == "unknown_key":
        df.loc[2, ["block_start", "made_at"]] = (pd.Timestamp(df.loc[2, "block_start"]) + 3 * HOUR).isoformat()
    elif mutation == "nan":
        df.loc[3, "u_hat"] = np.nan
    f = tmp_path / "p.csv"
    df.to_csv(f, index=False)
    with pytest.raises(AD.PredictionError):
        AD.load_predictions(f, m, SPLIT)


def test_path_mode_persistence_scores_as_the_naive(tmp_path):
    m = market()
    ok = np.flatnonzero(m.blocks.in_split(SPLIT))
    df = pd.DataFrame({"block_start": [m.blocks.start[k].isoformat() for k in ok],
                       "made_at": [m.blocks.start[k].isoformat() for k in ok]})
    for h in range(1, m.hours + 1):
        df[f"p_h{h}"] = m.blocks.ref[ok]
    f = tmp_path / "persist.parquet"
    df.to_parquet(f)
    ps = AD.load_predictions(f, m, SPLIT)
    assert ps.mode == "path"
    res = AD.score(ps, m, SPLIT, "RF-M", cost_mult=1.0)
    assert res["path_mae_ratio_mean"] == pytest.approx(1.0)
    assert res["trades"] == 0 and res["valid"] is False


def test_excursion_mode_metrics_and_missing_rows_abstain(tmp_path):
    m = market()
    df = excursion_frame(m, k_rows=slice(0, 10))
    f = tmp_path / "p.csv"
    df.to_csv(f, index=False)
    ps = AD.load_predictions(f, m, SPLIT)
    assert ps.mode == "excursion" and int(np.isfinite(ps.u).sum()) == 10
    res = AD.score(ps, m, SPLIT, "RF-L", cost_mult=1.0)
    for key in ("mae_u_ratio_mean20", "mae_d_ratio_mean20", "direction_accuracy", "log_loss",
                "path_mae_ratio_hp", "path_mae_ratio_hs", "path_mae_ratio_mean", "net_profit_usd",
                "sharpe_daily_sqrt365", "trades_per_week", "valid", "coverage"):
        assert key in res
    assert res["coverage"] < 1.0


def test_reconstructed_path_round_trips_through_the_strategy():
    ref = np.array([1.10, 1.10])
    U, D = np.array([0.002, 0.001]), np.array([0.001, 0.003])
    ft = np.array([B.LOW_FIRST, B.HIGH_FIRST])
    P = AD.reconstruct_path(ref, U, D, ft, 24)
    H, L, f = S.path_to_range(P)
    np.testing.assert_allclose(H, ref + U)
    np.testing.assert_allclose(L, ref - D)
    np.testing.assert_array_equal(f, ft)


# ---------------------------------------------------------------- causal predictors
def table(seed=51, segments=30, hours=24):
    m = market(seed, segments, hours)
    return PR.block_table(m.blocks, m.calendar, hours)


@pytest.mark.parametrize("cadence", ["weekly", "daily"])
def test_har_and_logit_are_causal(cadence):
    t = table()
    i = len(t) // 2
    base_u = PR.har_predict(t, 24, "U", cadence, min_rows=20)
    base_p = PR.logit_predict(t, 24, cadence, min_rows=20)
    t2 = t.copy()
    t2.loc[t2.index > i, "U"] *= 3.0
    t2.loc[t2.index > i, "y"] = 1 - t2.loc[t2.index > i, "y"]
    u2 = PR.har_predict(t2, 24, "U", cadence, min_rows=20)
    p2 = PR.logit_predict(t2, 24, cadence, min_rows=20)
    np.testing.assert_array_equal(base_u[: i + 1], u2[: i + 1])
    np.testing.assert_array_equal(base_p[: i + 1], p2[: i + 1])
    assert np.isfinite(base_u[-5:]).all() and ((base_p[-5:] > 0) & (base_p[-5:] < 1)).all()


def test_refit_uses_only_rows_before_the_group():
    t = table()
    groups = PR.refit_groups(t, "weekly")
    for g_start, rows in groups:
        assert all(r >= g_start for r in rows)
    assert sorted(r for _, rows in groups for r in rows) == list(range(len(t)))


def test_ewma_halflife_selected_on_train_only():
    t = table()
    train = np.zeros(len(t), bool)
    train[: len(t) // 2] = True
    hl = PR.select_halflife(t, train, grid=(2, 5, 10, 20))
    t2 = t.copy()
    t2.loc[~train, ["U", "D"]] *= 10
    assert PR.select_halflife(t2, train, grid=(2, 5, 10, 20)) == hl
    e = PR.ewma_predict(t["U"].to_numpy(), hl)
    assert np.isnan(e[0]) and e[1] == pytest.approx(t["U"].iloc[0])


def test_base_rate_and_log_loss():
    t = table()
    p = PR.base_rate_predict(t, "weekly", min_rows=5)
    y = t["y"].to_numpy()
    ok = np.isfinite(p) & np.isfinite(y)
    assert ((p[ok] > 0) & (p[ok] < 1)).all()
    assert PR.log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(np.log(2))
    assert PR.log_loss(np.array([0.9]), np.array([1])) == pytest.approx(-np.log(0.9))
