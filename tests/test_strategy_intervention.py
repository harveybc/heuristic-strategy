"""Intervention battery (owner 2026-10-26): persistent shocks, spliced regime paths (continuous at the joints), spread
blow-up window profile, missing-observation runs, the loss-limiting ratio with its bootstrap CI, and the per-family
ablations. Synthetic inputs only."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import execution as X
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import strategy_full as SF
from tools import strategy_intervention as I


def _bars(n=200, seed=0):
    rng = np.random.default_rng(seed)
    c = 1.1 + np.cumsum(rng.standard_normal(n)) * 1e-3
    idx = pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": c - 1e-4, "HIGH": c + 3e-4, "LOW": c - 3e-4, "CLOSE": c}, index=idx)


def test_shock_is_persistent_and_sized_in_sigma():
    b = _bars()
    sig = np.full(len(b), 5e-4)
    b2 = I.shocked_bars(b, sig, 50, -1, 3.0)
    d = (b2["CLOSE"] - b["CLOSE"]).to_numpy()
    assert np.all(d[:50] == 0) and np.allclose(d[50:], -3.0 * 5e-4)
    assert np.allclose((b2["HIGH"] - b["HIGH"]).to_numpy()[50:], -1.5e-3)


def test_spliced_path_is_continuous_and_uses_donor_returns():
    b = _bars(400, seed=1)
    segs = [("2022-01-05", "2022-01-07", "2022-01-12", "2022-01-14")]
    b2 = I.spliced_bars(b, segs)
    v = b.index.as_unit("ns").asi8
    from app.short_horizon.expert_guided import data as DG
    t0 = int(np.flatnonzero(v >= DG._utc_ns("2022-01-05"))[0])
    assert abs(b2["CLOSE"].iloc[t0 - 1] - b["CLOSE"].iloc[t0 - 1]) < 1e-12          # untouched before
    assert abs(np.log(b2["CLOSE"].iloc[t0] / b2["CLOSE"].iloc[t0 - 1])) < 1e-12     # continuous at the joint
    r2 = np.diff(np.log(b2["CLOSE"].to_numpy()))
    r1 = np.diff(np.log(b["CLOSE"].to_numpy()))
    after = int(np.flatnonzero(v >= DG._utc_ns("2022-01-07"))[0])
    assert np.allclose(r2[after:], r1[after:])                                      # the path after the segment keeps its own returns
    assert not np.allclose(r2[t0:after], r1[t0:after])                              # the segment carries the donor's returns


def test_spread_window_profile_multiplies_inside_the_window_only():
    base = X.RealisticProfile()
    t0 = 10 ** 15
    p = I.SpreadWindowProfile(**dataclasses.asdict(base), window_t0_ns=t0, window_hours=24.0, factor=5.0)
    open_ns = t0 - 3 * I.H_NS
    assert p.spread_at(t0 + 5 * I.H_NS, open_ns) == pytest.approx(5.0 * base.spread_at(t0 + 5 * I.H_NS, open_ns))
    assert p.spread_at(t0 + 30 * I.H_NS, open_ns) == pytest.approx(base.spread_at(t0 + 30 * I.H_NS, open_ns))


def test_ratio_ci_known_answer_and_tolerance_verdict():
    r = I.ratio_ci([50.0] * 20, [100.0] * 20)
    assert r["ratio"] == pytest.approx(0.5) and r["low"] == pytest.approx(0.5) and r["high"] == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    w, wo = rng.normal(60, 10, 40), rng.normal(100, 10, 40)
    r = I.ratio_ci(w, wo)
    assert 0.45 < r["low"] < r["ratio"] < r["high"] < 0.8
    assert np.isnan(I.ratio_ci([1.0], [0.0])["ratio"])


def test_ablations_per_family():
    st = C3.StrategyC3(0.1, 0.0, 0.0, 0.9, 0.9, False, 0.8, True)
    assert I.ablate(st, "early_close", "c3").early_close is False and I.ablate(st, "nop_veto", "c3").nop_veto is None
    assert I.applies("nop_veto", "c3", "perbar", st) and not I.applies("volume", "c3", "perbar", st)
    g = {"open_margin": 0, "close_margin_l": 0, "close_margin_s": 0, "close_p_l": 1, "close_p_s": 1, "allow_reverse": 0, "nop_veto": 0.8,
         "observe_bars": 2, "v_alpha_l": 0.3, "v_beta_l": 0.5, "v_gamma_l": 0.2, "v_delta_l": 0.1, "v_alpha_s": 0.3, "v_beta_s": 0.5, "v_gamma_s": 0.2, "v_delta_s": 0.1, "v_min": 0.1}
    sf = SF.StrategyFull.from_genes(g, True, "cell", "param")
    assert I.applies("volume", "param", "perbar", sf) and I.applies("best_of_day", "param", "perbar", sf)
    a = I.ablate(sf, "volume", "param")
    assert a.coef["v_alpha_l"] == 1.0 and a.coef["v_beta_l"] == 0.0 and a.v_min == 0.0
    assert I.ablate(sf, "best_of_day", "param").observe_bars == 0 and I.ablate(sf, "nop_veto", "param").base.nop_veto is None
