"""Hierarchical catalog and Stage-A machinery (owner 2026-10-26): the synthetic generator's witnesses (drift sign, vol
ratio, fine-pivot density monotone in the L2 level), the Latin-hypercube design, the catalog's levels, the composite
fitness with normalised coefficients, CVaR-10 %, the Pareto front and the robust-region rule. No data, no engine."""
import numpy as np
import pandas as pd
import pytest

from tools import strategy_hierarchy as H


def _index(n=3000):
    return pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")


def test_generator_witnesses_match_the_leaf():
    idx = _index()
    sd = 4e-4
    up = H.synthetic_path(idx, sd, "trend_up", "low", "medium", 1)
    dn = H.synthetic_path(idx, sd, "trend_down", "low", "medium", 1)
    hi = H.synthetic_path(idx, sd, "range", "high", "medium", 1)
    r_up = np.diff(np.log(up["CLOSE"].to_numpy()))
    r_dn = np.diff(np.log(dn["CLOSE"].to_numpy()))
    r_hi = np.diff(np.log(hi["CLOSE"].to_numpy()))
    assert r_up.sum() > 0 and r_dn.sum() < 0
    assert r_hi.std() > 1.5 * r_up.std()                           # high vol >> low vol
    assert np.all(up["HIGH"] >= up[["OPEN", "CLOSE"]].max(axis=1)) and np.all(up["LOW"] <= up[["OPEN", "CLOSE"]].min(axis=1))


def test_fine_structure_density_is_monotone():
    idx = _index()
    sd = 4e-4
    dens = [H.fine_pivot_count(H.synthetic_path(idx, sd, "range", "low", fine, 3), sd) for fine in ("low", "medium", "high")]
    assert dens[0] < dens[1] < dens[2]


def test_lhs_covers_every_stratum_once():
    u = H.lhs(20, 4, 0)
    for j in range(4):
        assert sorted(np.floor(u[:, j] * 20).astype(int).tolist()) == list(range(20))


def test_catalog_levels_and_ids():
    leaves = H.catalog(n_lhs=10, seed=1)
    lv = {k: sum(1 for x in leaves if x["level"] == k) for k in ("L1xL2", "L1xL4", "LHS")}
    assert lv == {"L1xL2": 18, "L1xL4": 36, "LHS": 10}
    assert len({x["id"] for x in leaves}) == len(leaves)
    assert all({"expected", "witness", "regime", "vol", "fine", "variant", "quality"} <= set(x) for x in leaves)
    assert all(x["exec"] for x in leaves if x["level"] == "LHS")


def test_composite_cvar_pareto_and_robust_region():
    scales = {"sharpe": 2.0, "r_ann": 0.5, "opd": 0.5}
    c = {"sharpe": 1.0, "r_ann": 0.25, "opd": 0.5, "trades": 10}
    assert H.composite(c, scales) == pytest.approx(0.5 + 0.5 + 1.0)
    assert np.isnan(H.composite({"sharpe": float("nan"), "trades": 0}, scales))
    assert H.cvar(np.arange(1.0, 11.0), 0.10) == 1.0 and H.cvar(np.arange(1.0, 101.0), 0.10) == pytest.approx(5.5)
    pts = [{"genes": {"g": i}, "mean": m, "cvar10": cv} for i, (m, cv) in enumerate([(1.0, 0.1), (0.9, 0.5), (0.5, 0.4), (1.0, 0.1)])]
    front = H._pareto(pts)
    assert {(p["mean"], p["cvar10"]) for p in front} == {(1.0, 0.1), (0.9, 0.5)}
    worlds = [{"leaf": {"regime": r}} for r in ("trend_up", "range")]
    mk = lambda g, s1, s2: {"genes": {"g": g}, "mean": (s1 + s2) / 2, "cvar10": min(s1, s2), "valid": True,  # noqa: E731
                            "per_leaf": [{"sharpe": s1, "r_ann": 0.0, "opd": 0.0, "trades": 5}, {"sharpe": s2, "r_ann": 0.0, "opd": 0.0, "trades": 5}]}
    valid = [mk(0.1, 2.0, 2.0), mk(0.2, 1.98, 1.99), mk(0.9, 2.0, 1.0)]
    reg = H.robust_region(valid, worlds, ["g"])
    assert reg["points"] == 2 and reg["box"]["g"] == [0.1, 0.2]


def test_robust_region_escalates_the_tolerance_when_regimes_trade_off():
    """Owner rule 5 % first; when no point is near-best in every L1 regime at once, the tolerance widens along
    TOLERANCES until MIN_ROBUST_POINTS points qualify, and both the tolerance used and the 5 % result are recorded."""
    worlds = [{"leaf": {"regime": r}} for r in ("trend_up", "trend_down")]
    mk = lambda g, s1, s2: {"genes": {"g": g}, "mean": (s1 + s2) / 2, "cvar10": min(s1, s2), "valid": True,  # noqa: E731
                            "per_leaf": [{"sharpe": s1, "r_ann": 0.0, "opd": 0.0, "trades": 5}, {"sharpe": s2, "r_ann": 0.0, "opd": 0.0, "trades": 5}]}
    valid = [mk(0.1, 10.0, 2.0), mk(0.2, 2.0, 10.0), mk(0.3, 7.0, 7.0), mk(0.4, 6.5, 6.5), mk(0.5, 6.0, 6.0), mk(0.6, 5.5, 5.5)]
    reg = H.robust_region_escalated(valid, worlds, ["g"])
    assert reg["at_5pct"]["points"] == 0
    assert reg["points"] >= H.MIN_ROBUST_POINTS and reg["tolerance_used"] > 0.05
    assert reg["box"]["g"][0] <= 0.3 and reg["box"]["g"][1] >= 0.3
