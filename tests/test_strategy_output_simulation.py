"""Strategy simulation with simulated head outputs (owner 2026-10-26): known answers for the quality -> noise mapping
(requested AUC achieved within 0.01, requested Spearman within 0.03), deterministic block noise, Platt calibration on
the calibration block only, the simulated C3 table on a synthetic truth (shapes, qualities per block, regime drop,
one-sided variant), the break-even rule and the reality-based NOP. No data, no engine fit."""
import math

import numpy as np
import pandas as pd
import pytest

from tools import strategy_output_simulation as S
from tools import strategy_output_simulation_report as R


@pytest.mark.parametrize("auc", [0.55, 0.62, 0.70, 0.80, 0.90])
def test_requested_auc_is_achieved_within_0_01(auc):
    rng = np.random.default_rng(0)
    y = (rng.random(60000) < 0.3).astype(int)
    z = S.beta_for_auc(auc) * (2 * y - 1) + rng.standard_normal(len(y))
    assert abs(S.auc_of(z, y) - auc) < 0.01


def test_auc_one_is_separable_and_half_is_chance():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.3).astype(int)
    assert S.auc_of(S.beta_for_auc(1.0) * (2 * y - 1) + rng.standard_normal(len(y)), y) > 0.999
    assert abs(S.auc_of(S.beta_for_auc(0.5) * (2 * y - 1) + rng.standard_normal(len(y)), y) - 0.5) < 0.01


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9, 1.0])
def test_requested_correlation_is_achieved(rho):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(80000) * 0.7
    r = x + S.sigma_for_rho(rho, x.std()) * rng.standard_normal(len(x))
    assert abs(np.corrcoef(x, r)[0, 1] - rho) < 0.03
    assert abs(S.spearman(x[:20000], r[:20000]) - rho) < 0.06


def test_noise_is_deterministic_and_day_blocked():
    n1, n2 = S.Noise(7, 0.0), S.Noise(7, 0.0)
    assert np.array_equal(n1.draw(("open", 5, 0), 100, 3), n2.draw(("open", 5, 0), 100, 3))
    assert not np.array_equal(n1.draw(("open", 5, 0), 100, 3), S.Noise(8, 0.0).draw(("open", 5, 0), 100, 3))
    nb = S.Noise(7, 0.6)
    a = np.array([nb.draw(("open", i, 0), i // 10, 1)[0] for i in range(6000)])       # rows i share a day with the 9 next
    b = np.array([nb.draw(("open", i, 1), i // 10, 1)[0] for i in range(6000)])       # same days, same head family -> shared day component
    c = np.array([nb.draw(("open", i, 1), 10000 + i // 10, 1)[0] for i in range(6000)])   # other days -> independent
    assert abs(np.corrcoef(a, b)[0, 1] - 0.36) < 0.05 and abs(np.corrcoef(a, c)[0, 1]) < 0.06


def _truth(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cells = [[0.5, 0.5], [1.0, 1.0], [1.5, 1.0], [0.5, 1.5]]
    period = np.array(["TRAIN"] * (n // 3) + ["INNER_A"] * (n // 3) + ["INNER_B"] * (n - 2 * (n // 3)))
    rows = np.arange(n)
    hit3 = rng.integers(0, 3, size=(n, 2))
    net = rng.standard_normal((n, 2, len(cells))) * 0.8
    net[:, :, 1] += np.where(hit3 == 0, 0.8, np.where(hit3 == 1, -0.9, -0.1))
    best = np.max(net.reshape(n, -1), axis=1)
    cost = float(np.quantile(best, 0.4))                        # 40 % of rows have no cell above the cost: a balanced NOP
    nop = (best <= cost).astype(int)
    day = rows // 24
    cal = np.flatnonzero(period == "INNER_A")
    xvis = {"INNER_A": {"row": cal[:400], "bar": cal[:400], "side": rng.integers(0, 2, 400), "remaining": rng.standard_normal(400) * 0.5,
                        "sl_first": rng.integers(0, 2, 400), "censored": np.zeros(400, bool), "exit_bar": cal[:400] + 5}}
    xvis["INNER_B"] = xvis["INNER_A"]
    return S.Truth("toy", cells, 1, rows, period, day, hit3, net, nop, [-0.1, -0.1], cost, np.full(n, 2.4), np.full(n, 0.3), xvis, {})


class _NoCx:
    pass


def test_simulated_table_matches_requested_quality_per_block(monkeypatch):
    t = _truth()
    monkeypatch.setattr(S, "SimCloser", lambda *a, **k: None)
    tab, sim = S.simulate_table(None, None, t, S.Quality(0.75, 0.75, 0.75, 0.75), seed=3)
    assert tab["net"].shape == (3000, 2, 4) and tab["p3"].shape == (3000, 2, 3) and tab["pn"].shape == (3000,)
    rows = t.rows
    assert np.allclose(tab["p3"][rows].sum(axis=2), 1.0) and np.all(np.isfinite(tab["pn"][rows]))
    a = sim["achieved"]
    for blk in ("calibration", "decision"):
        assert abs(a[blk]["auc_tp_first_side0"] - 0.75) < 0.03 and abs(a[blk]["auc_nop"] - 0.75) < 0.03
        assert abs(a[blk]["spearman_net_side0"] - S.rho_of_level(0.75)) < 0.05      # rho = sin(pi (level - 1/2)) = 0.707
        assert 0.3 < a[blk]["mae_over_naive_net_side0"] < 1.0
    # Platt maps fitted on the calibration block only: the calibrated TP-first probability matches the base rate there
    cal = rows[t.period[rows] == "INNER_A"]
    assert abs(tab["p3"][cal, 0, 0].mean() - (t.hit3[cal, 0] == 0).mean()) < 0.03


def test_regime_drop_and_one_sided_variants(monkeypatch):
    t = _truth(seed=1)
    monkeypatch.setattr(S, "SimCloser", lambda *a, **k: None)
    _, sim = S.simulate_table(None, None, t, S.Quality(0.80, 0.80, 0.80, 0.80, decision_drop=0.15), seed=0)
    assert abs(sim["achieved"]["calibration"]["auc_tp_first_side0"] - 0.80) < 0.03
    assert 0.62 < sim["achieved"]["decision"]["auc_tp_first_side0"] < 0.78     # the drop (0.15) applies from the MIDDLE of the decision block
    _, sim = S.simulate_table(None, None, t, S.quality_for("onesided", 0.80), seed=0)
    assert abs(sim["achieved"]["decision"]["auc_tp_first_side0"] - 0.80) < 0.03
    assert abs(sim["achieved"]["decision"]["auc_tp_first_side1"] - 0.50) < 0.03
    q = S.quality_for("closer_only", 0.8)
    assert q.open == 0.5 and q.close == 0.8 and S.quality_for("opener_only", 0.8).close == 0.5


def test_reality_based_nop_is_one_where_no_cell_beats_the_cost():
    t = _truth(seed=2)
    best = np.max(t.net.reshape(len(t.rows), -1), axis=1)
    assert np.array_equal(t.nop, (best <= t.cost).astype(int)) and 0 < t.nop.mean() < 1


def test_break_even_rule_is_monotone_and_reports_ceiling_and_floor():
    rows = []
    for lv, lo in zip(S.LADDER, [-0.2, -0.1, 0.01, -0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.2, 0.3]):
        rows.append({"variant": "base", "level": lv, "dec_exp_lo_mean": lo, "dec_sharpe": lv, "dec_net": 100 * lv, "dec_exp": lo + 0.1})
    p = pd.DataFrame(rows)
    b = R.break_even(p, "base")
    assert b["break_even_level"] == 0.62 and b["first_positive_level"] == 0.55
    assert b["oracle_ceiling"]["dec_sharpe"] == 1.0 and b["chance_floor"]["dec_sharpe"] == 0.5
    assert R.break_even(pd.DataFrame([{"variant": "x", "level": 0.5, "dec_exp_lo_mean": -1.0, "dec_sharpe": 0, "dec_net": 0, "dec_exp": 0}]), "x")["break_even_level"] is None


def test_pooled_table_counts_refusals_and_failed_checks():
    df = pd.DataFrame([{"variant": "base", "level": 0.7, "seed": s, "refused": None if s else "STRATEGY_SPACE_INFEASIBLE", "checks_failed": 0, "early_close_disabled": False,
                        "auc_dec": 0.7, "mae_over_naive_dec": 0.8, "spearman_dec": 0.7, "dec_sharpe": 1.0 + s, "dec_net": 100.0, "dec_exp": 0.05, "dec_opw": 3.0, "dec_dd": 5.0,
                        "dec_early_close_value": 10.0, "dec_win": 0.5, "buy_win": 0.5, "sell_win": 0.5, "dec_exp_lo": 0.01, "dec_trades": 100} for s in range(3)])
    p = R.pooled(df)
    assert int(p["refused"].iloc[0]) == 1 and int(p["n"].iloc[0]) == 3 and abs(float(p["dec_sharpe"].iloc[0]) - 2.5) < 1e-9


def test_catalog_knobs_lag_drift_saturation_and_inversion(monkeypatch):
    t = _truth(seed=4)
    monkeypatch.setattr(S, "SimCloser", lambda *a, **k: None)
    tab0, _ = S.simulate_table(None, None, t, S.Quality(0.8, 0.8, 0.8, 0.8), seed=0)
    tab1, _ = S.simulate_table(None, None, t, S.quality_for("lag1", 0.8), seed=0)
    r = t.rows
    assert np.allclose(tab1["pn"][r[1:]], tab0["pn"][r[:-1]]) and np.allclose(tab1["net"][r[1:]], tab0["net"][r[:-1]])
    tab2, _ = S.simulate_table(None, None, t, S.quality_for("cal12", 0.8), seed=0)
    z0, z2 = np.log(tab0["pn"][r] / (1 - tab0["pn"][r])), np.log(tab2["pn"][r] / (1 - tab2["pn"][r]))
    assert np.allclose(z2, 1.2 * z0, atol=1e-5)
    tab3, _ = S.simulate_table(None, None, t, S.quality_for("sat", 0.8), seed=0)
    assert set(np.round(np.unique(tab3["pn"][r]), 2)) <= {0.01, 0.99}
    assert S.beta_for_auc(0.3) < 0 and S.quality_for("inv_open", 0.8).open == pytest.approx(0.2)
    _, sim = S.simulate_table(None, None, t, S.quality_for("inv_open", 0.8), seed=0)
    assert abs(sim["achieved"]["decision"]["auc_tp_first_side0"] - 0.2) < 0.03
    tab4, _ = S.simulate_table(None, None, t, S.quality_for("const_net", 0.8), seed=0)
    assert np.all(tab4["net"][r] == 0.0)
    _, sim = S.simulate_table(None, None, t, S.quality_for("regime", 0.8), seed=0)
    dec = r[t.period[r] == "INNER_B"]
    assert abs(sim["achieved"]["calibration"]["auc_tp_first_side0"] - 0.8) < 0.03
    # the drop (0.10) applies from the middle of the decision block: the pooled decision AUC sits between the two halves' values
    assert 0.64 < sim["achieved"]["decision"]["auc_tp_first_side0"] < 0.80
    assert S.quality_for("asym", 0.8).open_s == pytest.approx(0.7) and S.quality_for("nop_only", 0.8).nop == 0.8 and S.quality_for("nop_only", 0.8).open == 0.5
    assert set(S.SCENARIOS) == set(S.VARIANTS)


def test_stage_b_box_restricts_bounds_and_keeps_declared_ones():
    """Stage B (owner 2026-10-26): the gene bounds become box ∩ declared; genes absent from the box keep their bounds; a
    degenerate box widens to 2 % of the declared range so NSGA-II still has a segment to search."""
    from tools import strategy_output_simulation as S
    AC = {"strategy": {"genes": {"a": [0.0, 1.0], "b": [0.2, 0.8], "c": [0.0, 1.0]}, "fitness": {"default": "F2"}}}
    sa = {"family": "c3", "evaluations": 3, "robust_region": {"points": 2, "box": {"a": [0.3, 0.6], "b": [0.0, 0.5], "c": [0.4, 0.4]}}, "best_mean": {"genes": {"a": 0.5}}}
    out = S.apply_stage_a(AC, sa)
    assert out["strategy"]["genes"]["a"] == [0.3, 0.6]
    assert out["strategy"]["genes"]["b"] == [0.2, 0.5]
    lo, hi = out["strategy"]["genes"]["c"]
    assert lo == 0.4 and 0.41 < hi < 0.43
    assert out["strategy"]["stage_a"]["best_mean_genes"] == {"a": 0.5}
    assert AC["strategy"]["genes"]["a"] == [0.0, 1.0]                   # the input contract is untouched
    assert S.apply_stage_a(AC, None) is AC


def test_regression_level_maps_to_information_equivalent_rho():
    """Level 0.5 must be chance for the regression too (round 1 gave rho = level and the chance floor carried
    Spearman 0.48); the mapping rho = sin(pi (level - 1/2)) equals the AUC = 1/2 + arcsin(rho)/pi identity."""
    import math
    from tools import strategy_output_simulation as S
    assert S.rho_of_level(0.5) == 0.0 and S.rho_of_level(1.0) == 1.0
    assert S.rho_of_level(0.7) == pytest.approx(math.sin(math.pi * 0.2))
    for lv in (0.55, 0.62, 0.8):
        r = S.rho_of_level(lv)
        assert 0.5 + math.asin(r) / math.pi == pytest.approx(lv)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    sig = S.sigma_for_rho(S.rho_of_level(0.5), 1.0)
    r = x + sig * rng.standard_normal(20000)
    assert abs(np.corrcoef(x, r)[0, 1]) < 0.02                      # chance floor
    sig7 = S.sigma_for_rho(S.rho_of_level(0.7), 1.0)
    r7 = x + sig7 * rng.standard_normal(20000)
    assert abs(np.corrcoef(x, r7)[0, 1] - math.sin(math.pi * 0.2)) < 0.03
