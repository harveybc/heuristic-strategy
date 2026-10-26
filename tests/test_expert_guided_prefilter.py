"""Strict pre-filter rules R1-R6 on synthetic known answers (no data, no model)."""
import numpy as np
import pytest

from tools import expert_guided_prefilter as P


def test_r1_drops_constant_and_nonfinite_only():
    assert P.r1_constants({"a": 0.0, "b": 1.2, "c": float("nan"), "d": None, "e": 1e-9}) == {"a", "c", "d"}


def test_r2_keeps_higher_family_gain_then_recipe_order_and_moves_blocks():
    cols = ["x", "y", "z", "h_sin", "h_cos"]
    r2 = np.zeros((5, 5))
    r2[0, 1] = r2[1, 0] = 0.995                 # x ~ y (x family gains more)
    r2[2, 3] = r2[3, 2] = 0.999                 # z ~ h_sin (same family gain -> recipe order keeps z)
    fam = {"x": "A", "y": "B", "z": "C", "h_sin": "C", "h_cos": "C"}
    gain = {"A": 0.5, "B": 0.1, "C": 0.0}
    dropped, clusters = P.r2_duplicates(cols, r2, fam, gain, P.coherent_blocks(cols), 0.99)
    assert dropped == {"y", "h_sin", "h_cos"}          # the sin/cos block leaves together
    assert {c["keep"] for c in clusters} == {"x", "z"}
    assert P.r2_duplicates(cols, r2, fam, gain, P.coherent_blocks(cols), 0.999)[0] == {"h_sin", "h_cos"}


def test_r3_drops_proxies_but_keeps_declared_calendar():
    auc = {"cot": 0.80, "vol": 0.74, "hod_sin": 0.95, "cal.utc_dow_cos": 0.9, "none": None}
    assert P.r3_proxies(auc, 0.75, ("hod_", "dow_", "cal.")) == {"cot"}
    assert P.r3_proxies(auc, 0.70, ("hod_", "dow_", "cal.")) == {"cot", "vol"}


def test_r4_drops_only_families_dead_on_every_task_and_spares_rescued():
    fam = {"a": "F1", "b": "F1", "c": "F2", "d": "F3"}
    gain = {"F1": {"t1": -0.1, "t2": 0.0}, "F2": {"t1": -0.1, "t2": 0.001}, "F3": {"t1": -0.5, "t2": -0.5}}
    assert P.r4_family_evidence(fam, gain, {"F3": "RESCUED_INTERACTION"}) == {"a", "b"}


def test_r5_drops_components_dead_on_both_tasks_and_whole_blocks_only():
    cols = ["d1", "d2", "p_sin", "p_cos", "e"]
    cg = {"d1": {"opener3": -0.1, "hit": -0.2}, "d2": {"opener3": -0.1, "hit": 0.05}, "p_sin": {"opener3": -0.1, "hit": -0.1},
          "p_cos": {"opener3": 0.2, "hit": -0.1}, "e": {"opener3": 0.0, "hit": 0.0}}
    out = P.r5_components(cg, P.coherent_blocks(cols), cols, ("opener3", "hit"))
    assert out == {"d1", "e"}                          # p_sin dead but p_cos alive -> the block stays; zero gain counts as dead


def test_r6_keeps_pca_count_by_block_importance_with_floor_two():
    rng = np.random.default_rng(0)
    n = 2000
    base = rng.normal(size=(n, 2))
    Z = {"a": base[:, 0], "b": base[:, 0] * 0.99 + 0.01 * rng.normal(size=n), "c": base[:, 1],
         "d": base[:, 1] * 0.98 + 0.02 * rng.normal(size=n), "e": 0.01 * rng.normal(size=n)}
    fam_cols = {"F": ["a", "b", "c", "d", "e"], "small": ["a", "b"]}
    dropped, detail = P.r6_pca(fam_cols, Z, P.coherent_blocks(list(Z)), 0.99)
    assert detail["F"]["k"] == 2 and dropped == {"c", "d", "e"} or detail["F"]["k"] == 2 and len(dropped) == 3
    assert "small" not in detail                        # families with <= 4 columns are untouched
    tiny = {"G": ["a", "b", "c", "d", "e"]}
    d2, det2 = P.r6_pca(tiny, {k: v for k, v in Z.items()}, P.coherent_blocks(list(Z)), 0.0)
    assert det2["G"]["k"] == 2                          # floor of two columns


def test_pipeline_order_and_reasons_on_a_toy():
    cols = ["k", "x", "y", "w"]
    fam = {"k": "F", "x": "F", "y": "F", "w": "G"}
    meas = {"std": {"k": 0.0, "x": 1.0, "y": 1.0, "w": 1.0}, "r2": np.zeros((4, 4)), "auc": {"k": None, "x": 0.5, "y": 0.9, "w": 0.5},
            "Z": {c: np.zeros(10) for c in cols}}
    meas["r2"][1, 2] = meas["r2"][2, 1] = 0.0
    res = P.run(cols, fam, {}, {"F": {"opener3": 0.1}, "G": {"opener3": -0.1, "hit": -0.1}}, {}, meas, 0.99, 0.75, 0.99, ("opener3", "hit"), ("hod_",))
    assert res["reasons"] == {"k": "R1_constant", "y": "R3_date_proxy", "w": "R4_family_no_evidence"} and res["survivors"] == ["x"]
    assert res["counts"] == {"start": 4, "R1": 3, "R2": 3, "R3": 2, "R4": 1, "R5": 1, "R6": 1}
