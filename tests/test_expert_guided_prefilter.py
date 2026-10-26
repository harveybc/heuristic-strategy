"""Pre-filter v2: definitive vs reversible tiers, every rule on synthetic known answers, plus the selector self-tests
required by the auditor (interaction-only signal survives, useful component of a weak parent survives, exact
duplicates removed, legitimate drift lands in tier B). No data, no model."""
import numpy as np

from tools import expert_guided_prefilter as P

RNG = np.random.default_rng(0)


def _meas(cols, raw, periods=4):
    """build a `measure()`-shaped dict from raw vectors (no NaN), with equal-length pseudo periods."""
    n = len(next(iter(raw.values())))
    Z = {c: np.asarray(raw[c], float) for c in cols}
    bounds = np.linspace(0, n, periods + 1).astype(int)
    Zp = {f"p{i}": {c: Z[c][bounds[i]:bounds[i + 1]] for c in cols} for i in range(periods)}
    y = (np.arange(n) >= bounds[-2]).astype(int)
    from sklearn.metrics import roc_auc_score
    auc = {c: (None if np.std(Z[c]) == 0 else float(max(roc_auc_score(y, Z[c]), 1 - roc_auc_score(y, Z[c])))) for c in cols}
    return {"raw": Z, "blocks_per_col": {c: [Zp[p][c] for p in Zp] for c in cols}, "Z": Z, "Zp": Zp, "auc": auc, "auc_year": {c: {} for c in cols}}


def test_d1_constant_vs_absent_with_record():
    const, absent, rec = P.d1_constants({"k": np.ones(50), "a": np.full(50, np.nan), "x": np.r_[np.nan, RNG.normal(size=49)]})
    assert const == {"k"} and absent == {"a"} and rec["x"]["finite"] == 49 and rec["a"]["class"] == "ABSENT" and rec["k"]["class"] == "CONSTANT"


def test_d2_exact_duplicates_identical_and_affine_only():
    x = RNG.normal(size=500)
    Z = {"x": x, "y": x.copy(), "w": 24 - x, "near": x + 0.001 * RNG.normal(size=500)}
    cols = list(Z)
    dropped, clusters, pairs = P.d2_exact_duplicates(cols, Z, {c: "F" for c in cols}, {"F": 0}, P.coherent_blocks(cols), 1e-9)
    assert dropped == {"y", "w"} and "near" not in dropped and clusters[0]["keep"] == "x"
    assert {p[2].split()[0] for p in pairs} == {"identical", "affine"}


def test_b1_near_duplicates_go_to_tier_b_with_period_stability_and_blocks():
    x = RNG.normal(size=800)
    Z = {"x": x, "n": x + 0.05 * RNG.normal(size=800), "s_sin": x + 0.05 * RNG.normal(size=800), "s_cos": RNG.normal(size=800)}
    cols = list(Z)
    m = _meas(cols, Z)
    tier_b, clusters, pairs = P.b1_near_duplicates(cols, m["Z"], m["Zp"], {c: "F" for c in cols}, {"F": 0}, P.coherent_blocks(cols), 0.99)
    assert tier_b == {"n", "s_sin", "s_cos"} and clusters[0]["keep"] == "x"       # the sin/cos block moves together
    assert all(set(p["r2_per_period"]) == {"p0", "p1", "p2", "p3"} for p in pairs)


def test_b2_drift_alert_only_for_raw_levels_and_never_calendar_or_ratios():
    auc = {"cot.eur.net_long": 0.9, "ATR_ratio": 0.9, "hod_sin": 0.95, "ATR": 0.8, "px.logret_4": 0.9, "x": 0.5}
    flagged, rec = P.b2_drift_alert(auc, {}, 0.75)
    assert flagged == {"cot.eur.net_long", "ATR"} and rec["hod_sin"]["level"] is False and rec["ATR_ratio"]["level"] is False


def test_b3_family_order_is_ordering_and_only_no_evidence_goes_to_tier_b():
    fam = {"a": "F1", "b": "F2", "c": "F3"}
    rank, tier_b, best = P.b3_family_order(fam, {"F1": {"t": 0.5}, "F2": {"t": -0.1, "u": 0.0}, "F3": {"t": -0.3}}, {"F3": "RESCUED_INTERACTION"})
    assert rank["F1"] < rank["F2"] and tier_b == {"b"}             # F3 rescued stays; F1 ordered first


def test_b5_forward_selection_is_column_selection_with_blocks_and_floor():
    n = 3000
    u, v = RNG.normal(size=n), RNG.normal(size=n)
    Z = {"a": u, "b": u + 0.01 * RNG.normal(size=n), "c": v, "d": v + 0.01 * RNG.normal(size=n), "e": 0.1 * RNG.normal(size=n)}
    Zs = {c: (x - x.mean()) / x.std() for c, x in Z.items()}
    tier_b, detail = P.b5_forward_variance({"F": list(Z)}, Zs, P.coherent_blocks(list(Z)), 0.99, 2, 5)
    kept = detail["F"]["kept"]
    assert len(kept) == 3 and {kept[0], kept[1]} <= {"a", "b", "c", "d"} and set(tier_b) == set(Z) - set(kept)
    assert all(k in detail["F"]["note"] for k in ("variance", "utility"))
    _, d2 = P.b5_forward_variance({"G": list(Z)}, Zs, P.coherent_blocks(list(Z)), 0.0, 2, 5)
    assert len(d2["G"]["kept"]) == 2                                  # floor of two columns


# ------------------------------------------------------------------ auditor's selector self-tests (synthetic)
def _pipeline(cols, raw, fam, task_gain=None, status=None):
    m = _meas(cols, raw)
    return P.run(cols, fam, status or {}, task_gain or {f: {"opener3": 0.1} for f in set(fam.values())}, m, 0.99, 0.75, 0.99, 1e-9, 2, 5)


def test_interaction_only_signal_survives_in_a_tier():
    n = 2000
    a, b = RNG.normal(size=n), RNG.normal(size=n)          # useless alone, useful together (xor-like) -> no rule may delete them
    cols = ["a", "b", "k"]
    res = _pipeline(cols, {"a": a, "b": b, "k": np.ones(n)}, {"a": "F", "b": "F", "k": "G"}, {"F": {"opener3": -0.01, "hit": -0.02}, "G": {"opener3": 0.1}})
    assert set(res["tier_a"]) | set(res["tier_b"]) >= {"a", "b"} and res["reasons"]["k"] == "D1_constant"
    assert res["reasons"].get("a") == "B3_family_no_evidence"       # reversible, never definitive


def test_useful_component_of_weak_parent_survives():
    n = 2000
    comp, parent = RNG.normal(size=n), RNG.normal(size=n)
    res = _pipeline(["parent", "comp"], {"parent": parent, "comp": comp}, {"parent": "W", "comp": "W"}, {"W": {"opener3": -0.05, "hit": 0.0}})
    assert {"parent", "comp"} <= set(res["tier_a"]) | set(res["tier_b"])   # ordering only; nothing excluded by evidence


def test_exact_duplicates_are_definitive_and_legitimate_drift_is_reversible():
    n = 2000
    x = RNG.normal(size=n)
    level = np.linspace(0, 3, n) + 0.3 * RNG.normal(size=n)          # a volatility LEVEL with a real trend
    cols = ["ATR", "ATR_copy", "x"]
    res = _pipeline(cols, {"ATR": level, "ATR_copy": level.copy(), "x": x}, {c: "F" for c in cols})
    assert res["reasons"]["ATR_copy"] == "D2_exact_duplicate"
    assert res["reasons"]["ATR"] == "B2_drift_alert_level" and "ATR" in res["tier_b"] and "ATR" not in res["tier_a"]
    assert res["counts"]["tier_A"] + res["counts"]["tier_B"] + res["counts"]["D2_exact_duplicate"] == 3
