"""Walk-forward inside development, seed ensembles and Sharpe uncertainty (owner 2026-10-25): the two fold contracts,
the derived fold recipe (constants re-fitted on the fold's fit block, everything else byte-identical, declared base),
the launcher's WF stage (refusals + emission), the probability ensemble (N = 1 equals the single member exactly), the
Lo standard error and the stationary bootstrap (known answers), and the pre-registered verdict rules. No data, no fit."""
import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import stats as ST  # noqa: E402
from tests.test_expert_guided_clean_round import COLS, make_recipe, spec  # noqa: E402,F401  (fixture)
from tools import expert_guided_clean_round as C  # noqa: E402
from tools import expert_guided_seed_ensemble as SE  # noqa: E402
from tools import expert_guided_wf_recipe as WR  # noqa: E402


# ----------------------------------------------------------------------------- contracts
def test_walk_forward_contracts_roll_the_consolidated_shape_back_and_keep_test_closed():
    for fold, fit_end, val_end in (("wf2022", "2022-01-01", "2023-01-01"), ("wf2023", "2023-01-01", "2024-01-01")):
        sc = DG.split_contract(fold)
        P = sc["periods"]
        assert P["INNER_A"][1] == fit_end and P["INNER_B"] == (fit_end, val_end) and P["TRAIN"][0] == "2020-01-01"
        assert sc["scaler_fit"] == P["TRAIN"] and sc["selection"] == "INNER_A" and "INNER_B" in sc["report"]
        assert sc["train"] == ("2020-01-01", fit_end) and sc["validation"] == (fit_end, val_end)
        assert all(b <= DG.TEST_CLOSED_FROM for _, b in P.values())
    assert DG.split_contract("consolidated")["periods"]["TRAIN"] == ("2020-01-01", "2023-07-01")   # the principal contract is untouched


# ----------------------------------------------------------------------------- derived recipe
def synthetic_store(tmp_path, n=4000):
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": ts, "a": rng.normal(1.0, 2.0, n).cumsum(), "b": rng.normal(0, 1, n), "c": np.full(n, 7.0), "d": rng.normal(5, 3, n)})
    p = tmp_path / "store.parquet"
    df.to_parquet(p)
    return p


def base_recipe():
    r = make_recipe("0" * 64)
    r["tensor"]["columns"][0]["consumer_ops"] = ["DIFF"]
    r["scaler"] = {"policy": "fixed_train", "features": COLS, "n_features": 4, "fit_period": ["2020-01-01", "2023-07-01"], "fit_rows": 1, "ddof": 0,
                   "mean": [0.0] * 4, "std": [1.0] * 4, "std_raw": [1.0] * 4, "variance": [1.0] * 4, "rule": ["z"] * 4, "fit_rows_per_feature": [1] * 4,
                   "nan_share_train": [0.0] * 4, "nan_policy": "nanmean/nanstd", "transform": "fixed TRAIN z", "stationarity": {"policy": "diff"}, "artifact_sha256": "x"}
    r["split"] = {"arm": "consolidated"}
    r["scope"] = "test"
    return r


def test_derived_fold_recipe_changes_only_constants_and_split_and_declares_its_base(tmp_path):
    store = synthetic_store(tmp_path)
    base = base_recipe()
    rec = WR.derive(base, "basesha", store, "wf2022")
    for k in WR.COPIED_KEYS:
        assert rec[k] == base[k]
    assert rec["split"]["arm"] == "wf2022" and rec["split"]["periods"]["INNER_B"] == ["2022-01-01", "2023-01-01"]
    assert rec["derived_from"] == {"identity": base["identity"], "sha256": "basesha", "deviation": WR.DEVIATION, "fold": "wf2022"}
    assert rec["scientific_release"] is False and rec["identity"].startswith("recipe_") and rec["identity"] != base["identity"]
    sc = rec["scaler"]
    assert sc["fit_period"] == ["2020-01-01", "2021-07-01"] and sc["features"] == COLS and "artifact_sha256" not in sc
    # constants: column a is DIFFed (consumer op) -> mean ~ 1, std ~ 2 on the fit block; constant column c -> std_zero_to_one
    df = pd.read_parquet(store)
    ts = pd.to_datetime(df["timestamp"], utc=True) + pd.Timedelta(hours=1)
    fit = (ts >= pd.Timestamp("2020-01-01", tz="UTC")) & (ts < pd.Timestamp("2021-07-01", tz="UTC"))
    da = np.r_[np.nan, np.diff(df["a"].to_numpy())][1:][fit.to_numpy()[1:]]      # the loader drops the first row (DIFF NaN)
    assert math.isclose(sc["mean"][0], np.nanmean(da), rel_tol=1e-9) and math.isclose(sc["std"][0], np.nanstd(da), rel_tol=1e-9)
    assert sc["rule"][2] == "std_zero_to_one" and sc["std"][2] == 1.0 and sc["std_raw"][2] == 0.0
    assert sc["fit_rows"] == int(fit.sum())                                   # rows in the fit window (the DIFF NaN row counts, as in the loader)
    # identity covers the constants: a different fold -> a different identity; same inputs -> the same identity
    assert WR.derive(base, "basesha", store, "wf2023")["identity"] != rec["identity"]
    assert WR.derive(base, "basesha", store, "wf2022")["identity"] == rec["identity"]
    with pytest.raises(ValueError, match="WF_FOLD_UNKNOWN"):
        WR.derive(base, "basesha", store, "consolidated")


# ----------------------------------------------------------------------------- launcher WF stage
def wf_spec(spec_fixture, tmp_path):
    s, t = spec_fixture
    s = copy.deepcopy(s)
    base = json.loads((t / "CONSUMER_RECIPE.json").read_text())
    folds = {}
    for fold in ("wf2022", "wf2023"):
        fr = {k: copy.deepcopy(base[k]) for k in WR.COPIED_KEYS}
        sc = DG.split_contract(fold)
        fr.update(schema=base["schema"], scientific_release=False, scaler=dict(base["scaler"], fit_period=list(sc["scaler_fit"]), mean=[0.5] * 4),
                  split={"arm": fold}, identity=f"recipe_{fold}0000000000", derived_from={"identity": base["identity"], "sha256": s["recipe"]["sha256"], "fold": fold})
        p = t / f"CONSUMER_RECIPE_{fold}.json"
        p.write_text(json.dumps(fr))
        folds[fold] = {"recipe": {"path": str(p), "sha256": C.sha(p), "identity": fr["identity"]}}
    s["walk_forward"] = {"folds": folds, "arms": ["zz", "opp_hold"], "seeds": "1,2,3,4,5", "rule": "pre-registered text", "memory": "10G"}
    return s, t


def test_wf_stage_validates_and_refuses_every_divergence(spec):  # noqa: F811
    s, t = wf_spec(spec, None)
    assert C.validate(s) == []
    def mutate_fold(fn, code):
        s2 = copy.deepcopy(s)
        p = t / "CONSUMER_RECIPE_wf2022.json"
        fr = json.loads(p.read_text())
        fn(fr, s2)
        p2 = t / "R_wf.json"
        p2.write_text(json.dumps(fr))
        s2["walk_forward"]["folds"]["wf2022"]["recipe"] = {"path": str(p2), "sha256": C.sha(p2), "identity": fr["identity"]}
        assert any(e.startswith(code) for e in C.validate(s2)), (code, C.validate(s2))
    mutate_fold(lambda fr, s2: fr["representation"]["model"].update(head=32), "WF_FOLD_DIVERGES:WF/wf2022:representation")
    mutate_fold(lambda fr, s2: fr["tensor"]["columns"][0].update(consumer_ops=["DIFF"]), "WF_FOLD_DIVERGES:WF/wf2022:tensor")
    mutate_fold(lambda fr, s2: fr["split"].update(arm="wf2023"), "WF_FOLD_SPLIT_MISMATCH")
    mutate_fold(lambda fr, s2: fr["derived_from"].update(identity="recipe_other"), "WF_FOLD_NOT_DERIVED_FROM_RECIPE")
    mutate_fold(lambda fr, s2: fr["scaler"].update(fit_period=["2020-01-01", "2023-07-01"]), "WF_FOLD_SCALER_NOT_FOLD_LOCAL")
    mutate_fold(lambda fr, s2: fr.update(scientific_release=True), "WF_FOLD_SCHEMA")
    s2 = copy.deepcopy(s); s2["walk_forward"]["folds"]["wf2022"]["recipe"]["sha256"] = "0" * 64
    assert any(e.startswith("WF_FOLD_DIGEST_MISMATCH") for e in C.validate(s2))
    s2 = copy.deepcopy(s); s2["walk_forward"]["arms"] = ["nope"]
    assert any(e.startswith("WF_ARM_UNKNOWN") for e in C.validate(s2))
    s2 = copy.deepcopy(s); s2["walk_forward"]["rule"] = ""
    assert any(e.startswith("WF_RULE_NOT_PREREGISTERED") for e in C.validate(s2))
    s2 = copy.deepcopy(s); s2["walk_forward"]["folds"]["bad"] = s2["walk_forward"]["folds"]["wf2022"]
    assert any(e.startswith("WF_FOLD_UNKNOWN") for e in C.validate(s2))
    s2 = copy.deepcopy(s); s2.pop("walk_forward")
    assert C.validate(s2) == []                                                         # additive: no block, no checks


def test_wf_emission_uses_each_folds_recipe_split_and_seeds_and_keeps_the_gate(spec, tmp_path):  # noqa: F811
    s, t = wf_spec(spec, None)
    sp = t / "SPEC_WF.json"
    sp.write_text(json.dumps(s))
    (t / "LAUNCH_EXPERT_GUIDED.json").write_text("{}")
    a = type("A", (), {"spec": str(sp), "stage": "WF", "winner_from": None, "out": str(t / "q_wf.sh"), "arms": None, "seeds": None, "out_suffix": None})()
    C.cmd_emit(a)
    q = (t / "q_wf.sh").read_text()
    assert q.count("expert_guided_e3m.py run") == 4 and "LAUNCH_EXPERT_GUIDED.json" in q and 'log "STAGE_WF_DONE"' in q
    assert "--recipe $Q/recipe/CONSUMER_RECIPE_wf2022.json" in q and "--split-contract wf2022" in q and "--seeds 1,2,3,4,5" in q
    assert "--recipe $Q/recipe/CONSUMER_RECIPE_wf2023.json" in q and "--split-contract wf2023" in q
    assert q.count("-m 10G") == 4 and "STATE/WF_wf2022_zz" in q and "STATE/WF_wf2023_opp_hold" in q
    assert all(f not in q for f in C.FORBIDDEN_FLAGS) and "rolling" not in q and "consolidated" not in q.split("WF_wf2022_zz")[1].split("fi")[0]
    loc = C.localize(s, "/pkg")
    assert loc["walk_forward"]["folds"]["wf2023"]["recipe"]["path"] == "/pkg/recipe/CONSUMER_RECIPE_wf2023.json"


# ----------------------------------------------------------------------------- ensemble
class _Member:
    def __init__(self, lg, rg):
        self.lg, self.rg = np.asarray(lg, float), (None if rg is None else np.asarray(rg, float))

    def __call__(self, x):
        return self.lg[None, :], (None if self.rg is None else self.rg[None, :])


def test_probability_ensemble_of_one_equals_the_member_and_of_many_is_the_mean_probability():
    m = _Member([2.0, -1.0, 0.5], [1.2, 2.5])
    lg, rg = SE.ProbEnsemble([m])(np.zeros((1, 3)))
    sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()  # noqa: E731
    assert np.allclose(sm(lg[0]), sm(m.lg)) and np.allclose(rg[0], m.rg)
    m2 = _Member([-1.0, 3.0, 0.0], [0.8, 1.5])
    lg2, rg2 = SE.ProbEnsemble([m, m2])(np.zeros((1, 3)))
    assert np.allclose(sm(lg2[0]), (sm(m.lg) + sm(m2.lg)) / 2) and np.allclose(rg2[0], [1.0, 2.0])
    with pytest.raises(ValueError, match="ENSEMBLE_EMPTY"):
        SE.ProbEnsemble([])


def test_ensemble_arbiter_rule_is_majority_then_prob_with_median_rate():
    tr = {"1": {"arbiter": {"arbiter": "prob", "rate": 0.04}}, "2": {"arbiter": {"arbiter": "prob", "rate": 0.10}}, "3": {"arbiter": {"arbiter": "expected_gain", "rate": None}}}
    a = SE.ensemble_arbiter(tr)
    assert a["arbiter"] == "prob" and math.isclose(a["rate"], 0.07)
    tr2 = {"1": {"arbiter": {"arbiter": "expected_gain"}}, "2": {"arbiter": {"arbiter": "gain_over_risk"}}, "3": {"arbiter": {"arbiter": "expected_gain"}}}
    assert SE.ensemble_arbiter(tr2)["arbiter"] == "expected_gain" and SE.ensemble_arbiter(tr2)["rate"] is None
    tr3 = {"1": {"arbiter": {"arbiter": "expected_gain"}}, "2": {"arbiter": {"arbiter": "prob", "rate": 0.06}}}
    assert SE.ensemble_arbiter(tr3)["arbiter"] == "prob"                               # tie -> prob


# ----------------------------------------------------------------------------- Sharpe uncertainty
def test_lo_standard_error_and_bootstrap_have_known_answers():
    ret = np.array([0.01, -0.01] * 100)                       # mean 0 -> SR 0 -> SE = sqrt(1/T) * sqrt(365)
    assert math.isclose(ST.sharpe(ret), 0.0) and math.isclose(ST.sharpe_se_lo(ret), math.sqrt(1 / 200) * math.sqrt(365))
    ret2 = np.array([0.002] * 50 + [0.0] * 50)                # SR_d = mean/sd; SE = sqrt((1 + SR_d^2/2)/T) * sqrt(365)
    sr_d = ret2.mean() / ret2.std(ddof=1)
    assert math.isclose(ST.sharpe_se_lo(ret2), math.sqrt((1 + sr_d ** 2 / 2) / 100) * math.sqrt(365))
    assert math.isclose(ST.sharpe(ret2), sr_d * math.sqrt(365))
    assert math.isnan(ST.sharpe_se_lo(np.array([0.1]))) and ST.sharpe(np.array([0.1])) == 0.0
    # the engine's return series: equity per day preceded by the initial capital
    assert np.allclose(ST.daily_returns([110.0, 99.0], 100.0), [0.1, -0.1])
    # bootstrap: seeded -> reproducible; a drift with tiny noise -> an interval that excludes zero and brackets the point
    rng = np.random.default_rng(1)
    ret3 = 0.001 + 1e-4 * rng.normal(size=400)
    b1 = ST.stationary_bootstrap_ci(ret3, n=300, seed=7)
    b2 = ST.stationary_bootstrap_ci(ret3, n=300, seed=7)
    assert b1 == b2 and b1["low"] > 0 and b1["low"] <= ST.sharpe(ret3) <= b1["high"]
    idx = ST.stationary_bootstrap_indices(50, 5.0, np.random.default_rng(3))
    assert idx.shape == (50,) and idx.min() >= 0 and idx.max() < 50


def test_preregistered_verdict_rules():
    h = ST.holds({"wf2022": {"sharpe": 1.0, "se_lo": 0.4}, "wf2023": {"sharpe": 0.5, "se_lo": 0.4}})
    assert h["verdict"] == "HOLDS" and h["rule"] == ST.HOLDS_RULE
    assert ST.holds({"wf2022": {"sharpe": 1.0, "se_lo": 0.4}, "wf2023": {"sharpe": 0.3, "se_lo": 0.4}})["verdict"] == "DOES_NOT_HOLD"
    assert ST.holds({})["verdict"] == "DOES_NOT_HOLD"
    s = ST.separable({"sharpe": 1.5, "se_lo": 0.5}, {"sharpe": 0.5, "se_lo": 0.5})
    assert s["verdict"] == "SEPARABLE" and math.isclose(s["pooled_se"], math.sqrt(0.5)) and math.isclose(s["delta"], 1.0)
    assert ST.separable({"sharpe": 1.0, "se_lo": 0.5}, {"sharpe": 0.5, "se_lo": 0.5})["verdict"] == "NOT_SEPARABLE"
    assert ST.separable({"sharpe": 1.0, "se_lo": float("nan")}, {"sharpe": 0.5, "se_lo": 0.5})["verdict"] == "NOT_SEPARABLE"
