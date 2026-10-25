"""Stage C2 (causal-ladder design, 2026-10-25): outcome tables from the engine with known answers, label availability,
the C2 contract and its encoding audit (the 1-continuous-output-cut-into-classes design is refused), the C2 strategy
layer, the no-peek guarantee of the strategy optimiser, the E3-M flag guard, the launcher's WF_C2 block and the
report rules. No data, no model fit."""
import copy
import json
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import calendar as CAL
from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import c2 as C2
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import outcomes as OC

HOUR = pd.Timedelta(hours=1)
SIG = 0.0010
ROOT = Path(__file__).resolve().parents[1]


def synthetic(move: str):
    """3 weekly segments x 120 h, flat 1.1000 around the decision bar b, one move bar at b + 4 ('up' or 'down', 4 sigma),
    flat afterwards. Returns (env factory, b)."""
    stamps, t = [], pd.Timestamp("2020-01-05 21:00", tz="UTC")
    for _ in range(3):
        stamps += [t + k * HOUR for k in range(120)]
        t = stamps[-1] + 49 * HOUR
    n = len(stamps)
    bars = pd.DataFrame({"OPEN": 1.1, "HIGH": 1.1, "LOW": 1.1, "CLOSE": 1.1}, index=pd.DatetimeIndex(stamps, name="DATE_TIME"))
    segs = CAL.derive_segments(bars.index, max_gap_hours=24)
    cal = CAL.load_calendar({"calendar_id": "TEST", "version": "t1", "segments": segs, "entry_delay_after_open_hours": 12,
                             "force_flat_before_close_hours": 12, "entry_cutoff_before_flat_hours": 0})
    probe = E.BarEnv(bars, cal, np.full(n, SIG), np.full(n, 3 * SIG), bars.index[0], bars.index[-1] + HOUR, E.EnvConfig())
    b = int(np.searchsorted(probe.v, cal.segments[0].operational_start.value)) - 1 + 5
    lvl, m = 1.1, b + 4
    new = lvl + 4 * SIG if move == "up" else lvl - 4 * SIG
    bars.iloc[m] = [lvl, max(lvl, new), min(lvl, new), new]
    for k in range(m + 1, n):
        bars.iloc[k] = [new, new, new, new]
    mk = lambda: E.BarEnv(bars, cal, np.full(n, SIG), np.full(n, 3 * SIG), bars.index[0], bars.index[-1] + HOUR, E.EnvConfig())  # noqa: E731
    return mk, b


@pytest.mark.parametrize("move, d, hit, sign", [("up", 1, OC.HIT_TP, 1), ("up", -1, OC.HIT_SL, -1), ("down", 1, OC.HIT_SL, -1), ("down", -1, OC.HIT_TP, 1)])
def test_entry_outcome_is_the_engine_replay_with_known_answer(move, d, hit, sign):
    mk, b = synthetic(move)
    o = OC.simulate_entry(mk(), b, d, 1.5, 2.0)
    assert o["refused"] is None and o["hit"] == hit and np.sign(o["net_sigma"]) == sign and not o["censored"]
    assert o["duration"] == 3                                   # entry at b + 1, resolved on the move bar b + 4
    assert o["mae"] >= 0 and o["mfe"] >= 0
    if hit == OC.HIT_TP:
        assert o["mfe"] >= 1.5 - 1e-9 and o["net_sigma"] < 1.5  # TP reached; costs make the net smaller than the TP
    else:
        assert o["mae"] >= 2.0 - 1e-9 and o["net_sigma"] < -2.0 + 1e-9
    # the same entry replayed with a fresh engine, step by step, gives the identical net
    f = mk()
    f.step(b, E.LONG if d > 0 else E.SHORT, 1.5, 2.0)
    j = b + 1
    while f.pos is not None:
        f.step(j, E.HOLD)
        j += 1
    t = f.trades[-1]
    assert t["net_usd"] / t["units"] / SIG == pytest.approx(o["net_sigma"], abs=1e-12)


@pytest.mark.parametrize("move, sl_first, rem_sign", [("down", 1, -1), ("up", 0, 1)])
def test_hold_value_known_answer(move, sl_first, rem_sign):
    mk, b = synthetic(move)
    env = mk()
    env.step(b, E.LONG, 1.5, 2.0)
    pos = env.pos
    o = OC.hold_value(mk(), pos, b + 2)                          # decide at b + 2: the move comes at b + 4
    assert o["sl_first"] == sl_first and np.sign(o["remaining"]) == rem_sign and not o["censored"]


def stub_cx(n_rows=10, periods=None):
    idx = pd.date_range("2022-01-01", periods=n_rows, freq="h", tz="UTC")
    periods = periods or {"TRAIN": ("2022-01-01", "2022-01-01 05:00"), "INNER_A": ("2022-01-01 05:00", "2022-01-02")}
    per = np.array(["TRAIN" if i < 5 else "INNER_A" for i in range(n_rows)])
    ds = types.SimpleNamespace(bars=pd.DataFrame(index=idx), meta={"periods": periods}, period=per, row_bar=np.arange(n_rows))
    return types.SimpleNamespace(ds=ds)


def test_label_availability_needs_resolution_before_the_period_end_and_no_censoring():
    cx = stub_cx()
    eb = np.array([2, 3, 4, 6, -1])
    cen = np.array([False, False, False, False, False])
    ok = OC.available(cx, eb, cen, "TRAIN")                      # TRAIN ends at 05:00; bar 4 ends at 05:00 (ok), bar 6 later
    assert ok.tolist() == [True, True, True, False, False]
    assert not OC.available(cx, np.array([2]), np.array([True]), "TRAIN")[0]


def test_entry_sets_keep_only_rows_with_every_cell_available():
    cx = stub_cx()
    n, nc = 10, len(OC.CELLS)
    t = {"eligible": np.ones(n, bool), "net_sigma": np.zeros((n, 2, nc)), "hit": np.zeros((n, 2, nc), np.int8),
         "exit_bar": np.tile(np.arange(n)[:, None, None], (1, 2, nc)), "censored": np.zeros((n, 2, nc), bool)}
    t["exit_bar"][1, 0, 3] = 9                                   # one cell of row 1 resolves after TRAIN's end
    t["net_sigma"][2, 0, 0] = np.nan                             # one cell of row 2 refused
    rows, y, R = OC.entry_sets(cx, t, "TRAIN", 0)
    assert rows.tolist() == [0, 3, 4] and R.shape == (3, nc) and y.tolist() == [1, 1, 1]


# ----------------------------------------------------------------------------- contract and encoding audit
def test_committed_contract_is_the_sealed_default_and_validates():
    p = ROOT / "docs/short_horizon/clean_round/contracts/ACTION_CONTRACT_C2.json"
    doc = json.loads(p.read_text())
    assert doc == C2.default_contract() and C2.validate(doc) == [] and C2.encoding_audit(doc)["verdict"] == "PASS"
    assert doc["strategy"]["require_early_close"] is True and doc["constraints"]["teacher"] == "none"


def _mut(fn):
    d = C2.default_contract()
    fn(d)
    return C2.encoding_audit(d)["errors"]


def test_encoding_audit_refuses_one_continuous_output_cut_into_classes():
    errs = _mut(lambda d: d["specialists"]["open_l"].__setitem__("decision", {"encoding": "tanh_intervals", "bins": [-1, -0.5, 0, 0.5, 1], "n_classes": 5,
                                                                               "classes": list("abcde"), "output_dim": 1, "loss": "mse"}))
    assert any(e.startswith("CATEGORICAL_FROM_CONTINUOUS:open_l") for e in errs) and any(e.startswith("HEAD_DIM_MISMATCH:open_l") for e in errs)


@pytest.mark.parametrize("mut, code", [
    (lambda d: d["strategy"]["value_targets"].__setitem__("open_margin", [0.1, 0.5, 0.9]), "VALUE_TARGET_INTERVALS"),
    (lambda d: d["strategy"]["value_targets"].__setitem__("close_margin", "close_l.value:nonexistent"), "VALUE_TARGET_NOT_EXPECTED_VALUE"),
    (lambda d: d["action_space"]["produced_by"].__setitem__("OPEN_LONG", [["open_l", "value:nothing"]]), "ACTION_UNREACHABLE:OPEN_LONG"),
    (lambda d: d["strategy"].__setitem__("value_rule", "argmax_of_bins"), "VALUE_RULE_UNKNOWN"),
    (lambda d: d["specialists"]["close_l"]["continuous"]["remaining_sigma"].__setitem__("bins", [0, 1]), "VALUE_TARGET_INTERVALS"),
    (lambda d: d["specialists"]["close_l"]["continuous"]["remaining_sigma"].__setitem__("encoding", "softmax"), "CONTINUOUS_THROUGH_SOFTMAX"),
    (lambda d: d["strategy"]["threshold_targets"].__setitem__("close_p", "close_l.remaining_sigma"), "CATEGORICAL_FROM_CONTINUOUS"),
])
def test_encoding_audit_refusals(mut, code):
    assert any(e.startswith(code) for e in _mut(mut)), _mut(mut)


def test_validate_refuses_contract_drift():
    d = C2.default_contract()
    d["strategy"]["require_early_close"] = False
    assert "C2_CONTRACT_IDENTITY" in C2.validate(d) and "C2_CONTRACT_EARLY_CLOSE_RULE" in C2.validate(ACT.seal(d))
    d = C2.default_contract()
    d["constraints"]["teacher"] = "zigzag"
    assert "C2_CONTRACT_CONSTRAINTS" in C2.validate(ACT.seal(d))


# ----------------------------------------------------------------------------- strategy layer
def table(n=3):
    nc = len(OC.CELLS)
    t = {"net": np.full((n, 2, nc), -1.0), "ptp": np.full((n, 2), 0.5), "pn": np.full(n, 0.1), "rem": np.full((n, 2), 0.5), "psl": np.full((n, 2), 0.1)}
    t["net"][0, 1, 7] = 0.8                                      # row 0: short at cell 7 is the best expected net
    t["net"][0, 0, 2] = 0.3
    return t


def test_policy_opens_the_best_side_and_cell_and_respects_the_veto():
    t = table()
    p = C2.PolicyC2(t, C2.StrategyC2(0.2, 0.0, 0.9, False, None))
    a, tp, sl = p(None)(0, 0, np.zeros(9), None)
    assert a == E.SHORT and (tp, sl) == OC.CELLS[7]
    assert p(None)(1, 1, np.zeros(9), None)[0] == E.HOLD          # row 1: nothing above the margin
    t["pn"][0] = 0.95
    assert C2.PolicyC2(t, C2.StrategyC2(0.2, 0.0, 0.9, False, 0.9))(None)(0, 0, np.zeros(9), None)[0] == E.HOLD


def test_policy_closes_by_optimal_stopping_and_reverses_only_when_allowed():
    t = table()
    st_long = np.zeros(9)
    st_long[0] = 1
    s = C2.StrategyC2(0.2, 0.1, 0.6, False, None)
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, None)[0] == E.HOLD          # remaining 0.5 > -0.1, P(SL) 0.1 < 0.6
    t["rem"][1, 0] = -0.3
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, None)[0] == E.CLOSE         # predicted remaining < -close_margin
    t["rem"][1, 0], t["psl"][1, 0] = 0.5, 0.7
    assert C2.PolicyC2(t, s)(None)(1, 1, st_long, None)[0] == E.CLOSE         # calibrated P(SL first) > close_p
    t["psl"][0, 0] = 0.9
    assert C2.PolicyC2(t, s)(None)(0, 0, st_long, None)[0] == E.CLOSE         # reverse not allowed
    s2 = C2.StrategyC2(0.2, 0.1, 0.6, True, None)
    assert C2.PolicyC2(t, s2)(None)(0, 0, st_long, None)[0] == E.SHORT        # close-and-reverse when allowed


def test_restrict_blanks_every_field_outside_the_rows_and_the_policy_refuses_blank_rows():
    t = C2.restrict(table(), [0])
    assert all(np.all(np.isnan(v[1:])) for v in t.values()) and np.all(np.isfinite(t["net"][0]))
    with pytest.raises(ValueError, match="C2_ROW_OUTSIDE_TABLE"):
        C2.PolicyC2(t, C2.StrategyC2(0.2, 0.0, 0.9, False, None))(None)(1, 1, np.zeros(9), None)


def test_calibrate_with_identity_maps_returns_the_probabilities():
    t = table()
    raw = C2.to_raw(t)
    cal = {m: {"a": 1.0, "b": 0.0} for m in C2.SPECIALISTS}
    back = C2.calibrate_table(raw, cal)
    for f in ("ptp", "pn", "psl"):
        assert np.allclose(back[f], t[f])
    assert np.array_equal(back["net"], t["net"])


def comps(sharpe, dd, opw, closes):
    return {"sharpe": sharpe, "se_lo": 0.5, "lcb": sharpe - 0.5, "dd": dd, "opw": opw, "opd": opw / 5, "r_ann": 0.05 * sharpe, "trade_t": sharpe,
            "trade_mean_sigma": 0.1, "trade_tstat_reused": sharpe, "calmar": 1.0, "return_minus_dd": 0.01, "trades": 50, "days": 180,
            "weeks_below_2": 0, "weeks_below_2_share": 0.0, "agent_closes": closes, "valid": opw >= 2.0, "violation": max(0.0, 2.0 - opw), "reasons": []}


def test_strategy_optimiser_cannot_read_outside_the_calibration_rows():
    AC = C2.default_contract()
    AC["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 2}
    cal_rows = np.array([0, 1])

    def score(view, s):                                          # reads the view it is given, nothing else
        x = float(np.nansum(view["net"]) + np.nansum(view["psl"]))
        return comps(1.0 + 0.01 * x - 0.1 * s.open_margin, 5.0 + s.close_p, 3.0, 2)

    t = table(4)
    poisoned = copy.deepcopy(t)
    poisoned["net"][2:] = 50.0
    poisoned["psl"][2:] = 0.99
    a = C2.freeze(t, cal_rows, score, AC, 1, "final", "sharpe_lcb")
    b = C2.freeze(poisoned, cal_rows, score, AC, 1, "final", "sharpe_lcb")
    assert a["strategy"] == b["strategy"] and a["chosen"]["sharpe"] == b["chosen"]["sharpe"]
    assert [p["sharpe"] for p in a["runs"]["without_veto"]["points"]] == [p["sharpe"] for p in b["runs"]["without_veto"]["points"]]


def test_strategy_without_early_close_is_not_selectable():
    AC = C2.default_contract()
    AC["strategy"]["optimizer"]["final"] = {"population": 4, "generations": 1}
    fr = C2.freeze(table(2), np.array([0, 1]), lambda v, s: comps(2.0, 3.0, 3.0, 0), AC, 1, "final", "sharpe_lcb")
    assert fr["valid"] is False and "DEGENERATE_NO_EARLY_CLOSE" in fr["note"]
    assert fr["runs"]["without_veto"]["points"]                   # archived, never discarded


# ----------------------------------------------------------------------------- E3-M guard, launcher, report rules
def test_e3m_refuses_outcomes_without_a_c2_contract(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M
    a = M.build_parser().parse_args(["run", "--out", str(tmp_path / "o"), "--outcomes", str(tmp_path / "x.npz"), "--admitted", "a", "--csv-dir", "c"])
    with pytest.raises(SystemExit, match="OUTCOMES_REQUIRE_C2_ACTION_CONTRACT"):
        M.run(a)


def wf_c2_spec():
    p = ROOT / "docs/short_horizon/clean_round/contracts/ACTION_CONTRACT_C2.json"
    doc = json.loads(p.read_text())
    return {"stages": {"A": [{"name": "zz_hold"}]}, "walk_forward": {"folds": {"wf2022": {}, "wf2023": {}}}, "walk_forward_v1": {"folds": ["wf2022", "wf2023"]},
            "walk_forward_c2": {"action_contract": {"path": str(p), "sha256": ACT.sha256_file(p), "identity": doc["identity"]},
                                "arms": [{"name": "zz_feat_c2", "base": "zz_hold", "strategy_fitness": "sharpe_lcb"}], "folds": ["wf2022", "wf2023"],
                                "seeds": "1,2", "baseline": {"zz_feat_c2": {"wf": "WF_<fold>_zz_hold", "wf_v1": "WF_V1_<fold>_zz_hold_v1"}},
                                "fitness_variants": list(ACT.FITNESS), "rule": "r", "early_close_rule": "e", "fitness_rule": "f",
                                "outcomes": {"file_pattern": "OUTCOMES_ENTRY_<fold>.npz"}}}


def test_launcher_wf_c2_block():
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = wf_c2_spec()
    assert C.walk_forward_c2_errors(s) == []
    for mut, code in ((lambda s: s["walk_forward_c2"]["arms"][0].__setitem__("labels", "multires"), "WF_C2_ARM_HAS_TEACHER_TREATMENT"),
                      (lambda s: s["walk_forward_c2"].__setitem__("early_close_rule", ""), "WF_C2_EARLY_CLOSE_RULE_NOT_PREREGISTERED"),
                      (lambda s: s["walk_forward_c2"]["action_contract"].__setitem__("sha256", "0" * 64), "WF_C2_CONTRACT_DIGEST_MISMATCH"),
                      (lambda s: s["walk_forward_c2"]["baseline"].__setitem__("zz_feat_c2", {"wf": "x"}), "WF_C2_NO_BASELINE"),
                      (lambda s: s["walk_forward_c2"].__setitem__("folds", ["consolidated"]), "WF_C2_FOLD_REFUSED")):
        s2 = copy.deepcopy(s)
        mut(s2)
        assert any(e.startswith(code) for e in C.walk_forward_c2_errors(s2)), (code, C.walk_forward_c2_errors(s2))


def test_report_rules():
    from tools import expert_guided_c2 as R
    c = {"wf2022": {"value_usd": 10.0, "recall": 0.30}, "wf2023": {"value_usd": 5.0, "recall": 0.25}}
    b = {"wf2022": {"value_usd": -800.0, "recall": 0.06}, "wf2023": {"value_usd": -900.0, "recall": 0.11}}
    assert R.early_close_verdict(c, b)["verdict"] == "FIXES_EARLY_CLOSE"
    c["wf2023"]["recall"] = 0.21                                  # < 2 x 0.11
    assert R.early_close_verdict(c, b)["verdict"] == "DOES_NOT_FIX_EARLY_CLOSE"
    assert R.early_close_verdict({"wf2022": c["wf2022"]}, b)["verdict"] == "NOT_MEASURED"
    w = lambda s, se, v: {"sharpe": s, "se_lo": se, "valid": v}  # noqa: E731
    fc = R.fitness_choice({"wf2022": {"a": w(1.5, 1.0, True), "b": w(3.0, 1.0, False), "c": w(1.2, 0.2, True)},
                           "wf2023": {"c": w(1.1, 0.5, True)}})
    assert fc["chosen"] == "c" and fc["verdict"] == "CONFIRMED"
    fc = R.fitness_choice({"wf2022": {"c": w(1.2, 0.2, True)}, "wf2023": {"c": w(0.3, 0.5, True)}})
    assert fc["verdict"] == "REFUTED"
    assert R.fitness_choice({"wf2022": {"c": w(1.2, 0.2, True)}, "wf2023": {}})["verdict"] == "NOT_MEASURED"
