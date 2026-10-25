"""Lane L6 (2026-10-25): seed extension (--seeds, pooled decide), --aux-targets multires (fitness regression heads on
inherited class labels), launcher emit of named arms with new seeds into suffixed outputs, spec localization for the
staged package, and the stale sample-weights text. Synthetic; no data, no fit."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from app.short_horizon.expert_guided import multires as MR  # noqa: E402
from tests.test_expert_guided_multires import tables  # noqa: E402
from tests.test_expert_guided_clean_round import spec  # noqa: E402,F401  (fixture)
from tools import expert_guided_clean_round as C  # noqa: E402
from tools import expert_guided_e3m as M  # noqa: E402


def test_parse_seeds_default_and_refusals():
    assert M.parse_seeds(",".join(map(str, M.SEEDS_DEFAULT))) == M.SEEDS_DEFAULT == (20261019, 20261020, 20261021)
    assert M.parse_seeds("20261022,20261023") == (20261022, 20261023)
    for bad in ("", "1,1", "0", "x"):
        with pytest.raises(SystemExit, match="E3M_SEEDS_MALFORMED"):
            M.parse_seeds(bad)
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert a.seeds == "20261019,20261020,20261021" and a.aux_targets is None
    a2 = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--aux-targets"])
    a3 = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--aux-targets", "multires"])
    assert a2.aux_targets == "opportunity" and a3.aux_targets == "multires" and bool(a2.aux_targets) and bool(a3.aux_targets)


class _Aux(MR.MultiResAux):
    def __init__(self, tb):
        self.tb = tb


def test_multires_aux_heads_are_the_bounded_fitness_and_masked_when_unavailable():
    aux = _Aux(tables())                                       # period ends at bar 11: everything confirmed
    f, ok = aux.opportunity(2, 2, 1)                           # optimal long open at the valley: fitness 1
    assert ok == 1.0 and 0.9 < f <= 1.0                         # fill at open(b+1), just above the pivot level
    fs, oks = aux.opportunity(2, 2, -1)                        # the short side against the up leg: negative
    assert oks == 1.0 and fs < 0 and fs >= MR.F_OPEN_BOUND[0]
    fl, okl = aux.opportunity(4, 4, 1)                         # late entry: lower but positive
    assert okl == 1.0 and 0 < fl < 1
    hc, okc = aux.hold_gain(6, 6, 1)                           # long at the peak bar: f_close near 1 (close vs intrabar level)
    assert okc == 1.0 and 0.8 < hc <= MR.F_CLOSE_BOUND[1]
    early = _Aux(tables(period_end_bar=3))                     # nothing confirmed inside the period: masked, never filled
    assert early.opportunity(2, 2, 1) == (0.0, 0.0) and early.hold_gain(6, 6, 1) == (0.0, 0.0)
    d = aux.describe()
    assert d["mode"] == "multires" and "f_open" in d["opener_head"] and "f_close" in d["closer_head"]


def test_commands_carry_the_aux_mode_and_unknown_modes_are_refused(spec):  # noqa: F811
    s, _ = spec
    s = copy.deepcopy(s)
    s["stages"]["B"].append({"name": "winner_multires_aux", "aux_targets": "multires", "agent_state": "none"})
    assert C.validate(s) == []
    cmd = " ".join(C.e3m_cmd(s, s["stages"]["B"][-1], "OUT", "ROOT"))
    assert "--aux-targets multires" in cmd and "--labels zigzag" in cmd
    legacy = " ".join(C.e3m_cmd(s, s["stages"]["B"][0], "OUT", "ROOT"))
    assert legacy.endswith("--aux-targets") or "--aux-targets --" in legacy or "--aux-targets " in legacy
    assert "--aux-targets multires" not in legacy
    s2 = copy.deepcopy(s)
    s2["stages"]["B"][-1]["aux_targets"] = "fitness"
    assert any(e.startswith("AUX_TARGETS_UNKNOWN") for e in C.validate(s2))


def test_emit_named_arms_with_new_seeds_into_suffixed_units(spec, tmp_path):  # noqa: F811
    s, t = spec
    (t / "LAUNCH_EXPERT_GUIDED.json").write_text("{}")
    sp = t / "SPEC.json"
    sp.write_text(json.dumps(s))
    mk = lambda **kw: type("A", (), dict({"spec": str(sp), "stage": "A", "winner_from": None, "out": str(t / "q.sh"), "arms": None, "seeds": None, "out_suffix": None}, **kw))()  # noqa: E731
    C.cmd_emit(mk(arms="zz,opp_hold", seeds="20261022,20261023", out_suffix="_seeds2"))
    q = (t / "q.sh").read_text()
    assert q.count("expert_guided_e3m.py run") == 2 and "--seeds 20261022,20261023" in q
    assert "$Q/out/A_zz_seeds2" in q and "STATE/A_zz_seeds2" in q and "clean-A_zz_seeds2" in q and "both_experts" not in q
    with pytest.raises(SystemExit, match="SEEDS_REQUIRE_OUT_SUFFIX"):
        C.cmd_emit(mk(arms="zz", seeds="20261022"))
    with pytest.raises(SystemExit, match="ARM_UNKNOWN"):
        C.cmd_emit(mk(arms="nope"))
    C.cmd_emit(mk())                                           # the plain emission is unchanged
    assert (t / "q.sh").read_text().count("expert_guided_e3m.py run") == 3 and "--seeds" not in (t / "q.sh").read_text()


def _e3m(seeds, inner, train=3.0, valid=True, support="s1", recipe="r1"):
    tr = {str(s): {"trading": {"TRAIN": {"sharpe_daily_sqrt365": train, "valid": True},
                               "INNER_A": {"sharpe_daily_sqrt365": x, "valid": valid}}} for s, x in zip(seeds, inner)}
    return {"trials": {"FULL": tr}, "params": {"FULL": {"open_l": 1, "open_s": 1, "close_l": 1, "close_s": 1}},
            "decision": {"means": {"FULL": {"inner_a_sharpe_mean": float(np.mean(inner)), "all_valid": valid}}},
            "meta": {"support_hash": {"TRAIN": support, "INNER_A": support}}, "recipe": {"identity": recipe}}


def test_decide_pools_seed_extensions_restricts_arms_and_refuses_mismatches(spec, tmp_path):  # noqa: F811
    s, t = spec
    sp = t / "SPEC.json"
    sp.write_text(json.dumps(s))
    res = t / "res"
    for arm, inner, inner2 in (("zz", [1.0, 1.2, 1.1], [0.2, 0.3]), ("opp_hold", [1.3, 0.9, 1.0], [1.4, 1.5]), ("both_experts", [2.0, 2.0, 2.0], None)):
        (res / f"A_{arm}").mkdir(parents=True)
        (res / f"A_{arm}" / "E3M.json").write_text(json.dumps(_e3m([1, 2, 3], inner)))
        if inner2:
            (res / f"A_{arm}_seeds2").mkdir()
            (res / f"A_{arm}_seeds2" / "E3M.json").write_text(json.dumps(_e3m([4, 5], inner2)))
    mk = lambda **kw: type("A", (), dict({"spec": str(sp), "stage": "A", "results": str(res), "out": str(t / "D.json"), "arms": None, "pool_suffix": None, "tie_control": None}, **kw))()  # noqa: E731
    C.cmd_decide(mk())                                         # unchanged behaviour on the three seeds: both_experts wins
    d = json.loads((t / "D.json").read_text())
    assert d["chosen"] == "both_experts" and d["arms"][0]["n_seeds"] == 3 and d["arms"][0]["inner_a_sharpe_mean"] == pytest.approx(1.1)
    C.cmd_decide(mk(arms="zz,opp_hold", pool_suffix="_seeds2", tie_control="zz"))
    d = json.loads((t / "D.json").read_text())
    zz = next(r for r in d["arms"] if r["arm"] == "zz")
    assert d["chosen"] == "opp_hold" and zz["n_seeds"] == 5 and zz["seeds"] == ["1", "2", "3", "4", "5"]
    assert zz["inner_a_sharpe_mean"] == pytest.approx(np.mean([1.0, 1.2, 1.1, 0.2, 0.3])) and d["restricted_to"] == ["zz", "opp_hold"] and zz["is_control"]
    (res / "A_zz_seeds2" / "E3M.json").write_text(json.dumps(_e3m([4, 5], [0.2, 0.3], support="other")))
    with pytest.raises(SystemExit, match="POOL_MISMATCH"):
        C.cmd_decide(mk(arms="zz,opp_hold", pool_suffix="_seeds2"))
    (res / "A_zz_seeds2" / "E3M.json").write_text(json.dumps(_e3m([3, 5], [0.2, 0.3])))
    with pytest.raises(SystemExit, match="POOL_DUPLICATE_SEEDS"):
        C.cmd_decide(mk(arms="zz,opp_hold", pool_suffix="_seeds2"))
    (res / "A_zz_seeds2" / "E3M.json").write_text(json.dumps(_e3m([4, 5], [0.2, 0.3])))
    with pytest.raises(SystemExit, match="ARM_NOT_FINISHED:both_experts_seeds2"):
        C.cmd_decide(mk(pool_suffix="_seeds2"))


def test_localize_rewrites_paths_only_and_keeps_digests(spec):  # noqa: F811
    s, _ = spec
    s = copy.deepcopy(s)
    s["stages"]["A"][0]["multires_thresholds"] = "/somewhere/CHOSEN_THRESHOLDS.json"
    s["stages"]["A"][0]["labels"] = "multires"
    s["stages"]["A"][0].pop("closer_rule", None)
    L = C.localize(s, "/pkg")
    assert L["recipe"]["path"] == "/pkg/recipe/CONSUMER_RECIPE.json" and L["recipe"]["sha256"] == s["recipe"]["sha256"]
    assert L["store"]["path"] == "/pkg/data/store/store.parquet" and L["store"]["sha256"] == s["store"]["sha256"]
    assert L["demonstrators"]["file"] == "/pkg/in/DEMONSTRATORS.json" and L["gates"][0]["path"] == "/pkg/LAUNCH_EXPERT_GUIDED.json"
    assert L["stages"]["A"][0]["multires_thresholds"] == "/pkg/recipe/CHOSEN_THRESHOLDS.json" and L["remote_root"] == "/pkg"
    assert s["recipe"]["path"] != L["recipe"]["path"]           # the input is not mutated
    assert any(e in ("RECIPE_FILE_MISSING",) for e in C.validate(L))   # the package does not exist here: refused, not trusted
