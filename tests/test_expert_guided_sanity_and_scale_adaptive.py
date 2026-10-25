"""Scale-adaptive strategy genes and the sanity gate (2026-10-25, after WF_V1's zero trades: absolute gene bounds were
unreachable for small calibrated probabilities). The gene map's kinds with known answers; a synthetic table with tiny
calibrated probabilities reaches the activity rule under v1.2 while the absolute v1.1 space cannot; an infeasible
space is refused (STRATEGY_SPACE_INFEASIBLE) instead of freezing a zero-trade strategy; the v1.2 contract; every
sanity cause; the refusal of verdicts involving a FAILED_SANITY unit; the launcher's WF_V12 block. No data, no fit."""
import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest

from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import stats as ST

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs/short_horizon/clean_round/contracts"


# ----------------------------------------------------------------------------- gene map
def test_gene_map_kinds_known_answers():
    gm = ACT.GeneMap({"a": np.arange(101, dtype=float), "b": np.linspace(-1, 1, 201)},
                     {"x": ("rate", "a", "upper_rate"), "y": ("u", "a", "quantile"), "z": ("r", "b", "neg_lower_rate"), "w": ("flag", None, "copy")})
    out = gm.to_absolute({"rate": 0.10, "u": 0.25, "r": 0.5, "flag": 1.0})
    assert out == pytest.approx({"x": 90.0, "y": 25.0, "z": 0.0, "w": 1.0})
    assert gm.to_absolute({"rate": 0.0})["x"] == 100.0                     # rate 0 -> the max: a strict '>' never fires
    d = gm.describe()
    assert d["references"]["a"]["rows"] == 101 and len(d["references"]["a"]["quantiles"]) == len(d["grid"]) == 101
    with pytest.raises(SystemExit, match="STRATEGY_SPACE_INFEASIBLE"):
        ACT.GeneMap({"a": np.array([np.nan])}, {"x": ("rate", "a", "upper_rate")})


# ----------------------------------------------------------------------------- tiny probabilities: v1.2 reaches the rule, v1.1 cannot
N = 400


def tiny_table():
    rng = np.random.default_rng(0)
    t = {f: np.full(N, np.nan) for f in ACT.FIELDS}
    t["pL"], t["pS"] = rng.uniform(1e-3, 3e-2, N), rng.uniform(1e-3, 3e-2, N)
    t["tpL"], t["tpS"], t["slL"], t["slS"] = (np.full(N, 1.5), np.full(N, 1.5), np.full(N, 2.0), np.full(N, 2.0))
    t["cost"] = np.full(N, 0.2)
    t["pN"] = rng.uniform(0.6, 0.99, N)
    t["pCL"], t["pCS"] = rng.uniform(0.01, 0.2, N), rng.uniform(0.01, 0.2, N)
    return t


def score_from_view(view, s):
    """A deterministic stand-in for the engine replay on hourly rows (120 rows = one trading week; the replay contract's
    1 entry/day caps the activity at 5/week): every row where the frozen policy would open is an entry signal, every row
    where a closer would fire is an agent close."""
    pol = ACT.ContractPolicy(view, s)
    rows = np.flatnonzero(np.isfinite(view["pL"]))
    opens = sum(1 for r in rows if pol.open_decision(int(r)) is not None)
    closes = int(np.sum(np.fmax(view["pCL"][rows], view["pCS"][rows]) > s.close_threshold)) if opens else 0
    opw = min(opens / (len(rows) / 120.0), 5.0)
    trades = int(min(opens, 5.0 * len(rows) / 120.0))
    sh = 1.0 + 0.01 * opens if trades else float("nan")
    return {"sharpe": sh, "se_lo": 0.5, "lcb": sh - 0.5 if trades else float("nan"), "dd": 5.0, "opw": opw, "opd": opw / 5, "r_ann": 0.05,
            "trade_t": 1.0, "trade_mean_sigma": 0.1, "trade_tstat_reused": 1.0 if trades >= 15 else float("nan"), "calmar": 1.0, "return_minus_dd": 0.01,
            "trades": trades, "days": 80, "weeks_below_2": 0, "weeks_below_2_share": 0.0, "agent_closes": closes,
            "valid": bool(trades and opw >= 2.0), "violation": max(0.0, 2.0 - opw), "reasons": [] if trades and opw >= 2.0 else ["below_2"]}


def small(ac):
    ac = copy.deepcopy(ac)
    ac["strategy"]["optimizer"]["final"] = {"population": 8, "generations": 3}
    return ac


def test_tiny_probabilities_reach_the_activity_rule_under_v12_but_not_under_v11():
    v11 = small(json.loads((CONTRACTS / "ACTION_CONTRACT_v1_1.json").read_text()))
    v12 = small(json.loads((CONTRACTS / "ACTION_CONTRACT_v1_2.json").read_text()))
    t, rows = tiny_table(), np.arange(N)
    a = ACT.freeze_strategy(t, rows, score_from_view, v11, 1, "final", "sharpe_lcb")
    assert a["valid"] is False and a["runs"]["without_veto"]["share_valid"] == 0.0      # the absolute space is unreachable (WF_V1's failure)
    b = ACT.freeze_strategy(t, rows, score_from_view, v12, 1, "final", "sharpe_lcb")
    assert b["valid"] is True and b["chosen"]["opw"] >= 2.0 and b["chosen"]["agent_closes"] >= 1
    assert b["feasibility_probe"]["verdict"] == "FEASIBLE" and b["gene_space"] == ACT.GENE_SPACE_Q
    assert b["strategy"]["open_threshold"] < 0.05                           # mapped below v1.1's absolute lower bound
    assert set(b["gene_map"]["rules"]) == {"open_threshold", "gain_margin", "close_threshold", "allow_reverse", "nop_veto"}


def test_infeasible_space_is_refused_and_never_freezes_a_zero_trade_strategy():
    v12 = small(json.loads((CONTRACTS / "ACTION_CONTRACT_v1_2.json").read_text()))
    t, rows = tiny_table(), np.arange(N)
    dead = lambda view, s: dict(score_from_view(view, s), trades=0, opw=0.0, agent_closes=0, valid=False)  # noqa: E731
    with pytest.raises(SystemExit, match="STRATEGY_SPACE_INFEASIBLE"):
        ACT.freeze_strategy(t, rows, dead, v12, 1, "final", "sharpe_lcb")
    noclose = lambda view, s: dict(score_from_view(view, s), agent_closes=0)  # noqa: E731
    with pytest.raises(SystemExit, match="STRATEGY_SPACE_INFEASIBLE"):
        ACT.freeze_strategy(t, rows, noclose, v12, 1, "final", "sharpe_lcb")
    v = ACT.freeze_strategy(t, rows, dead, v12, 1, "visitation", "sharpe_lcb")
    assert v["valid"] is False and v["strategy"] is None and v["feasibility_probe"]["feasible_points"] == 0


def test_probe_covers_every_corner_and_the_q_max_point():
    pts = ACT.probe_points(ACT.Q_BOUNDS_V1)
    assert len(pts) == 2 ** 3 + 1
    assert {(p["open_rate"], p["gain_quantile"], p["close_rate"]) for p in pts[:8]} == {(a, b, c) for a in (0.002, 0.30) for b in (0.0, 0.95) for c in (0.0, 0.60)}
    assert pts[-1] == {"open_rate": 0.30, "gain_quantile": 0.475, "close_rate": 0.30, "allow_reverse": 0.0}
    assert all("nop_veto_rate" not in p for p in pts)


def test_v12_contract_validates_and_refuses_drift():
    d = json.loads((CONTRACTS / "ACTION_CONTRACT_v1_2.json").read_text())
    assert ACT.validate(d) == [] and d["strategy"]["gene_space"] == ACT.GENE_SPACE_Q and d["strategy"]["require_early_close"] is True
    for mut, code in ((lambda d: d["strategy"]["genes"].__setitem__("open_rate", [0.05, 0.95]), "ACTION_CONTRACT_GENES"),
                      (lambda d: d["strategy"].__setitem__("require_early_close", False), "ACTION_CONTRACT_EARLY_CLOSE_RULE"),
                      (lambda d: d["strategy"].__setitem__("gene_space", "absolute_v2"), "ACTION_CONTRACT_GENE_SPACE")):
        x = copy.deepcopy(d)
        mut(x)
        x["identity"] = None
        assert any(e.startswith(code) for e in ACT.validate(ACT.seal(x))), code


# ----------------------------------------------------------------------------- sanity gate
def summ(trades=60, opw=3.0, sharpe=1.0, tp=10, sl=10, ff=5, agent=30, rev=5, long=30, short=30):
    return {"trades": trades, "orders_per_week": opw, "sharpe_daily_sqrt365": sharpe, "n_close_TP": tp, "n_close_SL": sl, "n_close_FORCED_FLAT": ff,
            "n_close_AGENT_CLOSE": agent, "n_close_REVERSAL": rev, "n_entries_long": long, "n_entries_short": short}


def write_unit(tmp, name, seeds: dict, ensemble=None, share_valid=0.4):
    d = tmp / name
    d.mkdir(parents=True)
    tr = {s: {"trading": per, "action_contract": {"strategy": {"runs": {"without_veto": {"share_valid": share_valid}, "with_veto": {"share_valid": share_valid}}}}}
          for s, per in seeds.items()}
    (d / "E3M.json").write_text(json.dumps({"trials": {"FULL": tr}}))
    if ensemble is not None:
        (d / "ENSEMBLE.json").write_text(json.dumps(ensemble))
    return d


@pytest.mark.parametrize("kw, cause", [
    ({"trades": 0, "opw": 0.0, "sharpe": float("nan"), "tp": 0, "sl": 0, "ff": 0, "agent": 0, "rev": 0, "long": 0, "short": 0}, "zero_trades"),
    ({"opw": 1.9}, "below_2_trades_per_week"),
    ({"long": 56, "short": 4}, "one_side_over_90pct"),
    ({"agent": 0, "rev": 0}, "zero_agent_closes"),
    ({"tp": 0, "sl": 0, "ff": 0, "rev": 0, "agent": 60}, "single_exit_reason"),
    ({"sharpe": float("nan")}, "sharpe_undefined"),
])
def test_each_sanity_cause_fails_the_unit(tmp_path, kw, cause):
    from tools import expert_guided_sanity_gate as SG
    d = write_unit(tmp_path, "U", {"1": {"INNER_A": summ(), "INNER_B": summ(**kw)}, "2": {"INNER_A": summ(), "INNER_B": summ()}})
    g = SG.gate_unit(d)
    assert g["status"] == "FAILED_SANITY" and any(c["cause"] == cause and c["block"] == "INNER_B" and c["where"] == "seed 1" for c in g["causes"]), g["causes"]


def test_clean_unit_passes_and_share_valid_replay_and_failed_units_fail(tmp_path):
    from tools import expert_guided_sanity_gate as SG
    ok = {"INNER_A": summ(), "INNER_B": summ()}
    ens = {"ensemble": {"INNER_A": {"metrics": summ()}, "INNER_B": {"metrics": summ()}}, "ensemble_strategy": {"runs": {"a": {"share_valid": 0.3}}},
           "verification": {"per_seed_replay_equals_run": {"1/INNER_A": True}}}
    assert SG.gate_unit(write_unit(tmp_path, "OK", {"1": ok}, ens))["status"] == "PASS"
    g = SG.gate_unit(write_unit(tmp_path, "SV", {"1": ok}, ens, share_valid=0.0))
    assert [c["cause"] for c in g["causes"]] == ["strategy_share_valid_zero"]
    bad = copy.deepcopy(ens)
    bad["verification"]["per_seed_replay_equals_run"]["1/INNER_B"] = False
    g = SG.gate_unit(write_unit(tmp_path, "RM", {"1": ok}, bad))
    assert [c["cause"] for c in g["causes"]] == ["replay_mismatch"]
    bad = copy.deepcopy(ens)
    bad["ensemble"]["INNER_B"]["metrics"] = summ(agent=0, rev=0)
    g = SG.gate_unit(write_unit(tmp_path, "EN", {"1": ok}, bad))
    assert g["causes"] == [{"cause": "zero_agent_closes", "where": "ensemble", "block": "INNER_B"}]
    (tmp_path / "GONE.log").write_text('{"seed": 1}\nSystemExit: STRATEGY_SPACE_INFEASIBLE:{"verdict": "STRATEGY_SPACE_INFEASIBLE"}\n')
    (tmp_path / "GONE").mkdir()
    g = SG.gate_unit(tmp_path / "GONE")
    assert g["status"] == "FAILED_SANITY" and g["causes"][0]["cause"] == "unit_failed" and "STRATEGY_SPACE_INFEASIBLE" in g["causes"][0]["detail"]


def test_side_balance_unmeasured_is_reported_not_assumed(tmp_path):
    from tools import expert_guided_sanity_gate as SG
    m = summ()
    m.pop("n_entries_long"), m.pop("n_entries_short")
    g = SG.gate_unit(write_unit(tmp_path, "OLD", {"1": {"INNER_A": m}}))
    assert g["status"] == "PASS" and any(u.endswith("side_balance") for u in g["unmeasured"])
    bal = {"long": {"entries": 58}, "short": {"entries": 2}}
    assert "one_side_over_90pct" in SG.check_window(m, bal)["causes"]


def test_status_of_and_verdicts_refuse_failed_sanity_units(tmp_path):
    from tools import expert_guided_sanity_gate as SG
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "SANITY.json").write_text(json.dumps({"status": "FAILED_SANITY", "cause_counts": {"zero_trades": 2}, "causes": []}))
    assert SG.status_of(tmp_path, "A") == {"sanity": "FAILED_SANITY", "sanity_causes": {"zero_trades": 2}}
    assert SG.status_of(tmp_path, "B")["sanity"] == "NOT_GATED"
    good = {"sharpe": 1.5, "se_lo": 0.5}
    bad = {"sharpe": 2.0, "se_lo": 0.5, "unit": "A", "sanity": "FAILED_SANITY"}
    assert ST.holds({"f1": good, "f2": good})["verdict"] == "HOLDS"
    h = ST.holds({"f1": good, "f2": bad})
    assert h["verdict"] == ST.SANITY_REFUSAL and h["failed_units"] == ["A"]
    assert ST.separable(good, bad)["verdict"] == ST.SANITY_REFUSAL and ST.separable(good, good)["verdict"] == "NOT_SEPARABLE"


def test_c2g_report_refuses_verdicts_on_failed_sanity_units(tmp_path):
    from tools import expert_guided_c2 as R
    ens = {"ensemble": {"INNER_B": {"metrics": {"orders_per_week": 3.0, "max_drawdown_pct": 5.0, "valid": True, "trades": 60}, "sharpe": 2.0, "se_lo": 0.5}},
           "by_fitness": {}}
    for u in ("WF_C2G_wf2022_g", "WF_C2G_wf2023_g", "WF_C2_wf2022_c", "WF_C2_wf2023_c", "WF_wf2022_b", "WF_wf2023_b"):
        (tmp_path / u).mkdir()
        (tmp_path / u / "ENSEMBLE.json").write_text(json.dumps(ens))
    (tmp_path / "WF_C2G_wf2023_g" / "SANITY.json").write_text(json.dumps({"status": "FAILED_SANITY", "cause_counts": {"zero_agent_closes": 1}, "causes": []}))
    spec = {"identity": "t", "walk_forward_c2g": {"action_contract": {}, "rule": "r", "early_close_rule": "e", "fitness_rule": "f", "folds": ["wf2022", "wf2023"],
                                                  "arms": [{"name": "g"}], "baseline": {"g": {"c2": "WF_C2_<fold>_c", "wf": "WF_<fold>_b"}}}}
    rec = R.build_report_c2g(spec, tmp_path, tmp_path)["arms"]["g"]
    assert rec["holds"]["verdict"] == ST.SANITY_REFUSAL and rec["folds"]["wf2023"]["separable_vs_c2"]["verdict"] == ST.SANITY_REFUSAL
    assert rec["folds"]["wf2022"]["separable_vs_c2"]["verdict"] == "NOT_SEPARABLE"
    assert rec["early_close"]["verdict"] == ST.SANITY_REFUSAL and rec["fitness_choice"]["verdict"] == ST.SANITY_REFUSAL
    assert rec["sanity"]["WF_C2G_wf2023_g"]["sanity"] == "FAILED_SANITY"


# ----------------------------------------------------------------------------- launcher
def test_spec_v9_preregisters_wf_v12_and_supersedes_wf_v1(tmp_path):
    pytest.importorskip("torch")
    from tools import expert_guided_clean_round as C
    s = json.loads((ROOT / "docs/short_horizon/clean_round/CLEAN_ROUND_SPEC.json").read_text())
    assert s["walk_forward_v1"]["status"] == "SUPERSEDED_INFEASIBLE_STRATEGY_SPACE"
    w = s["walk_forward_v12"]
    assert [a["name"] for a in w["arms"]] == ["zz_hold_v12", "both_hold_v12"] and w["action_contract"]["sha256"] == ACT.sha256_file(CONTRACTS / "ACTION_CONTRACT_v1_2.json")
    assert C.walk_forward_v12_errors(s) == [] and "sanity" in s["sanity_gate"]["rule"].lower()
    x = copy.deepcopy(s)
    x["walk_forward_v12"]["action_contract"] = dict(s["walk_forward_v1"]["action_contract"])
    assert "WF_V12_CONTRACT_NOT_SCALE_ADAPTIVE" in C.walk_forward_v12_errors(x)
    e = copy.deepcopy(s)
    e["remote_root"] = "/r"
    out = tmp_path / "q.sh"
    C.emit_walk_forward_v1(e, types.SimpleNamespace(out=str(out)), "walk_forward_v12", C.WF_V12_STAGE)
    q = out.read_text()
    assert q.count("if [ ! -f $Q/STATE/WF_V12_wf20") == 4 and "ACTION_CONTRACT_v1_2.json" in q and "[WF_V12]" in q and 'log "STAGE_WF_V12_DONE"' in q and "WF_V1_" not in q
