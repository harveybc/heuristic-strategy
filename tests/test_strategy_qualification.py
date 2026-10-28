"""QUALIFICATION_SPEC v2 verdict helpers (Musashi GM13 / GM18, coordinator order 2026-10-28): negative-control verdicts
against the independent construction, seeded-panel verdicts with every declared stratum and explicit NOT_EVALUABLE,
confirmation with manifest completeness and paired identity. No engine, no data."""
import json
import math

import numpy as np

from tools import strategy_qualification as Q


def test_control_verdict_is_within_delta_of_the_construction_and_never_passes_on_nan():
    assert Q.control_delta(12.0) == 3.0 and Q.control_delta(2.0) == 1.0 and math.isnan(Q.control_delta(float("nan")))
    assert Q.control_verdict(-9.0, -10.0, 3.0) == "PASS" and Q.control_verdict(2.0, -10.0, 3.0) == "FAIL"
    assert Q.control_verdict(float("nan"), -10.0, 3.0) == "NOT_EVALUABLE" and Q.control_verdict(1.0, 1.0, float("nan")) == "NOT_EVALUABLE"


def test_seeded_verdict_lists_every_stratum_and_marks_empty_or_thin_ones_not_evaluable():
    summ = {"BUY|age0|loss": {"n": 40, "sl_destined": 35, "sl_destined_closed_before_sl": 0.97, "normalized_regret_mean": 0.05, "normalized_regret_p90": 0.2},
            "BUY|age0|profit": {"n": 10, "sl_destined": 3, "sl_destined_closed_before_sl": 0.0, "normalized_regret_mean": 0.05, "normalized_regret_p90": 0.2},
            "SELL|age6|loss": {"n": 40, "sl_destined": 35, "sl_destined_closed_before_sl": 0.5, "normalized_regret_mean": 0.05, "normalized_regret_p90": 0.2}}
    v = Q.seeded_verdict(summ, {"n": 0}, [0, 6])
    assert len(v) == 2 * 2 * 3 + 1                                   # every declared stratum, never omitted
    assert v["BUY|age0|loss"]["overall"] == "PASS"
    assert v["BUY|age0|profit"]["overall"] == "NOT_EVALUABLE" and v["BUY|age0|profit"]["sl_destined_rate"] == "NOT_EVALUABLE"   # < 30 SL-destined: never PASS
    assert v["SELL|age6|loss"]["overall"] == "FAIL"
    assert v["SELL|age0|loss"]["overall"] == "NOT_EVALUABLE" and v["SELL|age0|loss"]["reason"] == "empty stratum"
    assert v["reversal_then_recovery"]["overall"] == "NOT_EVALUABLE"
    v2 = Q.seeded_verdict(summ, {"n": 12, "premature_regret_norm_mean": 0.05, "early_share": 0.3, "Ostar_early_share": 0.3}, [0, 6])
    assert v2["reversal_then_recovery"]["overall"] == "PASS"


def _recs(ids, seeds, sharpe=10.0, code="c1", spec="s1"):
    return [{"id": i, "seed": s, "sharpe": sharpe, "net_usd": 100.0, "code_sha": code, "spec_sha": spec} for i in ids for s in seeds]


def test_confirm_panel_requires_completeness_and_paired_identity_before_any_claim():
    ids = ["A", "B", "C"]
    a, b = _recs(ids, [0, 1, 2]), _recs(ids, [3, 4, 5], sharpe=11.0)
    ok = Q.confirm_panel(a, b, ids, [0, 1, 2], [3, 4, 5], "p", "p")
    assert ok["status"] == "PASS" and ok["cells"] == 3 and ok["sharpe"]["claim"] is True and ok["sharpe"]["mean_diff_b_minus_a"] == 1.0
    missing = Q.confirm_panel(a, b[:-1], ids, [0, 1, 2], [3, 4, 5], "p", "p")
    assert missing["status"] == "NOT_EVALUABLE" and missing["missing"]["b"] == ["C:s5"] and missing["sharpe"]["claim"] == "NOT_EVALUABLE"
    ident = Q.confirm_panel(a, b, ids, [0, 1, 2], [3, 4, 5], "p", "q")
    assert ident["status"] == "NOT_EVALUABLE" and not ident["identity"]["params_sha_equal"]
    code = Q.confirm_panel(a, _recs(ids, [3, 4, 5], code="c2"), ids, [0, 1, 2], [3, 4, 5], "p", "p")
    assert code["status"] == "NOT_EVALUABLE" and not code["identity"]["code_sha_single"]
    wrong_seed = Q.confirm_panel(a, _recs(ids, [3, 4, 6]), ids, [0, 1, 2], [3, 4, 5], "p", "p")
    assert wrong_seed["status"] == "NOT_EVALUABLE" and wrong_seed["identity"]["undeclared_seeds"]["b"] == [6]
    bn = _recs(ids, [3, 4, 5])
    bn[0]["sharpe"] = float("nan")
    nonfinite = Q.confirm_panel(a, bn, ids, [0, 1, 2], [3, 4, 5], "p", "p")
    assert nonfinite["status"] == "NOT_EVALUABLE" and nonfinite["not_evaluable_cells"] == ["A"] and nonfinite["cells"] == 2   # listed, never dropped silently


def test_qualification_spec_is_v2_with_construction_history_and_diagnostics():
    d = json.loads(Q.SPEC_PATH.read_text())
    assert d["schema"] == "strategy_qualification.v2" and "v2" in d["versions"]
    nc = d["D4_negative_controls"]
    assert "construction" in nc and "history_v1" in nc and "delta" in nc["expected"] and "monotonicity" in nc["diagnostics"]["delayed_sharpe_by_k"]
    d7 = d["D7_closer_quality_vs_information_limited_oracle"]
    assert "monotone" not in d7["expected_per_family_and_side"] and "monotone_in_quality" in d7["diagnostics_not_witnesses"]
    assert "verdict_rule_v2" in d["D3_unseen_confirmation"] and "verdict_rule_v2" in d["D4_seeded_positions"]
