"""The threshold optimizer's pure parts: gene decoding, nondominated union, bound hits and the owner's closure table."""
import json

import numpy as np
import pytest

pytest.importorskip("deap")

from tools import expert_guided_multires_deap as D  # noqa: E402


def test_genes_decode_to_thresholds_with_fixed_min_votes_and_rounding():
    th = D.thresholds_of([0.123456, 0.5, 0.25, 0.1, 0.9])
    assert th.theta_f == 0.1235 and th.min_votes == 1 and D.key_of([0.123456, 0.5, 0.25, 0.1, 0.9])[0] == 0.1235
    assert [g[0] for g in D.GENES] == ["theta_f", "theta_mae", "theta_vol", "theta_close", "f_close_take"]


def test_objectives_penalize_invalid_demonstrations():
    assert D.objectives({"sharpe_daily_sqrt365": 3.0, "max_drawdown_pct": 2.0, "valid": True}) == (3.0, 2.0)
    assert D.objectives({"sharpe_daily_sqrt365": 3.0, "max_drawdown_pct": 2.0, "valid": False}) == (D.ZO.PENALTY["sharpe"], D.ZO.PENALTY["dd"])


def _pt(k, sh, dd):
    return {"key": tuple(k), "genes": list(k), "obj": [sh, dd]}


def test_nondominated_union_and_bound_hits():
    pts = [_pt([0.1, 0.5, 0.3, 0.1, 0.9], 10, 5), _pt([0.5, 0.5, 0.3, 0.1, 1.0], 8, 2), _pt([0.3, 0.5, 0.3, 0.1, 0.7], 7, 6),
           _pt([0.1, 0.5, 0.3, 0.1, 0.9], 10, 5)]
    front = D.nondominated(pts)
    assert [p["obj"] for p in front] == [[10, 5], [8, 2]]                 # the third is dominated by the first; duplicate dropped
    hits = D.bound_hits(front)
    assert hits["theta_f"] == {"lower": 1, "upper": 0, "n": 2} and hits["f_close_take"] == {"lower": 0, "upper": 1, "n": 2}


def _final(chosen=True):
    m = lambda s, d, v=True: {"sharpe_daily_sqrt365": s, "max_drawdown_pct": d, "orders_per_week": 4.0, "valid": v}  # noqa: E731
    return {"chosen": ({"thresholds": {"theta_f": 0.4}, "metrics": {"DECISION_2020_2022": m(12, 3), "VALIDATION_2023": m(11, 2), "OBSERVED_2024": m(10, 4)}}
                       if chosen else None),
            "comparators": {"default_thresholds": {"DECISION_2020_2022": m(11.5, 3.9), "VALIDATION_2023": m(12.2, 1.7)},
                            "registered_optimum_demonstration": {"DECISION_2020_2022": m(17.7, 7.2), "VALIDATION_2023": m(18.2, 3.0)}},
            "union_front_size": 5, "n_valid_both": 4, "bound_hits": {"theta_f": {"lower": 0, "upper": 0, "n": 5}},
            "chains": [{"seed": 1, "generations": 100, "stop": "CONVERGED", "hv": [1.0] * 101, "evals": 4848, "sims": 3000, "seconds": 600.0}],
            "limitation": "HINDSIGHT LIMITATION: test"}


def test_closure_table_has_naive_skill_literature_and_single_read_rows():
    rows = D.closure_rows(_final())
    per = {(r["period"], r["naive"].split(":")[0]) for r in rows}
    assert ("DECISION_2020_2022", "default_thresholds") in per and ("OBSERVED_2024", "registered_optimum_demonstration") in per
    r24 = [r for r in rows if r["period"] == "OBSERVED_2024"]
    assert all("NOT_READ" in r["naive"] and "single-read" in r["skill"] for r in r24)
    r0 = next(r for r in rows if r["period"] == "DECISION_2020_2022" and r["naive"].startswith("default_thresholds"))
    assert "delta Sharpe +0.50" in r0["skill"] and r0["literature"].startswith("NOT_COMPARABLE")
    md = D.closure_markdown(_final())
    assert "| DECISION_2020_2022 |" in md and "HINDSIGHT LIMITATION" in md and "| 1 | 100 | CONVERGED |" in md
    assert D.closure_rows(_final(chosen=False))[0]["comparability"] == "NO_NEW_MEASUREMENT"
    json.dumps(_final())
