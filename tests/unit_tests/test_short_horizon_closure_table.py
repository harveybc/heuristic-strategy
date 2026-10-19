"""Owner closure-table rule: the table is generated from retained artifacts, never retyped.

Written before tools/short_horizon_closure_table.py existed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools import short_horizon_closure_table as CT


def _artifacts(tmp: Path):
    (tmp / "calendar_range").mkdir()
    pd.DataFrame({"block_hours": [24, 24], "period": ["2024", "2024"], "model": ["HAR", "HAR"],
                  "direction": ["coin", "logit"], "leverage_scenario": [100.0, 100.0],
                  "mae_r_ratio_mean20": [0.80, 0.80], "mae_u_ratio_mean20": [0.95, 0.95],
                  "mae_d_ratio_mean20": [0.90, 0.90], "mae_r_pips": [16.0, 16.0],
                  "direction_accuracy": [0.5, 0.56], "sharpe_daily_sqrt365": [-1.0, -0.2], "valid": [True, True]}) \
        .to_csv(tmp / "calendar_range" / "scores.csv", index=False)
    pd.DataFrame({"block_hours": [24, 24], "blocks": ["event", "no_event"], "n": [10, 5],
                  "mean_range_pips": [60.0, 50.0], "har_mae_r_ratio_mean20": [0.9, 0.8]}) \
        .to_csv(tmp / "calendar_range" / "event_diagnostic.csv", index=False)
    (tmp / "causal_event").mkdir()
    (tmp / "causal_event" / "BREAK_EVENS.json").write_text(json.dumps(
        {"DECISION_2011_2017": {"direction_accuracy": 0.66, "drift_pips": None},
         "CONFIRM_2018_2020": {"direction_accuracy": 0.72, "drift_pips": 31.0}}))
    (tmp / "causal_event" / "PREP.json").write_text(json.dumps(
        {"clock": {"vol_ratio_peak_vs_other_hours": 3.9, "days": 47}, "nfp_release_bar_vol_ratio_vs_previous_day": 1.36}))
    pd.DataFrame({"arm": ["NEXT_BAR_UNGATED", "NEXT_BAR_UNGATED", "NEXT_BAR"], "stage": ["decision_chosen", "confirm", "confirm"],
                  "period": ["DECISION_2011_2017", "CONFIRM_2018_2020", "CONFIRM_2018_2020"],
                  "event_spread": ["5pips"] * 3, "leverage": [100.0] * 3, "trades": [133, 37, 0],
                  "direction_accuracy": [0.55, 0.49, float("nan")], "sharpe_daily_sqrt365": [-1.5, -1.1, 0.0],
                  "valid": [False, False, False]}).to_csv(tmp / "causal_event" / "real_rule.csv", index=False)
    pd.DataFrame({"family": ["A", "B"], "n": [50, 40], "p_value": [0.05, 0.6], "tradable": [False, False],
                  "slope": [1e-4, 2e-4]}).to_csv(tmp / "causal_event" / "families_decision.csv", index=False)
    (tmp / "predictors").mkdir()
    pd.DataFrame({"mode": ["path"], "predictor": ["PERSISTENCE_PATH"], "block_hours": [24], "split": ["VALIDATION_2024"],
                  "cost_mult": [1], "path_mae_mean": [0.0021], "path_mae_ratio_mean": [1.0]}) \
        .to_csv(tmp / "predictors" / "predictores_resultados.csv", index=False)
    (tmp / "volatility_strategies").mkdir()
    (tmp / "volatility_strategies" / "break_even_v1_v2.json").write_text(json.dumps(
        [{"block_hours": 24, "split": "VALIDATION_2024", "strategy": "V1", "family": "a1r", "ideal_sharpe": None,
          "be_rr": None, "be_ideal_valid": False, "ideal_trades_per_week": 1.6},
         {"block_hours": 24, "split": "VALIDATION_2024", "strategy": "RFM_COIN", "family": "a1r", "ideal_sharpe": 12.7,
          "be_rr": 0.91, "be_ideal_valid": True, "ideal_trades_per_week": 3.8}]))


def test_rows_come_from_artifacts_with_skill_on_the_same_rows(tmp_path):
    _artifacts(tmp_path)
    rows = CT.closure_rows(tmp_path)
    df = pd.DataFrame(rows)
    assert {"item", "error", "naive_same_rows", "skill", "literature", "comparability", "evidence"} <= set(df.columns)
    har = df[df.item.str.contains("HAR range") & df.item.str.contains("24 h")].iloc[0]
    assert har.skill == pytest.approx((0.80 - 1.0) / 1.0)          # (err - naive) / naive on the same rows
    assert har.comparability.startswith("COMPARABLE") or har.comparability.startswith("NOT_COMPARABLE")
    assert all(("COMPARABLE" in c) or ("NO_NEW_MEASUREMENT" in c) for c in df.comparability)
    ev = df[df.item.str.contains("Event-day HAR range")].iloc[0]
    assert ev.skill == pytest.approx((0.9 - 0.8) / 0.8)
    assert any("Corsi" in str(x) for x in df.literature)


def test_missing_artifact_yields_no_new_measurement_not_a_crash(tmp_path):
    _artifacts(tmp_path)
    (tmp_path / "volatility_strategies" / "break_even_v1_v2.json").unlink()
    df = pd.DataFrame(CT.closure_rows(tmp_path))
    v1 = df[df.item.str.contains("V1")]
    assert len(v1) == 1 and v1.comparability.iloc[0] == "NO_NEW_MEASUREMENT"


def test_markdown_table_has_one_line_per_row_and_decimal_commas(tmp_path):
    _artifacts(tmp_path)
    rows = CT.closure_rows(tmp_path)
    md = CT.markdown(rows)
    body = [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("| Item")]
    assert len(body) == len(rows)
    assert "0,80" in md and "0.80" not in md.replace("0.80 ", "")
