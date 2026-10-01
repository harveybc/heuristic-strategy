"""Explicit cost lines, per-run observability, and the direction / per-horizon ablations."""
import json

import pytest

from app.paired_backtest import HeuristicParams, decide_targets, paired_backtest
from tests.unit_tests.test_paired_backtest import _bars, _declared, _predictions, _record
from tests.unit_tests.test_forecast_naive_gate import _horizon


def _run(**kw):
    record = _record([_horizon(1, 0.5, 0.9), _horizon(2, 0.6, 0.9)])
    bars = _bars([100 + 3 * ((-1) ** i) for i in range(40)])
    return paired_backtest(bars, _predictions(bars, [1, 2]), record, _declared(record, [1, 2]),
                           HeuristicParams(**kw.pop("params", {})), **kw)


def test_cost_lines_are_explicit_and_marked_modelled_or_broker_fill():
    out = _run()
    lines = {line["line"]: line for line in out["cost_lines"]}
    fills = out["trajectory"]["fills"]
    expected = sum(0.001 * abs(delta) * price for _, delta, price in fills)
    assert lines["commission"]["source"] == "MODELLED" and lines["commission"]["value"] == pytest.approx(expected)
    assert lines["slippage"]["source"] == "MODELLED" and lines["slippage"]["value"] == 0.0
    assert lines["swap_financing"]["source"] == "MODELLED" and lines["swap_financing"]["value"] == 0.0
    assert lines["broker_fill_costs"]["source"] == "BROKER_FILL"
    assert lines["broker_fill_costs"]["status"] == "NOT_AVAILABLE" and lines["broker_fill_costs"]["reason"]
    assert out["sizing"] == {"position_units": 1.0, "initial_cash": 10000.0, "rule": "one asset unit per entry"}


def test_observability_records_every_bar_decision_and_reason(tmp_path):
    out = _run(observability_path=tmp_path / "obs.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "obs.jsonl").read_text().splitlines()]
    header, bars = lines[0], lines[1:]
    assert header["kind"] == "header" and header["gate"]["consumed_horizons"] == [1, 2]
    assert len(bars) == 40 and all(b["kind"] == "bar" for b in bars)
    assert {b["reason"] for b in bars} <= {"entry_long", "entry_short", "take_profit", "stop_loss", "hold_flat",
                                           "hold_position", "no_forecast"}
    assert [b["target"] for b in bars] == out["trajectory"]["targets"]


def test_skipped_runs_write_the_gate_reasons_to_observability(tmp_path):
    record = _record([_horizon(1, 1.0, 0.9)])
    bars = _bars([100.0] * 10)
    out = paired_backtest(bars, _predictions(bars, [1]), record, _declared(record, [1]), HeuristicParams(),
                          observability_path=tmp_path / "obs.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "obs.jsonl").read_text().splitlines()]
    assert out["status"] == "SKIPPED_NOT_BETTER_THAN_NAIVE"
    assert lines[0]["kind"] == "header" and lines[0]["skipped_by_gate"][0]["failures"][0]["reason"] == \
        "not_better_than_naive"
    assert len(lines) == 1


@pytest.mark.parametrize("direction, wanted, forbidden", [("long_only", 1, -1), ("short_only", -1, 1)])
def test_direction_ablation_never_takes_the_other_side(direction, wanted, forbidden):
    record = _record([_horizon(1, 0.5, 0.9)])
    bars = _bars([100.0] * 40)
    preds = _predictions(bars, [1], value_for=lambda i, h: 103.0 if (i // 5) % 2 == 0 else 97.0)
    out = paired_backtest(bars, preds, record, _declared(record, [1]), HeuristicParams(direction=direction))
    targets = out["trajectory"]["targets"]
    assert forbidden not in targets and wanted in targets  # not vacuous: the allowed side does trade
    both = paired_backtest(bars, preds, record, _declared(record, [1]), HeuristicParams())
    assert 1 in both["trajectory"]["targets"] or -1 in both["trajectory"]["targets"]


def test_ablation_runner_requires_a_fully_eligible_candidate(tmp_path):
    from run_ablations import ablations

    record = _record([_horizon(1, 0.5, 0.9), _horizon(2, 1.2, 0.9)])  # h2 fails: not eligible
    bars = _bars([100 + 3 * ((-1) ** i) for i in range(40)])
    out = ablations(bars, _predictions(bars, [1, 2]), record, _declared(record, [1, 2]))
    assert out["status"] == "NOT_ELIGIBLE_NO_ABLATION" and out["runs"] == []
    good = _record([_horizon(1, 0.5, 0.9), _horizon(2, 0.6, 0.9)])
    out = ablations(bars, _predictions(bars, [1, 2]), good, _declared(good, [1, 2]))
    names = sorted(run["name"] for run in out["runs"])
    assert names == sorted(["both/all", "long_only/all", "short_only/all", "both/h1", "both/h2"])
    assert all(run["result"]["naive_gate"]["passed"] for run in out["runs"])
