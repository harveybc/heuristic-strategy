"""Paired forecasting+heuristic backtest on lane G's validation episode (ETH 4h).

Execution and metrics mirror lane G's rl_temporal/reconciliation.py::reconcile_episode
(agent-multi 02db0701): decision after bar t closes, market fill at bar t+1 OPEN,
commission 0.001 of notional per side, one unit, initial cash 10,000; net_return,
per-bar max drawdown, Sharpe = mean/std(ddof=1) of per-bar equity returns over ALL
bars (not annualized), turnover units, closed trades, exposure; no-trade baseline.
"""
import json
import math

import pytest

from app.forecast_naive_gate import seal
from app.paired_backtest import (
    COSTS, HeuristicParams, run_episode, episode_metrics, no_trade_baseline, paired_backtest,
)
from tests.unit_tests.test_forecast_naive_gate import _horizon


def _bars(closes, opens=None):
    opens = opens or closes
    return [{"time": f"bar-{i:05d}", "open": o, "high": max(o, c) * 1.001, "low": min(o, c) * 0.999, "close": c}
            for i, (o, c) in enumerate(zip(opens, closes))]


# ------------------------------------------------------------ metrics mirror lane G


def test_metrics_follow_lane_g_definitions_on_a_hand_computed_episode():
    bars = _bars([100, 100, 110, 110, 105], opens=[100, 100, 102, 110, 106])
    # target positions decided after each close: long after bar 0, flat after bar 2
    targets = [1, 1, 0, 0, 0]
    episode = run_episode(bars, targets)
    # fills: long 1 @ open[1]=100 (commission 0.1); close @ open[3]=110 (commission 0.11)
    assert episode["fills"] == [(1, 1, 100.0), (3, -1, 110.0)]
    equity = episode["equity"]
    assert equity[0] == pytest.approx(10000.0)
    assert equity[1] == pytest.approx(10000 - 100 - 0.1 + 100)       # marked at close 100
    assert equity[2] == pytest.approx(10000 - 100.1 + 110)
    assert equity[3] == pytest.approx(10000 - 100.1 + 110 - 0.11)
    m = episode_metrics(episode)
    assert m["net_return"] == pytest.approx((equity[-1] - 10000) / 10000)
    assert m["turnover_units"] == 2 and m["trades_closed"] == 1
    assert m["exposure_fraction"] == pytest.approx(2 / 5)
    returns = [equity[i] / equity[i - 1] - 1 for i in range(1, 5)]
    mean = sum(returns) / 4
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / 3)
    assert m["sharpe"]["value"] == pytest.approx(mean / std)
    assert m["sharpe"]["convention"].startswith("per-bar")
    peak, dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak)
    assert m["max_drawdown_fraction"] == pytest.approx(dd)


def test_reversal_closes_and_opens_at_the_next_open_and_last_decision_never_fills():
    bars = _bars([100, 101, 99, 98], opens=[100, 100, 102, 97])
    episode = run_episode(bars, [1, -1, -1, 1])
    assert episode["fills"] == [(1, 1, 100.0), (2, -2, 102.0)]  # the final +1 has no next bar
    m = episode_metrics(episode)
    assert m["trades_closed"] == 1 and m["turnover_units"] == 3


def test_no_trade_baseline_is_flat_with_an_explained_undefined_sharpe():
    base = no_trade_baseline(5)
    assert base["net_return"] == 0 and base["max_drawdown_fraction"] == 0
    assert base["turnover_units"] == 0 and base["trades_closed"] == 0 and base["exposure_fraction"] == 0
    assert base["sharpe"]["value"] is None and "zero variance" in base["sharpe"]["undefined_reason"]


def test_costs_are_lane_g_values():
    assert COSTS == {"commission": 0.001, "commission_unit": "fraction of notional per side",
                     "slippage": 0.0, "spread": "not modelled", "financing_enabled": False,
                     "fill": "market order after bar t close fills at bar t+1 OPEN",
                     "initial_cash": 10000.0, "position_units": 1.0}


# ------------------------------------------------------------ the gate decides which horizons run


def _record(per_horizon):
    return seal({
        "schema": "predictor.forecast_naive_evidence.v1",
        "frozen_metric": {"primary": "MAE", "secondary": "MSE", "frozen": "declared before selection"},
        "artifact": {"model_sha256": "e" * 64},
        "population": {"dataset_id": "eth", "asset": "ETHUSDT", "targets": ["CLOSE"], "rows": 500,
                       "row_ids_sha256": "a" * 64, "first_origin": "2024-01-01", "last_origin": "2024-12-31",
                       "sample_hours": 4.0, "horizon_unit": "steps of 4.0 h"},
        "scale": {"metric_space": "price", "scaler_identity": "none"},
        "split": {"provenance": "held_out_validation", "test_used": False, "reserved_trading_test": False},
        "naive": {"definition": "persistence"}, "per_horizon": per_horizon})


def _declared(record, horizons):
    return {"asset": "ETHUSDT", "families": {"forecast": {
        "evidence_sha256": record["evidence_sha256"], "model_sha256": "e" * 64, "period_hours": 4.0,
        "metric_space": "price", "scaler_identity": "none", "horizons": horizons}}}


def _predictions(bars, horizons, value_for=None):
    rows = []
    for i, bar in enumerate(bars):
        row = {"time": bar["time"]}
        for h in horizons:
            row[h] = (value_for(i, h) if value_for else bar["close"] * (1 + 0.02 * ((-1) ** i)))
        rows.append(row)
    return rows


def test_no_passing_horizon_means_no_strategy_run(monkeypatch):
    import app.paired_backtest as pb

    calls = []
    monkeypatch.setattr(pb, "run_episode", lambda *a, **k: calls.append(1))
    record = _record([_horizon(h, 1.0, 0.9) for h in (1, 2, 3)])
    bars = _bars([100 + i for i in range(12)])
    out = paired_backtest(bars, _predictions(bars, [1, 2, 3]), record, _declared(record, [1, 2, 3]),
                          HeuristicParams())
    assert calls == [] and out["status"] == "SKIPPED_NOT_BETTER_THAN_NAIVE"
    assert out["trajectory"] is None and len(out["naive_gate"]["excluded_horizons"]) == 3
    assert all(row["mae"]["baseline"] == pytest.approx(0.9) for row in out["naive_gate"]["excluded_horizons"])


def test_failing_horizons_are_excluded_by_declaration_and_never_read():
    record = _record([_horizon(1, 0.5, 0.9), _horizon(2, 1.2, 1.0), _horizon(3, 0.6, 0.9)])
    bars = _bars([100.0] * 20)
    # horizon 2 fails the gate; give it absurd values that would trade if it were read
    preds = _predictions(bars, [1, 2, 3], value_for=lambda i, h: 1e6 if h == 2 else 100.0)
    out = paired_backtest(bars, preds, record, _declared(record, [1, 2, 3]), HeuristicParams())
    gate = out["naive_gate"]
    assert gate["passed"] is True and gate["consumed_horizons"] == [1, 3]
    assert gate["reduced_input_experiment"]["declared"] is True
    assert [r["horizon"] for r in gate["excluded_horizons"]] == [2]
    assert out["metrics"]["turnover_units"] == 0  # flat forecasts on passing horizons: no trade
    assert out["baselines"]["no_trade"]["net_return"] == 0
    assert out["baselines"]["heuristic"]["naive_gate"]["passed"] is True
    json.dumps(out, allow_nan=False)


def test_decisions_are_causal_in_the_forecasts():
    record = _record([_horizon(1, 0.5, 0.9), _horizon(2, 0.6, 0.9)])
    bars = _bars([100 + (i % 3) for i in range(30)])
    preds = _predictions(bars, [1, 2])
    base = paired_backtest(bars, preds, record, _declared(record, [1, 2]), HeuristicParams())
    changed = [dict(p) for p in preds]
    for row in changed[20:]:
        row[1] = row[1] * 3
    other = paired_backtest(bars, changed, record, _declared(record, [1, 2]), HeuristicParams())
    assert base["trajectory"]["targets"][:20] == other["trajectory"]["targets"][:20]


def test_rows_outside_the_declared_episode_are_refused():
    record = _record([_horizon(1, 0.5, 0.9)])
    bars = _bars([100.0] * 10)
    preds = _predictions(bars, [1])[:-1]  # one bar without a forecast
    with pytest.raises(ValueError, match="forecast missing"):
        paired_backtest(bars, preds, record, _declared(record, [1]), HeuristicParams())


def test_cli_runs_end_to_end_on_files(tmp_path):
    import csv
    import run_paired_eth4h

    closes = [100 + 3 * ((-1) ** i) for i in range(40)]
    with open(tmp_path / "view.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"])
        for i, c in enumerate(closes):
            w.writerow([f"bar-{i:05d}", c, c * 1.01, c * 0.99, c])
    with open(tmp_path / "pred.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["DATE_TIME", "h_1", "h_2"])
        for i, c in enumerate(closes[10:30], start=10):
            w.writerow([f"bar-{i:05d}", c * 1.03, c * 0.97])
    record = _record([_horizon(1, 0.5, 0.9), _horizon(2, 1.0, 0.9)])
    (tmp_path / "ev.json").write_text(json.dumps(record))
    (tmp_path / "decl.json").write_text(json.dumps(_declared(record, [1, 2])))
    assert run_paired_eth4h.main(["--bars", str(tmp_path / "view.csv"), "--rows", "10:30",
                                  "--predictions", str(tmp_path / "pred.csv"), "--evidence", str(tmp_path / "ev.json"),
                                  "--declared", str(tmp_path / "decl.json"), "--out", str(tmp_path / "out.json")]) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["evaluation_population"]["rows"] == 20 and out["status"] == "PILOT_NOT_A_RESULT"
    assert out["naive_gate"]["consumed_horizons"] == [1] and out["inputs"]["rows"] == "10:30"
