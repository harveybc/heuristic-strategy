"""Forecasting + heuristic backtest, paired with lane G's RL arms on the same episode.

Execution and metrics mirror lane G's ``rl_temporal/reconciliation.py::reconcile_episode``
(agent-multi 02db0701), as lane G specified them:

* the decision is taken after bar t closes and a market order fills at bar t+1 OPEN;
* commission is 0.001 of notional per side, with no slippage, no spread and no financing;
* one asset unit per position and an initial cash of 10,000;
* ``net_return`` = (final equity - initial cash) / initial cash, with equity marked at
  each close;
* ``max_drawdown_fraction`` is the per-bar peak-to-trough on that equity curve;
* ``sharpe`` = mean/std (ddof=1) of per-bar equity returns over ALL bars, not
  annualized. It is None with a reason when there are fewer than 2 returns or the
  variance is zero or non-finite;
* ``turnover_units`` = sum of |change in position|, ``trades_closed`` counts closed
  trades (a reversal closes one), and ``exposure_fraction`` = bars in position / bars.

The forecast-versus-naive gate (``app.forecast_naive_gate``, contract unchanged)
decides which horizons the strategy may read, from the evidence record alone.
Failing horizons are excluded through an explicitly declared reduced-input experiment
that lists each one with its MAE beside the same-row naive, and they are never read.
If no horizon passes, the strategy does not run and no trajectory is produced.

The decision rule is the heuristic strategy's entry rule (plugin_long_short_predictions,
mirrored in lts ``plugins_strategy/heuristic_strategy.compute_signal``), stated in
price fractions instead of FX pips. Exits are TP/SL on the close only (exit variant G).
The rule takes 1 unit to match lane G's sizing, and its parameters are frozen in
``HeuristicParams`` before any run. They are never tuned on the episode.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from app.forecast_naive_gate import CONTRACT_SHA256, ELIGIBLE, SKIPPED, evaluate

RESULT_SCHEMA = "heuristic_strategy.paired_backtest_result.v1"
COSTS = {"commission": 0.001, "commission_unit": "fraction of notional per side",
         "slippage": 0.0, "spread": "not modelled", "financing_enabled": False,
         "fill": "market order after bar t close fills at bar t+1 OPEN",
         "initial_cash": 10000.0, "position_units": 1.0}
SHARPE_CONVENTION = "per-bar equity returns, ddof=1, all bars incl. flat, not annualized"


@dataclass(frozen=True)
class HeuristicParams:
    """Frozen before the run; never tuned on the paired episode."""
    profit_threshold_frac: float = 0.005
    min_drawdown_frac: float = 0.0025
    tp_multiplier: float = 0.9
    sl_multiplier: float = 2.0
    exit_variant: str = "G (TP/SL on close only)"


def run_episode(bars: Sequence[Mapping[str, Any]], targets: Sequence[int]) -> dict[str, Any]:
    """Apply target positions decided after each close; fills at the next bar's open."""
    if len(targets) != len(bars):
        raise ValueError("one target per bar is required")
    cash, position = COSTS["initial_cash"], 0
    equity, positions, fills = [], [], []
    for t, bar in enumerate(bars):
        if t > 0:
            delta = int(targets[t - 1]) - position
            if delta:
                price = float(bar["open"])
                cash -= delta * price + COSTS["commission"] * abs(delta) * price
                fills.append((t, delta, price))
                position += delta
        positions.append(position)
        equity.append(cash + position * float(bar["close"]))
    return {"equity": equity, "positions": positions, "fills": fills, "targets": [int(x) for x in targets]}


def _sharpe(equity: Sequence[float]) -> dict[str, Any]:
    returns = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    if len(returns) < 2:
        return {"value": None, "convention": SHARPE_CONVENTION, "undefined_reason": "fewer than 2 bar returns"}
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if not math.isfinite(var) or var <= 0:
        return {"value": None, "convention": SHARPE_CONVENTION, "undefined_reason": "zero variance of bar returns"}
    return {"value": mean / math.sqrt(var), "convention": SHARPE_CONVENTION}


def episode_metrics(episode: Mapping[str, Any]) -> dict[str, Any]:
    equity, positions = episode["equity"], episode["positions"]
    peak, drawdown = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak)
    closed, previous = 0, 0
    for position in positions:
        if previous != 0 and (position == 0 or (position > 0) != (previous > 0)):
            closed += 1
        previous = position
    return {"net_return": (equity[-1] - COSTS["initial_cash"]) / COSTS["initial_cash"],
            "max_drawdown_fraction": drawdown, "sharpe": _sharpe(equity),
            "turnover_units": sum(abs(delta) for _, delta, _ in episode["fills"]),
            "trades_closed": closed,
            "exposure_fraction": sum(1 for p in positions if p != 0) / len(positions)}


def no_trade_baseline(bars: int) -> dict[str, Any]:
    """Identical episode, equity flat (lane G item 4); computed without the simulator."""
    return {"net_return": 0.0, "max_drawdown_fraction": 0.0,
            "sharpe": {"value": None, "convention": SHARPE_CONVENTION,
                       "undefined_reason": "zero variance of bar returns (flat equity)"},
            "turnover_units": 0, "trades_closed": 0, "exposure_fraction": 0.0, "bars": bars}


def decide_targets(bars, forecasts: Sequence[Sequence[float]], params: HeuristicParams) -> list[int]:
    """Heuristic entry over the readable horizons; TP/SL exits on the close."""
    targets, position, tp, sl = [], 0, None, None
    for bar, preds in zip(bars, forecasts):
        price = float(bar["close"])
        if position != 0:
            hit_tp = price >= tp if position > 0 else price <= tp
            hit_sl = price <= sl if position > 0 else price >= sl
            if hit_tp or hit_sl:
                position, tp, sl = 0, None, None
        elif preds:
            high, low = max(preds), min(preds)
            profit_long, profit_short = high - price, price - low
            dd_long = max(price - low, params.min_drawdown_frac * price)
            dd_short = max(high - price, params.min_drawdown_frac * price)
            rr_long = profit_long / dd_long if dd_long > 0 else 0.0
            rr_short = profit_short / dd_short if dd_short > 0 else 0.0
            if profit_long / price >= params.profit_threshold_frac and rr_long >= rr_short:
                position = 1
                tp = price + params.tp_multiplier * profit_long
                sl = price - params.sl_multiplier * dd_long
            elif profit_short / price >= params.profit_threshold_frac and rr_short > rr_long:
                position = -1
                tp = price - params.tp_multiplier * profit_short
                sl = price + params.sl_multiplier * dd_short
        targets.append(position)
    return targets


def paired_backtest(bars: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
                    record: Mapping[str, Any], declared: Mapping[str, Any], params: HeuristicParams,
                    *, family: str = "forecast", split: str = "validation",
                    manifest_status: str = "UNKNOWN", missing_forecast: str = "refuse") -> dict[str, Any]:
    """``missing_forecast="hold"`` lets bars without a forecast take no new entry (exits still
    apply); the count and the bars are recorded. The default refuses."""
    by_time = {row["time"]: row for row in predictions}
    missing = [bar["time"] for bar in bars if bar["time"] not in by_time]
    if missing and missing_forecast != "hold":
        raise ValueError(f"forecast missing for bar {missing[0]}")
    spec = (declared.get("families") or {}).get(family) or {}
    horizons = list(spec.get("horizons") or [])
    full = evaluate({family: record}, declared, {family: len(horizons)}, declared.get("asset"), families=(family,))
    by_horizon = {}
    for failure in full["failures"]:
        by_horizon.setdefault(failure["horizon"], []).append(failure)
    global_failures = by_horizon.pop(None, [])
    rows = {row["horizon"]: row for row in full["horizons"]}
    excluded = [dict(rows.get(h, {"horizon": h}), failures=by_horizon[h]) for h in horizons if h in by_horizon]
    passing = [h for h in horizons if h not in by_horizon]
    gate = {"contract_sha256": CONTRACT_SHA256, "evidence_sha256": record.get("evidence_sha256"),
            "declared_horizons": horizons, "consumed_horizons": passing,
            "excluded_horizons": excluded, "global_failures": global_failures,
            "passing_rows": [rows[h] for h in passing if h in rows], "provenance": full["provenance"]}
    population = {"split": split, "episodes": 1, "rows": len(bars), "first": bars[0]["time"],
                  "last": bars[-1]["time"], "selection_metric": "none (paired evaluation, no selection)",
                  "bars_without_forecast": {"count": len(missing), "policy": missing_forecast,
                                            "first": missing[0] if missing else None,
                                            "last": missing[-1] if missing else None}}
    if global_failures or not passing:
        gate.update(passed=False, reduced_input_experiment={"declared": False})
        return {"schema": RESULT_SCHEMA, "arm": "heuristic_forecast", "status": SKIPPED,
                "naive_gate": gate, "evaluation_population": population, "costs": COSTS,
                "params": asdict(params), "metrics": None, "trajectory": None,
                "baselines": {"no_trade": no_trade_baseline(len(bars)), "heuristic": "UNAVAILABLE"}}
    if excluded:
        reduced = dict(declared, families={family: dict(spec, horizons=passing)})
        check = evaluate({family: record}, reduced, {family: len(passing)}, declared.get("asset"),
                         families=(family,))
        if check["status"] != ELIGIBLE:
            raise RuntimeError("the reduced configuration did not re-verify")
        gate["reduced_input_experiment"] = {
            "declared": True, "reason": "horizons failing the naive gate are excluded; this is a separately "
                                        "declared reduced-input experiment, not the strategy's full configuration",
            "consumed": passing, "excluded": [r["horizon"] for r in excluded]}
    else:
        gate["reduced_input_experiment"] = {"declared": False}
    gate["passed"] = True
    forecasts = [[float(by_time[bar["time"]][h]) for h in passing] if bar["time"] in by_time else []
                 for bar in bars]  # failing columns are never read; [] = no forecast, no new entry
    targets = decide_targets(bars, forecasts, params)
    episode = run_episode(bars, targets)
    status = {"FROZEN": "RESULT", "FROZEN_DEVELOPMENT": "DEVELOPMENT_NOT_CONFIRMATORY"}.get(
        manifest_status, "PILOT_NOT_A_RESULT")
    result = {"schema": RESULT_SCHEMA, "arm": "heuristic_forecast", "status": status,
              "manifest_status": manifest_status, "naive_gate": gate, "evaluation_population": population,
              "costs": COSTS, "params": asdict(params), "metrics": episode_metrics(episode),
              "baselines": {"no_trade": no_trade_baseline(len(bars)),
                            "heuristic": {"naive_gate": {"passed": True, "evidence": record.get("evidence_sha256")}}},
              "trajectory": {"targets": episode["targets"], "fills": episode["fills"]}}
    result["digest"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False, default=str).encode()).hexdigest()
    return result
