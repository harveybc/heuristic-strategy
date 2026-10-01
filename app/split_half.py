"""Split-half check of horizon selection, and per-config seed summaries (diagnostics only).

Choosing the passing horizons on the same validation the backtest is scored on is
optimistic. The split-half check chooses the horizons on the FIRST half of the
validation origins (model MAE strictly below the strict-minimum naive, computed here
from the predictions and the bars) and reports the backtest and the MAE on the SECOND
half only.

The MAE is computed in log-return space. With z = (R - n·mu)/sigma, every MAE is
divided by the same sigma, so the comparisons equal the z_train ones. The naives are
zero return (R̂ = 0) and train mean (R̂ = n·mu); the strict minimum is the smaller of
the two, as in M07's records. ``mu`` must be the declared train mean of the 1-bar log
return.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from app.paired_backtest import (HeuristicParams, decide_targets, episode_metrics, no_trade_baseline,
                                 run_episode)


def horizon_mae(bars: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
                horizons: Sequence[int], mu: float, origins: set | None = None,
                steps: Mapping[int, int] | None = None) -> dict[int, dict[str, Any]]:
    """Per horizon: model, zero-return, train-mean and strict-minimum naive MAE on the same origins.

    ``predictions`` rows carry ``time`` and ``logret_hat_h<h>``. The target is
    log(close[t+n]/close[t]) with n = steps[h] (default h) bars ahead. Origins without a
    target inside ``bars`` are dropped from every column alike.
    """
    index = {bar["time"]: i for i, bar in enumerate(bars)}
    out = {}
    for h in horizons:
        n = (steps or {}).get(h, h)
        model, zero, mean = [], [], []
        for row in predictions:
            t = row["time"]
            if (origins is not None and t not in origins) or t not in index:
                continue
            i = index[t]
            if i + n >= len(bars):
                continue
            actual = math.log(float(bars[i + n]["close"]) / float(bars[i]["close"]))
            model.append(abs(float(row[f"logret_hat_h{h}"]) - actual))
            zero.append(abs(actual))
            mean.append(abs(n * mu - actual))
        if not model:
            out[h] = {"rows": 0, "status": "NOT_AVAILABLE", "reason": "no origin with a target"}
            continue
        m, z, a = (sum(x) / len(x) for x in (model, zero, mean))
        naive = min(z, a)
        out[h] = {"rows": len(model), "model_mae": m, "zero_return_mae": z, "train_mean_mae": a,
                  "naive_mae": naive, "strict_naive": "zero_return" if z <= a else "train_mean",
                  "delta": m - naive, "skill": (1 - m / naive) if naive > 0 else "NOT_AVAILABLE",
                  "passes": m < naive}
    return out


def split_half(bars: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
               horizons: Sequence[int], mu: float, params: HeuristicParams,
               price_column: str = "close_hat_h{h}") -> dict[str, Any]:
    """Choose horizons on the first half of the forecast origins; report the second half only."""
    times = [row["time"] for row in predictions]
    cut = len(times) // 2
    first, second = set(times[:cut]), set(times[cut:])
    chosen_table = horizon_mae(bars, predictions, horizons, mu, origins=first)
    chosen = [h for h in horizons if chosen_table[h].get("passes")]
    held_table = horizon_mae(bars, predictions, chosen, mu, origins=second) if chosen else {}
    second_bars = [bar for bar in bars if bar["time"] in second]
    by_time = {row["time"]: row for row in predictions}
    result = {"schema": "heuristic_strategy.split_half_diagnostic.v1", "status": "SPLIT_HALF_DIAGNOSTIC",
              "first_half": {"origins": len(first), "first": times[0] if times else None,
                             "last": times[cut - 1] if cut else None, "mae": chosen_table, "chosen": chosen},
              "second_half": {"origins": len(second), "first": times[cut] if times[cut:] else None,
                              "last": times[-1] if times else None, "mae_on_chosen": held_table,
                              "chosen_still_passing": [h for h in chosen if held_table[h].get("passes")]},
              "baselines": {"no_trade": no_trade_baseline(len(second_bars))}}
    if not chosen or not second_bars:
        result["second_half"]["metrics"] = None
        result["second_half"]["note"] = "no horizon passed on the first half: the strategy does not run"
        return result
    forecasts = [[float(by_time[bar["time"]][price_column.format(h=h)]) for h in chosen] for bar in second_bars]
    episode = run_episode(second_bars, decide_targets(second_bars, forecasts, params))
    result["second_half"]["metrics"] = episode_metrics(episode)
    return result


def seed_summary(results: Sequence[Mapping[str, Any]], key: str = "label") -> dict[str, Any]:
    """Mean and sample sd over seeds per config; skipped seeds are counted, never averaged in."""
    groups: dict[str, list] = {}
    for item in results:
        groups.setdefault(item[key], []).append(item)
    out = {}
    for label, items in sorted(groups.items()):
        ran = [i for i in items if i.get("metrics")]
        row = {"seeds": sorted(i.get("seed") for i in items), "ran": len(ran),
               "skipped": len(items) - len(ran)}
        for metric in ("net_return", "max_drawdown_fraction", "turnover_units", "trades_closed",
                       "exposure_fraction"):
            values = [float(i["metrics"][metric]) for i in ran]
            row[metric] = {"mean": statistics.fmean(values) if values else None,
                           "sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        sharpes = [i["metrics"]["sharpe"]["value"] for i in ran if i["metrics"]["sharpe"]["value"] is not None]
        row["sharpe"] = {"mean": statistics.fmean(sharpes) if sharpes else None,
                         "sd": statistics.stdev(sharpes) if len(sharpes) > 1 else None, "n": len(sharpes),
                         "undefined": len(ran) - len(sharpes)}
        out[label] = row
    return out
