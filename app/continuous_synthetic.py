"""Bounded hourly synthetic sensitivity pilot for the unchanged variant E book."""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import pandas as pd

from app.elapsed_hour_harness import _horizon_errors, _label_frame, _rename_family
from app.strategy_support import (
    RESERVED_START_UTC, _normalize_one, create_elapsed_hour_predictions,
    fit_development_parameters,
)
from app.sweep_executor import (
    HORIZONS, LONG_HORIZONS, SHORT_HORIZONS, _build_family_frame,
    _plain, run_book, synthetic_dev_frame,
)


def build_trajectory() -> pd.DataFrame:
    """Fifteen synthetic days; 216 consecutive origins retain all 144h targets."""
    index = pd.date_range("2019-05-01", periods=360, freq="h", name="DATE_TIME")
    closes = [
        1.10 + 0.012 * math.sin(i / 10.0) + 0.004 * math.sin(i / 3.0)
        for i in range(len(index))
    ]
    frame = pd.DataFrame({"CLOSE": closes}, index=index)
    frame["OPEN"] = frame["CLOSE"].shift(1).fillna(frame["CLOSE"].iloc[0])
    frame["HIGH"] = frame[["OPEN", "CLOSE"]].max(axis=1) + 0.0002
    frame["LOW"] = frame[["OPEN", "CLOSE"]].min(axis=1) - 0.0002
    return frame[["OPEN", "HIGH", "LOW", "CLOSE"]]


def prepare_continuous(
    bars: pd.DataFrame,
    dev: pd.DataFrame,
    short_kind: str,
    long_kind: str,
    *,
    intensity: float = 1.0,
    seed: int = 42,
) -> dict:
    """All eligible hourly origins share support; DEV is strictly prior."""
    if not isinstance(bars.index, pd.DatetimeIndex) or (
        len(bars.index) and _normalize_one(bars.index.max(), source_timezone="UTC") >= RESERVED_START_UTC
    ):
        raise ValueError("trajectory must end before the reserved cut")
    label = _label_frame(bars, source_timezone="UTC")
    if label.empty:
        raise ValueError("trajectory is empty")
    ideal = create_elapsed_hour_predictions(label, HORIZONS, source_timezone="UTC")
    if ideal.empty:
        raise ValueError("trajectory has no complete hourly origins")
    fitted = fit_development_parameters(dev, HORIZONS, source_timezone="UTC")
    if not fitted.consumed_timestamps_utc or max(fitted.consumed_timestamps_utc) >= ideal.index.min():
        raise ValueError("DEV scale overlaps scored trajectory")
    scales = {str(h): float(v) for h, v in fitted.per_horizon_mean_abs_residual}
    short = _build_family_frame(
        ideal, label, SHORT_HORIZONS, short_kind, family="short",
        intensity=intensity, seed=seed, dev_scales=scales,
    )
    long = _build_family_frame(
        ideal, label, LONG_HORIZONS, long_kind, family="long",
        intensity=intensity, seed=seed, dev_scales=scales,
    )
    elapsed = pd.concat([short, long], axis=1)
    mae, naive = _horizon_errors(label, elapsed, HORIZONS, price_column="CLOSE")
    plugin = _rename_family(short, SHORT_HORIZONS, "Prediction_h_").join(
        _rename_family(long, LONG_HORIZONS, "Prediction_d_")
    )
    plugin.index = plugin.index.tz_localize(None)
    plugin.index.name = "DATE_TIME"
    return {
        "elapsed_frame": elapsed,
        "plugin_frame": plugin,
        "per_horizon_mae": mae,
        "per_horizon_naive_mae": naive,
        "per_horizon_dev_scale": scales,
        "dev_last_utc": max(fitted.consumed_timestamps_utc).isoformat(),
    }


def run_pilot(work_root: str) -> dict:
    """Three fixed paired controls; never launches the declared 44-cell grid."""
    root = Path(work_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("pilot output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    bars = build_trajectory()
    dev = synthetic_dev_frame()
    cells = []
    for name, short_kind, long_kind in (
        ("ideal_ideal", "ideal", "ideal"),
        ("persistence_persistence", "persistence", "persistence"),
        ("short_noise_long_ideal", "noise", "ideal"),
    ):
        prepared = prepare_continuous(bars, dev, short_kind, long_kind)
        result = run_book(bars, prepared["plugin_frame"], str(root / name))
        result.update({
            "name": name,
            "per_horizon_mae": prepared["per_horizon_mae"],
            "per_horizon_naive_mae": prepared["per_horizon_naive_mae"],
            "per_horizon_dev_scale": prepared["per_horizon_dev_scale"],
            "dev_last_utc": prepared["dev_last_utc"],
            "forecast_origin_count": len(prepared["plugin_frame"]),
        })
        cells.append(result)
    report = _plain({
        "label": "SYNTHETIC",
        "b0_status": "B0_NOT_STARTED",
        "financial_utility": "NOT_CLAIMED",
        "grid_status": "NOT_RUN_PILOT_ONLY",
        "cells": cells,
    })
    (root / "pilot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the bounded synthetic hourly pilot")
    parser.add_argument("--output", required=True, help="new output directory for pilot evidence")
    args = parser.parse_args()
    result = run_pilot(args.output)
    for cell in result["cells"]:
        causes = {
            cause: sum(row["close_cause"] == cause for row in cell["events"])
            for cause in ("take_profit", "stop_loss", "early_prediction")
        }
        print(cell["name"], "origins=", cell["forecast_origin_count"],
              "trades=", cell["n_closed_trades"], "causes=", causes,
              "equity=", round(cell["marked_equity"], 2))
