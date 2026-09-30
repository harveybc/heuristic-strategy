"""Short/long factorial manifest. It does not run a cell and does not start B0.

The historic 3600 CPU-second ceiling of the 241-cell sweep is not a B0 budget.
Holdout decisions are out of scope. Scales are computed only on DEV rows,
strictly before 2019-05-16 00:00 UTC.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from app.strategy_support import (
    B0_STATUS,
    RESERVED_START_UTC,
    fit_development_parameters,
)

PAIRED_SEEDS = (42, 43, 44)
INPUT_KINDS = ("persistence", "mae_matched_noise", "ideal")
ORIENTATIONS = (
    {"varying": "short", "fixed": "long", "fixed_kind": "ideal"},
    {"varying": "long", "fixed": "short", "fixed_kind": "ideal"},
)
HISTORIC_CPU_CEILING_SECONDS = 3600
HISTORIC_CEILING_IS_B0_BUDGET = False


def scale_on_dev(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    *,
    source_timezone: str,
    price_column: str = "CLOSE",
) -> dict[int, float]:
    """Per-horizon mean absolute residual on DEV only. Not a noise model."""
    fitted = fit_development_parameters(
        frame,
        horizons,
        source_timezone=source_timezone,
        price_column=price_column,
    )
    if fitted.fits_noise_model:
        raise RuntimeError("DEV scale must not fit a noise model")
    return {hours: value for hours, value in fitted.per_horizon_mean_abs_residual}


def family_input(
    kind: str,
    *,
    origin_price: float,
    future_prices: Sequence[float],
    target_mae: float,
    seed: int | None,
) -> list[float]:
    """Three different inputs. Equal MAE does not make them the same path."""
    future = [float(value) for value in future_prices]
    if kind == "ideal":
        return future
    if kind == "persistence":
        return [float(origin_price)] * len(future)
    if kind != "mae_matched_noise":
        raise ValueError("kind must be persistence, mae_matched_noise, or ideal")
    if seed is None:
        raise ValueError("mae-matched noise requires a paired seed")
    if target_mae < 0.0 or not len(future):
        raise ValueError("mae-matched noise needs a horizon and a non-negative scale")
    draw = np.random.default_rng(seed).normal(size=len(future))
    scale = float(np.mean(np.abs(draw)))
    if scale == 0.0:
        raise RuntimeError("noise draw has no scale")
    noise = draw * (float(target_mae) / scale)
    return [price + float(shock) for price, shock in zip(future, noise)]


def _support(origins: Iterable) -> tuple[str, ...]:
    stamps = tuple(pd.Timestamp(origin) for origin in origins)
    if not stamps:
        raise ValueError("factorial support is empty")
    if any(stamp.tzinfo is None for stamp in stamps):
        raise ValueError("support timestamps need a timezone")
    normalized = tuple(stamp.tz_convert("UTC") for stamp in stamps)
    if any(stamp >= RESERVED_START_UTC for stamp in normalized):
        raise ValueError("factorial support cannot include a reserved timestamp")
    if len(set(normalized)) != len(normalized):
        raise ValueError("factorial support has duplicate origins")
    if list(normalized) != sorted(normalized):
        raise ValueError("factorial support must be in temporal order")
    return tuple(stamp.isoformat() for stamp in normalized)


def build_manifest(
    origins: Iterable,
    horizons: Sequence[int],
    dev_scales: dict[int, float],
) -> dict:
    """Describe the grid. Does not draw a holdout path and does not backtest."""
    support = _support(origins)
    missing = [hours for hours in horizons if hours not in dev_scales]
    if missing:
        raise ValueError(f"DEV scale missing for horizons {missing}")
    cells = []
    for orientation in ORIENTATIONS:
        for kind in INPUT_KINDS:
            for seed in PAIRED_SEEDS:
                cells.append(
                    {
                        "varying": orientation["varying"],
                        "fixed": orientation["fixed"],
                        "fixed_kind": orientation["fixed_kind"],
                        "input_kind": kind,
                        "seed": None if kind != "mae_matched_noise" else seed,
                        "paired_seed": seed,
                        "support_id": "common",
                        "scale_source": "DEV",
                        "executed": False,
                    }
                )
    return {
        "status": B0_STATUS,
        "executed": False,
        "cell_count": len(cells),
        "orientations": [dict(item) for item in ORIENTATIONS],
        "input_kinds": list(INPUT_KINDS),
        "paired_seeds": list(PAIRED_SEEDS),
        "support_id": "common",
        "support": list(support),
        "horizons_hours": [int(hours) for hours in horizons],
        "dev_scales": {str(hours): float(dev_scales[hours]) for hours in horizons},
        "scale_source": "DEV",
        "holdout_decisions": "OUT_OF_SCOPE",
        "b0_budget": "OUT_OF_SCOPE",
        "historic_cpu_ceiling_seconds": HISTORIC_CPU_CEILING_SECONDS,
        "historic_ceiling_is_b0_budget": HISTORIC_CEILING_IS_B0_BUDGET,
        "retained_sweep_cells_not_run": 241,
        "equal_mae_is_not_the_same_decision": True,
        "parameters_not_chosen_by_pnl": True,
        "cells": cells,
    }


def execute_manifest(manifest: dict) -> None:
    """The runner is present so the harness is closed, and it does not start."""
    raise RuntimeError(
        f"{manifest.get('status', B0_STATUS)}: factorial harness is not executed"
    )
