"""Synthetic short/long sweep. It does not start B0 and it does not rank families.

Noise is added to predictions only. The scale is the DEV mean absolute
residual, applied as sigma = target / sqrt(2/pi). Draws are not divided by
their own mean absolute value. Persistence and ideal are single baselines.
The historic 3600 CPU-second ceiling is not a budget.
"""

from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np
import pandas as pd

from app.account_convention import configure_successor_broker
from app.elapsed_hour_harness import (
    _ObservedStrategy,
    _broker_frame,
    _horizon_errors,
    _iso,
    _label_frame,
    _macro,
    _public_trades,
    _rename_family,
)
from app.plugins.plugin_long_short_predictions import Plugin
from app.strategy_support import (
    B0_STATUS,
    HISTORIC_CPU_CEILING_SECONDS,
    LONG_HORIZONS_HOURS,
    REQUIRED_HORIZONS_HOURS,
    RESERVED_START_UTC,
    RETAINED_SWEEP_CELLS,
    SHORT_HORIZONS_HOURS,
    _normalize_one,
    create_elapsed_hour_predictions,
    fit_development_parameters,
)

INITIAL_CASH = 10_000.0
MARGIN_FRACTION = 0.05
LEVERAGE = 100.0
COMMISSION_PER_LOT = 7.0
GAUSSIAN_ABS_MEAN = math.sqrt(2.0 / math.pi)
# CrispDM wall cap is 120s. This is the grid budget inside that cap, not the
# historic 3600s ceiling and not a B0 allowance.
CELL_WALL_BUDGET_SECONDS = 80.0
HORIZONS = tuple(REQUIRED_HORIZONS_HOURS)
SHORT_HORIZONS = tuple(SHORT_HORIZONS_HOURS)
LONG_HORIZONS = tuple(LONG_HORIZONS_HOURS)
NOISE_LEVELS = (0.5, 1.0, 2.0)
PAIRED_SEEDS = (42, 43, 44)
ORIENTATIONS = (
    {"varying": "short", "fixed": "long"},
    {"varying": "long", "fixed": "short"},
)
FIXED_KINDS = (
    {"kind": "persistence", "role": "primary"},
    {"kind": "ideal", "role": "additional_control"},
)
SCORED_SUPPORT = (
    pd.Timestamp("2019-05-01 00:00:00", tz="UTC"),
    pd.Timestamp("2019-05-01 04:00:00", tz="UTC"),
    pd.Timestamp("2019-05-01 08:00:00", tz="UTC"),
    pd.Timestamp("2019-05-01 09:00:00", tz="UTC"),
)
DEV_FIRST = pd.Timestamp("2019-04-20 00:00:00", tz="UTC")
DEV_LAST = pd.Timestamp("2019-04-30 23:00:00", tz="UTC")
SLICE_NAMES = ("full", "fallback", "minimum", "probe_only")
EQUAL_MAE_NOTE = (
    "Equal MAE does not make persistence and gaussian noise the same control. "
    "Long-family persistence that blocks entry on this fixture is not evidence "
    "that the short family has no use. Financial utility is not claimed."
)
SLICE_RULE = (
    "One probe cell is timed before any further cell. The slice is full, "
    "fallback, minimum, or probe_only from that wall time alone. PnL is not "
    "an input. The historic 3600 CPU-second ceiling is not a budget."
)

_DRAW_COUNT = 0


def synthetic_dev_frame() -> pd.DataFrame:
    """Hourly bars strictly before the scored support. Not a market sample."""
    start = pd.Timestamp("2019-04-20 00:00:00")
    index = pd.DatetimeIndex(start + pd.Timedelta(hours=i) for i in range(11 * 24))
    closes = [
        1.10 + 0.02 * math.sin(i / 3.0) + 0.005 * ((i % 5) - 2) for i in range(len(index))
    ]
    frame = pd.DataFrame({"CLOSE": closes}, index=index)
    frame.index.name = "DATE_TIME"
    return frame


def _origin_key(origin) -> str:
    stamp = pd.Timestamp(origin)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def mix_seed(seed: int, origin, family: str, horizon: int) -> int:
    """Stream id for one origin, family, and horizon. The config seed alone is not enough."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if family not in {"short", "long"}:
        raise ValueError("family must be short or long")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise TypeError("horizon must be a positive int")
    payload = f"{seed}|{_origin_key(origin)}|{family}|{horizon}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def standard_normal_draw(seed: int, origin, family: str, horizon: int) -> float:
    """One N(0, 1) draw. Callers scale it. They do not divide by the sample MAE."""
    global _DRAW_COUNT
    _DRAW_COUNT += 1
    return float(np.random.default_rng(mix_seed(seed, origin, family, horizon)).standard_normal())


def draw_count() -> int:
    return _DRAW_COUNT


def sigma_for_target(target_mae: float) -> float:
    target = float(target_mae)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target MAE must be a non-negative finite number")
    return target / GAUSSIAN_ABS_MEAN


def _intensity_token(intensity) -> str:
    if intensity is None:
        return "na"
    return f"{float(intensity):.1f}".replace(".", "p")


def make_cell_id(
    *,
    varying: str,
    fixed: str,
    fixed_kind: str,
    input_kind: str,
    intensity,
    seed,
) -> str:
    seed_token = "na" if seed is None else str(int(seed))
    return (
        f"{varying}_var_{fixed}_fix_{fixed_kind}_{input_kind}"
        f"_i{_intensity_token(intensity)}_s{seed_token}"
    )


PROBE_CELL_ID = make_cell_id(
    varying="short",
    fixed="long",
    fixed_kind="persistence",
    input_kind="noise",
    intensity=NOISE_LEVELS[1],
    seed=PAIRED_SEEDS[0],
)


def cell_in_slice(cell: dict, name: str) -> bool:
    if name == "full":
        return True
    if name == "probe_only":
        return cell["cell_id"] == PROBE_CELL_ID
    if name == "fallback":
        if cell["input_kind"] != "noise":
            return True
        return cell["seed"] == PAIRED_SEEDS[0]
    if name == "minimum":
        if cell["role"] == "primary" and cell["input_kind"] == "persistence":
            return True
        primary_noise = (
            cell["role"] == "primary"
            and cell["input_kind"] == "noise"
            and cell["seed"] == PAIRED_SEEDS[0]
            and cell["intensity"] == NOISE_LEVELS[1]
        )
        if primary_noise:
            return True
        return (
            cell["varying"] == "short"
            and cell["fixed_kind"] == "ideal"
            and cell["input_kind"] == "noise"
            and cell["seed"] == PAIRED_SEEDS[0]
            and cell["intensity"] == NOISE_LEVELS[1]
        )
    raise ValueError(f"unknown slice {name}")


def choose_slice(probe_wall_seconds: float, manifest: dict) -> str:
    """Pick a slice from the probe wall time. PnL is not an argument."""
    if isinstance(probe_wall_seconds, bool) or not isinstance(
        probe_wall_seconds, (int, float)
    ):
        raise TypeError("probe wall must be a real number")
    wall = float(probe_wall_seconds)
    if not math.isfinite(wall) or wall < 0.0:
        raise ValueError("probe wall must be a non-negative finite number")
    cells = manifest["cells"]
    for name in SLICE_NAMES:
        count = sum(1 for cell in cells if cell_in_slice(cell, name))
        if wall * count <= CELL_WALL_BUDGET_SECONDS:
            return name
    return "probe_only"


def _scale_map(dev_scales: dict) -> dict[str, float]:
    mapped = {str(key): float(value) for key, value in dev_scales.items()}
    missing = [str(hours) for hours in HORIZONS if str(hours) not in mapped]
    if missing:
        raise ValueError(f"DEV scale missing for horizons {missing}")
    for key, value in mapped.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"DEV scale for horizon {key} must be positive")
    return mapped


def _support_index(scored_frame: pd.DataFrame) -> pd.DatetimeIndex:
    label = _label_frame(scored_frame, source_timezone="UTC")
    if label.empty:
        raise ValueError("scored frame has no bars before the reserved cut")
    if label.index.max() >= RESERVED_START_UTC:
        raise RuntimeError("scored frame retained a reserved bar")
    ideal = create_elapsed_hour_predictions(
        label,
        HORIZONS,
        price_column="CLOSE",
        source_timezone="UTC",
    )
    got = tuple(pd.Timestamp(stamp) for stamp in ideal.index)
    if got != SCORED_SUPPORT:
        raise ValueError("scored frame does not produce the declared common support")
    consumed = tuple(ideal.attrs["consumed_timestamps_utc"])
    if any(stamp >= RESERVED_START_UTC for stamp in consumed):
        raise RuntimeError("scored predictions consumed a reserved timestamp")
    return ideal.index


def build_manifest(dev_frame: pd.DataFrame, scored_frame: pd.DataFrame) -> dict:
    """Declare the protocol. Does not backtest and does not start B0."""
    _support_index(scored_frame)
    dev_stamps = []
    for stamp in dev_frame.index:
        clock = pd.Timestamp(stamp)
        if clock.tzinfo is None:
            clock = clock.tz_localize("UTC")
        else:
            clock = clock.tz_convert("UTC")
        dev_stamps.append(clock)
    dev_index = pd.DatetimeIndex(dev_stamps)
    if any(stamp >= SCORED_SUPPORT[0] for stamp in dev_index):
        raise ValueError("DEV frame is not strictly before the scored support")
    if any(stamp >= RESERVED_START_UTC for stamp in dev_index):
        raise ValueError("DEV frame includes a reserved bar")
    if any(stamp in set(SCORED_SUPPORT) for stamp in dev_index):
        raise ValueError("DEV frame includes a scored origin")
    fitted = fit_development_parameters(
        dev_frame,
        HORIZONS,
        source_timezone="UTC",
        price_column="CLOSE",
    )
    if fitted.fits_noise_model:
        raise RuntimeError("DEV scale must not fit a noise model")
    consumed = tuple(pd.Timestamp(stamp) for stamp in fitted.consumed_timestamps_utc)
    if not consumed:
        raise RuntimeError("DEV scale consumed no timestamps")
    if any(stamp >= SCORED_SUPPORT[0] for stamp in consumed):
        raise RuntimeError("DEV scale consumed a scored-window timestamp")
    if any(stamp >= RESERVED_START_UTC for stamp in consumed):
        raise RuntimeError("DEV scale consumed a reserved timestamp")
    dev_scales = _scale_map(dict(fitted.per_horizon_mean_abs_residual))
    cells = []
    for orientation in ORIENTATIONS:
        for fixed in FIXED_KINDS:
            specs = [("persistence", None, None), ("ideal", None, None)]
            specs.extend(
                ("noise", intensity, seed)
                for intensity in NOISE_LEVELS
                for seed in PAIRED_SEEDS
            )
            for kind, intensity, seed in specs:
                cells.append(
                    {
                        "cell_id": make_cell_id(
                            varying=orientation["varying"],
                            fixed=orientation["fixed"],
                            fixed_kind=fixed["kind"],
                            input_kind=kind,
                            intensity=intensity,
                            seed=seed,
                        ),
                        "varying": orientation["varying"],
                        "fixed": orientation["fixed"],
                        "fixed_kind": fixed["kind"],
                        "role": fixed["role"],
                        "input_kind": kind,
                        "intensity": intensity,
                        "seed": seed,
                        "executed": False,
                    }
                )
    ids = [cell["cell_id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate cell id")
    if PROBE_CELL_ID not in set(ids):
        raise RuntimeError("probe cell is not in the manifest")
    return {
        "label": "SYNTHETIC",
        "status": "DECLARED",
        "protocol_declared_before_execution": True,
        "b0_status": B0_STATUS,
        "b0_started": False,
        "financial_utility": "NOT_CLAIMED",
        "short_versus_long_utility": "NOT_CLAIMED",
        "parameters_not_chosen_by_pnl": True,
        "equal_mae_note": EQUAL_MAE_NOTE,
        "noise_levels": list(NOISE_LEVELS),
        "paired_seeds": list(PAIRED_SEEDS),
        "orientations": [dict(item) for item in ORIENTATIONS],
        "fixed_kinds": [dict(item) for item in FIXED_KINDS],
        "input_kinds": ["persistence", "ideal", "noise"],
        "deterministic_baselines_are_not_replicated": True,
        "support_id": "micro_2019-05-01_four_origins",
        "support": [_iso(stamp) for stamp in SCORED_SUPPORT],
        "support_source": "synthetic_elapsed_hour_fixture",
        "horizons_hours": list(HORIZONS),
        "horizon_protocol": "twelve_elapsed_hours",
        "horizon_protocol_is_complete": True,
        "scale_source": "DEV",
        "dev_source": "synthetic_hourly_before_scored_support",
        "dev_scales": dev_scales,
        "dev_origin_count": len(fitted.admitted_origins),
        "dev_consumed_count": len(consumed),
        "dev_first_utc": _iso(min(consumed)),
        "dev_last_utc": _iso(max(consumed)),
        "noise_model": {
            "distribution": "gaussian_iid",
            "applied_to": "prediction",
            "not_applied_to": ["broker_actions", "costs", "fills"],
            "base": "elapsed_hour_future_price",
            "target_mae_formula": "intensity * dev_mean_abs_residual",
            "sigma_formula": "target_mae / sqrt(2/pi)",
            "gaussian_abs_mean": GAUSSIAN_ABS_MEAN,
            "gaussian_abs_mean_is_a_constant": True,
            "sample_mae_renormalized": False,
            "seed_components": ["config_seed", "origin_utc", "family", "horizon_hours"],
            "same_config_seed_repeats_across_origins": False,
            "draws_paired_across_intensities": True,
        },
        "slice_rule": SLICE_RULE,
        "cell_wall_budget_seconds": CELL_WALL_BUDGET_SECONDS,
        "crispdm_wall_ceiling_seconds": 120,
        "historic_cpu_ceiling_seconds": int(HISTORIC_CPU_CEILING_SECONDS),
        "historic_ceiling_is_b0_budget": False,
        "historic_ceiling_is_this_budget": False,
        "retained_sweep_cells_not_run": int(RETAINED_SWEEP_CELLS),
        "ratios_21_not_run": True,
        "market_b0_not_run": True,
        "real_market": "NOT_RUN",
        "gpu": "NOT_RUN",
        "database_service": "NOT_RUN",
        "live_broker": "NOT_RUN",
        "holdout_decisions": "OUT_OF_SCOPE",
        "cell_count": len(cells),
        "cells": cells,
    }


def _family_kind(cell: dict, family: str) -> str:
    if family == cell["varying"]:
        return cell["input_kind"]
    if family == cell["fixed"]:
        return cell["fixed_kind"]
    raise ValueError("family is neither the varying nor the fixed side")


def _build_family_frame(
    ideal: pd.DataFrame,
    label: pd.DataFrame,
    horizons: tuple[int, ...],
    kind: str,
    *,
    family: str,
    intensity,
    seed,
    dev_scales: dict[str, float],
) -> pd.DataFrame:
    columns = [f"elapsed_{hours}h" for hours in horizons]
    if kind == "ideal":
        return ideal.loc[:, columns].copy()
    if kind == "persistence":
        origin = label["CLOSE"].reindex(ideal.index)
        values = [float(value) for value in origin.tolist()]
        return pd.DataFrame({column: values for column in columns}, index=ideal.index)
    if kind != "noise":
        raise ValueError("family kind must be persistence, ideal, or noise")
    if seed is None or intensity is None:
        raise ValueError("noise requires a paired seed and an intensity")
    frame = ideal.loc[:, columns].astype(float).copy()
    for origin in frame.index:
        for hours, column in zip(horizons, columns):
            target = float(intensity) * dev_scales[str(hours)]
            shock = sigma_for_target(target) * standard_normal_draw(
                int(seed), origin, family, int(hours)
            )
            frame.at[origin, column] = float(frame.at[origin, column]) + shock
    return frame


def prepare_cell(cell: dict, *, dev_scales: dict, scored_frame: pd.DataFrame) -> dict:
    """Build one cell's predictions and its MAE. Does not call the broker."""
    scales = _scale_map(dev_scales)
    label = _label_frame(scored_frame, source_timezone="UTC")
    ideal = create_elapsed_hour_predictions(
        label,
        HORIZONS,
        price_column="CLOSE",
        source_timezone="UTC",
    )
    if tuple(pd.Timestamp(stamp) for stamp in ideal.index) != SCORED_SUPPORT:
        raise ValueError("scored frame does not produce the declared common support")
    short_frame = _build_family_frame(
        ideal,
        label,
        SHORT_HORIZONS,
        _family_kind(cell, "short"),
        family="short",
        intensity=cell["intensity"],
        seed=cell["seed"],
        dev_scales=scales,
    )
    long_frame = _build_family_frame(
        ideal,
        label,
        LONG_HORIZONS,
        _family_kind(cell, "long"),
        family="long",
        intensity=cell["intensity"],
        seed=cell["seed"],
        dev_scales=scales,
    )
    elapsed = pd.concat([short_frame, long_frame], axis=1)
    mae, naive = _horizon_errors(label, elapsed, HORIZONS, price_column="CLOSE")
    targets: dict[str, float | None] = {}
    for hours in HORIZONS:
        family = "short" if hours in SHORT_HORIZONS else "long"
        kind = _family_kind(cell, family)
        key = str(hours)
        if kind == "noise":
            targets[key] = float(cell["intensity"]) * scales[key]
        elif kind == "ideal":
            targets[key] = 0.0
        elif kind == "persistence":
            targets[key] = None
        else:
            raise ValueError(kind)
    gaps = {
        key: None if target is None else float(mae[key] - target)
        for key, target in targets.items()
    }
    numeric = [value for value in targets.values() if value is not None]
    targeted_keys = [key for key, value in targets.items() if value is not None]
    targeted_achieved = (
        None
        if not targeted_keys
        else float(sum(mae[key] for key in targeted_keys) / len(targeted_keys))
    )
    hourly = _rename_family(short_frame, SHORT_HORIZONS, "Prediction_h_")
    daily = _rename_family(long_frame, LONG_HORIZONS, "Prediction_d_")
    plugin = hourly.join(daily, how="inner")
    plugin.index = plugin.index.tz_convert("UTC").tz_localize(None)
    plugin.index.name = "DATE_TIME"
    return {
        "elapsed_frame": elapsed,
        "plugin_frame": plugin,
        "per_horizon_mae": mae,
        "per_horizon_naive_mae": naive,
        "per_horizon_target_mae": targets,
        "per_horizon_achieved_minus_target": gaps,
        "per_horizon_dev_scale": {str(hours): scales[str(hours)] for hours in HORIZONS},
        "mae_macro_average": _macro(mae),
        "naive_macro_average": _macro(naive),
        "target_macro_average": (
            None if not numeric else float(sum(numeric) / len(numeric))
        ),
        "targeted_horizon_achieved_macro": targeted_achieved,
        "target_horizon_count": len(numeric),
        "sample_mae_renormalized": False,
    }


class _SweepStrategy(_ObservedStrategy):
    """Same decisions as the plugin. Records the bar around each fill."""

    def _should_early_close_long(self, *args):
        self._early_close_evaluated = True
        triggered = super()._should_early_close_long(*args)
        self._early_close_triggered = bool(triggered)
        return triggered

    def _should_early_close_short(self, *args):
        self._early_close_evaluated = True
        triggered = super()._should_early_close_short(*args)
        self._early_close_triggered = bool(triggered)
        return triggered

    def next(self):
        self._early_close_triggered = False
        self._early_close_evaluated = False
        direction = self.current_direction
        price = float(self.data0.close[0])
        tp, sl = self.current_tp, self.current_sl
        super().next()
        row = self.events[-1]
        cause = None
        if row["request"] == "close":
            if direction == "long":
                cause = "take_profit" if price >= tp else "stop_loss" if price <= sl else "early_prediction"
            elif direction == "short":
                cause = "take_profit" if price <= tp else "stop_loss" if price >= sl else "early_prediction"
            else:
                raise RuntimeError("close request without a direction")
            if (cause == "early_prediction") != self._early_close_triggered:
                raise RuntimeError("close cause disagrees with variant E evaluation")
        row["close_cause"] = cause
        row["direction_before"] = direction
        row["take_profit_before"] = None if tp is None else float(tp)
        row["stop_loss_before"] = None if sl is None else float(sl)
        row["early_close_evaluated"] = self._early_close_evaluated
        row["early_close_triggered"] = self._early_close_triggered
        row["equity"] = float(self.broker.getvalue())
        row["cash"] = float(self.broker.getcash())
        row["open_exposure_units"] = float(self.position.size)

    def notify_order(self, order):
        super().notify_order(order)
        if order.status == order.Completed and self.fill_rows:
            row = self.fill_rows[-1]
            row["bar_open"] = float(self.data0.open[0])
            row["bar_high"] = float(self.data0.high[0])
            row["bar_low"] = float(self.data0.low[0])
            row["requested_size"] = float(order.size or 0.0)


def _timer_start() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


def _timer_stop(started: tuple[float, float]) -> dict[str, float]:
    wall0, cpu0 = started
    return {
        "wall_seconds": time.perf_counter() - wall0,
        "cpu_seconds": time.process_time() - cpu0,
    }


def _plugin_params() -> dict:
    params = Plugin().params
    if params["exit_variant"] != "E":
        raise RuntimeError("plugin baseline is not variant E")
    if params["tp_multiplier"] != 0.9 or params["sl_multiplier"] != 2.0:
        raise RuntimeError("refusing to change TP or SL multipliers")
    if float(params["margin_fraction"]) != MARGIN_FRACTION:
        raise RuntimeError("refusing to change the margin fraction")
    if float(params["leverage"]) != LEVERAGE:
        raise RuntimeError("refusing to change leverage")
    if float(params["min_order_volume"]) != 10_000.0:
        raise RuntimeError("refusing to change the minimum order")
    if float(params["commission_per_lot"]) != COMMISSION_PER_LOT:
        raise RuntimeError("refusing to change commission")
    return params


def run_book(
    bars: pd.DataFrame,
    plugin_frame: pd.DataFrame,
    work_directory: str,
    *,
    session_end=None,
) -> dict:
    """Run the real plugin on a prepared prediction frame. Does not draw noise."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import backtrader as bt

    import app.plugins.plugin_long_short_predictions as plugin_module

    plugin_module._QUIET = True
    params = _plugin_params()
    label = _label_frame(bars, source_timezone="UTC")
    if session_end is None:
        broker_label = label
        session_iso = None
    else:
        end = _normalize_one(session_end, source_timezone="UTC")
        if end >= RESERVED_START_UTC:
            raise ValueError("broker session end is on or after the reserved cut")
        broker_label = label.loc[label.index <= end]
        session_iso = _iso(end)
        if broker_label.empty:
            raise ValueError("broker session is empty")
    os.makedirs(work_directory, exist_ok=True)
    table = plugin_frame.copy()
    table.index.name = "DATE_TIME"
    prediction_path = os.path.join(work_directory, "predictions.csv")
    table.reset_index().to_csv(prediction_path, index=False)
    rate = float(params["commission_per_lot"]) / 100_000.0
    slip = (float(params["spread_pips"]) + float(params["slippage_pips"])) * float(
        params["pip_cost"]
    ) / 2.0
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        _SweepStrategy,
        pred_file=prediction_path,
        pip_cost=params["pip_cost"],
        rel_volume=params["rel_volume"],
        min_order_volume=params["min_order_volume"],
        max_order_volume=params["max_order_volume"],
        leverage=LEVERAGE,
        profit_threshold=params["profit_threshold"],
        min_drawdown_pips=params["min_drawdown_pips"],
        tp_multiplier=params["tp_multiplier"],
        sl_multiplier=params["sl_multiplier"],
        lower_rr_threshold=params["lower_rr_threshold"],
        upper_rr_threshold=params["upper_rr_threshold"],
        max_trades_per_5days=params["max_trades_per_5days"],
        exit_variant="E",
        swap_per_lot_per_day=params["swap_per_lot_per_day"],
        accounting_convention="successor",
        margin_fraction=MARGIN_FRACTION,
        commission_per_unit=rate,
        spread_pips=params["spread_pips"],
        slippage_pips=params["slippage_pips"],
    )
    cerebro.adddata(bt.feeds.PandasData(dataname=_broker_frame(broker_label)))
    broker = configure_successor_broker(
        cerebro,
        cash=INITIAL_CASH,
        leverage=LEVERAGE,
        margin_fraction=MARGIN_FRACTION,
        commission_rate=rate,
        min_order_volume=float(params["min_order_volume"]),
        slippage_per_side=slip,
    )
    previous = os.getcwd()
    os.chdir(work_directory)
    try:
        strategy = cerebro.run()[0]
    finally:
        os.chdir(previous)
    if strategy.accounting_snapshot is None:
        raise RuntimeError("successor book did not report")
    trades = _public_trades(strategy.trades)
    snapshot = strategy.accounting_snapshot
    return {
        "label": "SYNTHETIC",
        "financial_utility": "NOT_CLAIMED",
        "accounting_convention": "successor",
        "exit_variant": "E",
        "tp_multiplier": float(params["tp_multiplier"]),
        "sl_multiplier": float(params["sl_multiplier"]),
        "initial_cash": INITIAL_CASH,
        "margin_fraction": MARGIN_FRACTION,
        "leverage": LEVERAGE,
        "shortcash": bool(broker.p.shortcash),
        "commission_rate": rate,
        "slippage_per_side": slip,
        "swap_per_lot_per_day": float(params["swap_per_lot_per_day"]),
        "spread_pips": float(params["spread_pips"]),
        "slippage_pips": float(params["slippage_pips"]),
        "broker_session_end": session_iso,
        "broker_bar_count": int(len(broker_label)),
        "cash": float(strategy.cash_before_stop),
        "marked_equity": float(strategy.marked_equity_before_stop),
        "accounting_cash": float(snapshot["cash"]),
        "accounting_equity": float(snapshot["equity"]),
        "realized_pnl": float(sum(trade["pnl"] for trade in trades)),
        "n_closed_trades": len(trades),
        "pending_position_units": float(strategy.position_after_stop),
        "position_before_stop": float(strategy.position_before_stop),
        "position_after_stop": float(strategy.position_after_stop),
        "closed_trades_before_stop": int(strategy.closed_trades_before_stop),
        "closed_trades_after_stop": int(strategy.closed_trades_after_stop),
        "stop_close_obtained_fill": bool(strategy.stop_close_obtained_fill),
        "current_direction": strategy.current_direction,
        "entry_sides": [
            "long" if event["request"] == "buy" else "short"
            for event in strategy.events
            if event["request"] in {"buy", "sell"}
        ],
        "trades": trades,
        "fills": list(strategy.fill_rows),
        "events": list(strategy.events),
        "ledger": list(strategy.ledger),
        "cap_events": list(snapshot["cap_events"]),
        "order_states": [event["state"] for event in strategy.order_events],
        "book": dict(snapshot["book"]),
    }


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("non-finite measured number")
        return number
    raise RuntimeError(f"unexpected measured type {type(value).__name__}")


def run_cell(
    cell: dict,
    *,
    dev_scales: dict,
    scored_frame: pd.DataFrame,
    work_directory: str,
    session_end=None,
) -> dict:
    started = _timer_start()
    prepared = prepare_cell(cell, dev_scales=dev_scales, scored_frame=scored_frame)
    measured = run_book(
        scored_frame,
        prepared["plugin_frame"],
        work_directory,
        session_end=session_end,
    )
    measured.update(_timer_stop(started))
    for key in (
        "cell_id",
        "varying",
        "fixed",
        "fixed_kind",
        "role",
        "input_kind",
        "intensity",
        "seed",
    ):
        measured[key] = cell[key]
    for key in (
        "per_horizon_mae",
        "per_horizon_naive_mae",
        "per_horizon_target_mae",
        "per_horizon_achieved_minus_target",
        "per_horizon_dev_scale",
        "mae_macro_average",
        "naive_macro_average",
        "target_macro_average",
        "targeted_horizon_achieved_macro",
        "target_horizon_count",
        "sample_mae_renormalized",
    ):
        measured[key] = prepared[key]
    measured["support"] = [_iso(stamp) for stamp in SCORED_SUPPORT]
    return _plain(measured)


def _warmup(work_directory: str) -> dict:
    """One tiny book so the probe clock is not the import. Not a declared cell."""
    index = pd.to_datetime(["2019-04-01 00:00:00", "2019-04-01 01:00:00"])
    bars = pd.DataFrame(
        {"OPEN": 1.0, "HIGH": 1.0, "LOW": 1.0, "CLOSE": 1.0},
        index=index,
    )
    columns = [f"Prediction_h_{i}" for i in range(1, 7)] + [
        f"Prediction_d_{i}" for i in range(1, 7)
    ]
    frame = pd.DataFrame([[1.0] * len(columns)], index=index[:1], columns=columns)
    frame.index.name = "DATE_TIME"
    started = _timer_start()
    run_book(bars, frame, work_directory)
    timing = _timer_stop(started)
    timing["scored"] = False
    timing["is_cell"] = False
    return timing


def _require_cell(manifest: dict, cell_id: str) -> dict:
    found = [cell for cell in manifest["cells"] if cell["cell_id"] == cell_id]
    if len(found) != 1:
        raise RuntimeError(f"cell {cell_id} is not declared once")
    return found[0]


def execute_declared(manifest: dict, scored_frame: pd.DataFrame, work_root: str) -> dict:
    """Time one cell, then run only the slice that fits the wall budget."""
    if manifest.get("b0_status") != B0_STATUS or manifest.get("b0_started"):
        raise RuntimeError("sweep executor refuses to start B0")
    if not manifest.get("protocol_declared_before_execution"):
        raise RuntimeError("protocol was not declared before execution")
    if any(cell["executed"] for cell in manifest["cells"]):
        raise RuntimeError("manifest cells are already marked executed")
    root = str(work_root)
    os.makedirs(root, exist_ok=True)
    started = _timer_start()
    warmup = _warmup(os.path.join(root, "_warmup"))
    probe = _require_cell(manifest, PROBE_CELL_ID)
    by_id = {
        probe["cell_id"]: run_cell(
            probe,
            dev_scales=manifest["dev_scales"],
            scored_frame=scored_frame,
            work_directory=os.path.join(root, probe["cell_id"]),
        )
    }
    probe["executed"] = True
    choice = choose_slice(by_id[probe["cell_id"]]["wall_seconds"], manifest)
    for cell in manifest["cells"]:
        if cell["cell_id"] in by_id or not cell_in_slice(cell, choice):
            continue
        by_id[cell["cell_id"]] = run_cell(
            cell,
            dev_scales=manifest["dev_scales"],
            scored_frame=scored_frame,
            work_directory=os.path.join(root, cell["cell_id"]),
        )
        cell["executed"] = True
    ordered = [by_id[cell["cell_id"]] for cell in manifest["cells"] if cell["executed"]]
    timing = _timer_stop(started)
    return _plain(
        {
            "slice": choice,
            "slice_rule": SLICE_RULE,
            "cell_wall_budget_seconds": CELL_WALL_BUDGET_SECONDS,
            "one_cell": {
                "cell_id": probe["cell_id"],
                "wall_seconds": by_id[probe["cell_id"]]["wall_seconds"],
                "cpu_seconds": by_id[probe["cell_id"]]["cpu_seconds"],
            },
            "warmup": warmup,
            "executed_cell_count": len(ordered),
            "declared_cell_count": len(manifest["cells"]),
            "results": ordered,
            "wall_seconds": timing["wall_seconds"],
            "cpu_seconds": timing["cpu_seconds"],
            "b0_status": B0_STATUS,
            "financial_utility": "NOT_CLAIMED",
        }
    )
