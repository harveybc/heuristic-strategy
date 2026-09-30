"""Synthetic sweep executor. Does not start B0 and does not rewrite prior evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.account_convention import margin_required
from app.strategy_support import fit_development_parameters
from app.sweep_executor import (
    CELL_WALL_BUDGET_SECONDS,
    GAUSSIAN_ABS_MEAN,
    HORIZONS,
    INITIAL_CASH,
    LONG_HORIZONS,
    NOISE_LEVELS,
    PAIRED_SEEDS,
    PROBE_CELL_ID,
    SCORED_SUPPORT,
    SHORT_HORIZONS,
    build_manifest,
    cell_in_slice,
    choose_slice,
    draw_count,
    execute_declared,
    make_cell_id,
    mix_seed,
    prepare_cell,
    run_book,
    run_cell,
    sigma_for_target,
    standard_normal_draw,
    synthetic_dev_frame,
)
from tests.unit_tests.test_strategy_micro_20260930 import _frame, _micro_bars, _ohlc

RATE = 0.00007
SLIP = 0.000015
EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "audits"
    / "evidence"
    / "STRATEGY_SWEEP_20260930"
)


def _bars():
    return _micro_bars()


def _manifest():
    return build_manifest(synthetic_dev_frame(), _bars())


def _cell(manifest, **wanted):
    found = [
        cell
        for cell in manifest["cells"]
        if all(cell[key] == value for key, value in wanted.items())
    ]
    assert len(found) == 1
    return found[0]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError("non-finite measured number")
        return value
    raise AssertionError(f"unexpected measured type {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    lowered = encoded.lower()
    for token in ("/home", "harvey", "hostname", "postgres", "secret"):
        assert token not in lowered
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def _assert_reconciled(result):
    book = result["book"]
    assert result["initial_cash"] == INITIAL_CASH
    assert result["shortcash"] is False
    assert result["margin_fraction"] == pytest.approx(0.05)
    assert result["leverage"] == pytest.approx(100.0)
    assert result["exit_variant"] == "E"
    assert result["tp_multiplier"] == pytest.approx(0.9)
    assert result["sl_multiplier"] == pytest.approx(2.0)
    assert result["commission_rate"] == pytest.approx(RATE)
    assert result["slippage_per_side"] == pytest.approx(SLIP)
    assert result["accounting_cash"] == pytest.approx(result["cash"], abs=1e-6)
    assert result["accounting_equity"] == pytest.approx(result["marked_equity"], abs=1e-6)
    assert book["equity_gap"] == pytest.approx(0.0, abs=1e-6)
    assert result["marked_equity"] == pytest.approx(book["equity_from_parts"], abs=1e-6)
    assert result["stop_close_obtained_fill"] is False
    assert result["closed_trades_before_stop"] == result["closed_trades_after_stop"]
    assert result["position_before_stop"] == pytest.approx(result["position_after_stop"])
    assert result["pending_position_units"] == pytest.approx(result["position_after_stop"])
    if book["flat"]:
        assert result["pending_position_units"] == pytest.approx(0.0)
        assert book["flat_cash_gap"] == pytest.approx(0.0, abs=1e-6)
        assert result["cash"] == pytest.approx(
            result["initial_cash"] + result["realized_pnl"], abs=1e-6
        )
        assert result["cash"] == pytest.approx(result["marked_equity"], abs=1e-6)
    else:
        assert result["pending_position_units"] != pytest.approx(0.0)
    commission = [row for row in result["ledger"] if row["kind"] == "commission"]
    swaps = [row for row in result["ledger"] if row["kind"] == "swap"]
    if book["flat"]:
        assert sum(row["amount"] for row in commission) == pytest.approx(
            sum(trade["commission"] for trade in result["trades"])
        )
        assert sum(row["amount"] for row in swaps) == pytest.approx(
            sum(trade["swap"] for trade in result["trades"])
        )
    for row in result["ledger"]:
        if row["kind"] == "commission":
            assert row["amount"] == pytest.approx(abs(row["units"]) * RATE * row["price"])
            assert row["debited_to_cash"] is True
        if row["kind"] in {"spread", "slippage"}:
            assert row["debited_to_cash"] is False
            assert row["included_in_fill_price"] is True
        if row["kind"] == "swap":
            assert row["debited_to_cash"] is True
            assert row["cash_after"] == pytest.approx(row["cash_before"] - row["amount"])
    for trade in result["trades"]:
        assert trade["pnl"] == pytest.approx(
            trade["gross_pnl"] - trade["commission"] - trade["swap"]
        )
    for fill in result["fills"]:
        delta = fill["price"] - fill["bar_open"]
        assert abs(delta) <= SLIP + 1e-9
        if fill["is_buy"]:
            assert delta >= -1e-12
        else:
            assert delta <= 1e-12
        assert abs(fill["size"]) != pytest.approx(1_000_000.0)
        assert margin_required(fill["size"], fill["price"], 100.0) > 0.0
    for event in result["cap_events"]:
        if "margin_used" in event:
            assert event["margin_used"] <= event["margin_budget"] + 1e-6
    assert result["sample_mae_renormalized"] is False
    assert result["financial_utility"] == "NOT_CLAIMED"


def test_manifest_declares_the_protocol_before_any_cell():
    dev = synthetic_dev_frame()
    manifest = build_manifest(dev, _bars())
    assert manifest["protocol_declared_before_execution"] is True
    assert manifest["b0_status"] == "B0_NOT_STARTED"
    assert manifest["b0_started"] is False
    assert manifest["financial_utility"] == "NOT_CLAIMED"
    assert manifest["short_versus_long_utility"] == "NOT_CLAIMED"
    assert manifest["parameters_not_chosen_by_pnl"] is True
    assert manifest["historic_ceiling_is_b0_budget"] is False
    assert manifest["historic_ceiling_is_this_budget"] is False
    assert manifest["retained_sweep_cells_not_run"] == 241
    assert manifest["ratios_21_not_run"] is True
    assert manifest["horizon_protocol_is_complete"] is True
    assert manifest["horizons_hours"] == list(HORIZONS)
    assert manifest["noise_levels"] == list(NOISE_LEVELS)
    assert manifest["paired_seeds"] == list(PAIRED_SEEDS)
    assert manifest["support"] == [
        "2019-05-01T00:00:00+00:00",
        "2019-05-01T04:00:00+00:00",
        "2019-05-01T08:00:00+00:00",
        "2019-05-01T09:00:00+00:00",
    ]
    assert manifest["noise_model"]["sample_mae_renormalized"] is False
    assert manifest["noise_model"]["applied_to"] == "prediction"
    assert manifest["noise_model"]["same_config_seed_repeats_across_origins"] is False
    assert manifest["cell_count"] == 44
    assert all(cell["executed"] is False for cell in manifest["cells"])
    deterministic = [
        cell for cell in manifest["cells"] if cell["input_kind"] in {"persistence", "ideal"}
    ]
    noise = [cell for cell in manifest["cells"] if cell["input_kind"] == "noise"]
    assert len(deterministic) == 8
    assert len(noise) == 36
    assert all(cell["seed"] is None and cell["intensity"] is None for cell in deterministic)
    assert {cell["seed"] for cell in noise} == set(PAIRED_SEEDS)
    assert {cell["intensity"] for cell in noise} == set(NOISE_LEVELS)
    assert {cell["fixed_kind"] for cell in manifest["cells"] if cell["role"] == "primary"} == {
        "persistence"
    }
    assert {
        cell["fixed_kind"]
        for cell in manifest["cells"]
        if cell["role"] == "additional_control"
    } == {"ideal"}
    assert {cell["varying"] for cell in manifest["cells"]} == {"short", "long"}
    assert sum(cell_in_slice(cell, "full") for cell in manifest["cells"]) == 44
    assert sum(cell_in_slice(cell, "fallback") for cell in manifest["cells"]) == 20
    assert sum(cell_in_slice(cell, "minimum") for cell in manifest["cells"]) == 5
    assert sum(cell_in_slice(cell, "probe_only") for cell in manifest["cells"]) == 1
    assert choose_slice(0.1, manifest) == "full"
    assert choose_slice(2.0, manifest) == "fallback"
    assert choose_slice(10.0, manifest) == "minimum"
    assert choose_slice(100.0, manifest) == "probe_only"
    assert CELL_WALL_BUDGET_SECONDS < 120.0
    with pytest.raises(ValueError):
        build_manifest(
            dev,
            _frame(
                [
                    _ohlc("2019-05-16 00:00", 1.0, 1.0),
                    _ohlc("2019-05-16 01:00", 1.0, 1.0),
                ]
            ),
        )


def test_dev_scale_excludes_the_scored_rows():
    dev = synthetic_dev_frame()
    scored = _bars()
    manifest = build_manifest(dev, scored)
    refit = fit_development_parameters(dev, HORIZONS, source_timezone="UTC")
    assert manifest["dev_scales"] == {
        str(hours): value for hours, value in refit.per_horizon_mean_abs_residual
    }
    assert manifest["dev_last_utc"] < "2019-05-01T00:00:00+00:00"
    assert manifest["dev_origin_count"] > 0
    moved = scored.copy()
    moved.iloc[0, moved.columns.get_loc("CLOSE")] = 1.5
    reserved = scored.copy()
    reserved.iloc[-1, reserved.columns.get_loc("CLOSE")] = 5.0
    assert build_manifest(dev, moved)["dev_scales"] == manifest["dev_scales"]
    assert build_manifest(dev, reserved)["dev_scales"] == manifest["dev_scales"]
    changed = dev.copy()
    changed.iloc[0, 0] = 1.5
    assert build_manifest(changed, scored)["dev_scales"] != manifest["dev_scales"]


def test_noise_is_reproducible_per_origin_and_not_renormalized():
    origin = SCORED_SUPPORT[0]
    other = SCORED_SUPPORT[1]
    assert mix_seed(42, origin, "short", 1) == mix_seed(42, origin, "short", 1)
    streams = {mix_seed(42, origin, "short", hours) for hours in HORIZONS}
    assert len(streams) == len(HORIZONS)
    assert streams.isdisjoint(mix_seed(42, other, "short", hours) for hours in HORIZONS)
    assert mix_seed(42, origin, "short", 1) != mix_seed(42, origin, "long", 1)
    first = [standard_normal_draw(42, origin, "short", hours) for hours in SHORT_HORIZONS]
    again = [standard_normal_draw(42, origin, "short", hours) for hours in SHORT_HORIZONS]
    other_draw = [standard_normal_draw(42, other, "short", hours) for hours in SHORT_HORIZONS]
    assert first == again
    assert first != other_draw
    scale = 0.01
    shock = sigma_for_target(scale) * first[0]
    doubled = sigma_for_target(2.0 * scale) * first[0]
    assert doubled == pytest.approx(2.0 * shock)
    assert sigma_for_target(scale) == pytest.approx(scale / GAUSSIAN_ABS_MEAN)
    manifest = _manifest()
    cell = _cell(
        manifest,
        varying="short",
        fixed_kind="persistence",
        input_kind="noise",
        intensity=1.0,
        seed=42,
    )
    louder = _cell(
        manifest,
        varying="short",
        fixed_kind="persistence",
        input_kind="noise",
        intensity=2.0,
        seed=42,
    )
    ideal = _cell(
        manifest, varying="short", fixed_kind="persistence", input_kind="ideal"
    )
    before = draw_count()
    quiet = prepare_cell(ideal, dev_scales=manifest["dev_scales"], scored_frame=_bars())
    assert draw_count() == before
    one = prepare_cell(cell, dev_scales=manifest["dev_scales"], scored_frame=_bars())
    two = prepare_cell(cell, dev_scales=manifest["dev_scales"], scored_frame=_bars())
    high = prepare_cell(louder, dev_scales=manifest["dev_scales"], scored_frame=_bars())
    pd.testing.assert_frame_equal(one["elapsed_frame"], two["elapsed_frame"])
    short_columns = [f"elapsed_{hours}h" for hours in SHORT_HORIZONS]
    long_columns = [f"elapsed_{hours}h" for hours in LONG_HORIZONS]
    delta = one["elapsed_frame"][short_columns] - quiet["elapsed_frame"][short_columns]
    delta_high = high["elapsed_frame"][short_columns] - quiet["elapsed_frame"][short_columns]
    pd.testing.assert_frame_equal(delta_high, delta * 2.0)
    pd.testing.assert_frame_equal(
        one["elapsed_frame"][long_columns], high["elapsed_frame"][long_columns]
    )
    paired = _cell(
        manifest,
        varying="short",
        fixed_kind="ideal",
        input_kind="noise",
        intensity=1.0,
        seed=42,
    )
    other_fixed = prepare_cell(
        paired, dev_scales=manifest["dev_scales"], scored_frame=_bars()
    )
    pd.testing.assert_frame_equal(
        one["elapsed_frame"][short_columns], other_fixed["elapsed_frame"][short_columns]
    )
    assert not one["elapsed_frame"][long_columns].equals(other_fixed["elapsed_frame"][long_columns])
    row_draws = np.array(
        [
            sigma_for_target(manifest["dev_scales"][str(hours)])
            * standard_normal_draw(42, origin, "short", hours)
            for hours in SHORT_HORIZONS
        ]
    )
    forced = row_draws * (row_draws[0] / np.mean(np.abs(row_draws)))
    assert not np.allclose(row_draws, forced)
    bare = np.random.default_rng(42).normal(size=len(SHORT_HORIZONS))
    bare = bare * (0.01 / np.mean(np.abs(bare)))
    assert not np.allclose(first, bare)
    label_gap = one["per_horizon_achieved_minus_target"]
    assert any(value is not None and abs(value) > 1e-12 for value in label_gap.values())
    for hours in SHORT_HORIZONS:
        key = str(hours)
        target = one["per_horizon_target_mae"][key]
        achieved = one["per_horizon_mae"][key]
        assert target == pytest.approx(manifest["dev_scales"][key])
        assert label_gap[key] == pytest.approx(achieved - target)
    for hours in LONG_HORIZONS:
        assert one["per_horizon_target_mae"][str(hours)] is None
        assert one["per_horizon_mae"][str(hours)] == pytest.approx(
            one["per_horizon_naive_mae"][str(hours)]
        )
    assert quiet["per_horizon_mae"]["1"] == pytest.approx(0.0)
    assert ideal["seed"] is None
    assert make_cell_id(
        varying="short",
        fixed="long",
        fixed_kind="persistence",
        input_kind="noise",
        intensity=1.0,
        seed=42,
    ) == PROBE_CELL_ID


def test_broker_run_does_not_draw_and_rejects_margin_without_a_position(tmp_path):
    manifest = _manifest()
    cell = _cell(
        manifest,
        varying="short",
        fixed_kind="ideal",
        input_kind="noise",
        intensity=1.0,
        seed=42,
    )
    prepared = prepare_cell(cell, dev_scales=manifest["dev_scales"], scored_frame=_bars())
    before = draw_count()
    measured = run_book(_bars(), prepared["plugin_frame"], str(tmp_path / "book"))
    assert draw_count() == before
    assert measured["commission_rate"] == pytest.approx(RATE)
    assert measured["slippage_per_side"] == pytest.approx(SLIP)
    assert measured["shortcash"] is False
    assert measured["initial_cash"] == INITIAL_CASH
    for fill in measured["fills"]:
        assert abs(fill["price"] - fill["bar_open"]) <= SLIP + 1e-9

    gap = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 10.0, 10.0),
        ]
    )
    gap_pred = pd.DataFrame(
        {"Prediction_h_1": [10.0], "Prediction_d_1": [10.0]},
        index=pd.to_datetime(["2019-05-01 00:00"]),
    )
    rejected = run_book(gap, gap_pred, str(tmp_path / "margin"))
    assert "Margin" in rejected["order_states"]
    assert "Completed" not in rejected["order_states"]
    assert rejected["pending_position_units"] == pytest.approx(0.0)
    assert rejected["n_closed_trades"] == 0
    assert rejected["fills"] == []
    assert rejected["current_direction"] is None
    assert rejected["stop_close_obtained_fill"] is False
    assert any(event["action"] == "margin_rejected" for event in rejected["cap_events"])

    priced = _frame(
        [
            _ohlc("2019-05-01 00:00", 2.0, 2.0),
            _ohlc("2019-05-01 01:00", 2.0, 2.01),
        ]
    )
    priced_pred = pd.DataFrame(
        {"Prediction_h_1": [2.04], "Prediction_d_1": [2.04]},
        index=pd.to_datetime(["2019-05-01 00:00"]),
    )
    capped = run_book(priced, priced_pred, str(tmp_path / "cap"))
    assert capped["pending_position_units"] != pytest.approx(0.0)
    assert capped["stop_close_obtained_fill"] is False
    assert capped["position_before_stop"] == pytest.approx(capped["position_after_stop"])
    assert capped["n_closed_trades"] == 0
    event = next(item for item in capped["cap_events"] if item["action"] == "capped")
    assert abs(event["requested_units"]) > event["allowed_units"]
    assert abs(capped["pending_position_units"]) == pytest.approx(event["allowed_units"])
    assert abs(capped["pending_position_units"]) != pytest.approx(abs(event["requested_units"]))
    assert capped["book"]["equity_gap"] == pytest.approx(0.0, abs=1e-6)
    assert "Completed" in capped["order_states"]


def test_open_position_survives_stop_without_a_fabricated_fill(tmp_path):
    manifest = _manifest()
    cell = _cell(manifest, varying="short", fixed_kind="ideal", input_kind="ideal")
    result = run_cell(
        cell,
        dev_scales=manifest["dev_scales"],
        scored_frame=_bars(),
        work_directory=str(tmp_path / "open"),
        session_end="2019-05-01 01:00",
    )
    assert result["n_closed_trades"] == 0
    assert result["pending_position_units"] != pytest.approx(0.0)
    assert result["pending_position_units"] != pytest.approx(1_000_000.0)
    assert result["stop_close_obtained_fill"] is False
    assert result["position_before_stop"] == pytest.approx(result["position_after_stop"])
    assert len(result["fills"]) == 1
    assert result["broker_session_end"] == "2019-05-01T01:00:00+00:00"
    _assert_reconciled(result)


def test_executed_cells_reconcile_and_record_only_what_ran(tmp_path):
    bars = _bars()
    manifest = _manifest()
    assert all(cell["executed"] is False for cell in manifest["cells"])
    report = execute_declared(manifest, bars, str(tmp_path / "sweep"))
    assert report["b0_status"] == "B0_NOT_STARTED"
    assert report["slice"] == choose_slice(report["one_cell"]["wall_seconds"], manifest)
    assert report["one_cell"]["cell_id"] == PROBE_CELL_ID
    assert report["one_cell"]["wall_seconds"] > 0.0
    assert report["one_cell"]["cpu_seconds"] >= 0.0
    assert report["warmup"]["is_cell"] is False
    executed = [cell for cell in manifest["cells"] if cell["executed"]]
    declared_only = [cell for cell in manifest["cells"] if not cell["executed"]]
    assert len(executed) == report["executed_cell_count"]
    assert len(executed) + len(declared_only) == manifest["cell_count"]
    assert {row["cell_id"] for row in report["results"]} == {cell["cell_id"] for cell in executed}
    assert all(cell_in_slice(cell, report["slice"]) for cell in executed)
    naive = None
    for result in report["results"]:
        _assert_reconciled(result)
        assert result["support"] == manifest["support"]
        if naive is None:
            naive = result["per_horizon_naive_mae"]
        else:
            assert result["per_horizon_naive_mae"] == pytest.approx(naive)
        assert set(result["per_horizon_mae"]) == {str(hours) for hours in HORIZONS}
        targeted = []
        for key, target in result["per_horizon_target_mae"].items():
            if target is None:
                assert result["per_horizon_achieved_minus_target"][key] is None
            else:
                targeted.append(key)
                assert result["per_horizon_achieved_minus_target"][key] == pytest.approx(
                    result["per_horizon_mae"][key] - target
                )
        if targeted:
            assert result["targeted_horizon_achieved_macro"] == pytest.approx(
                sum(result["per_horizon_mae"][key] for key in targeted) / len(targeted)
            )
            assert result["target_macro_average"] == pytest.approx(
                sum(result["per_horizon_target_mae"][key] for key in targeted) / len(targeted)
            )
        else:
            assert result["targeted_horizon_achieved_macro"] is None
            assert result["target_macro_average"] is None
        if result["input_kind"] == "ideal" and result["fixed_kind"] == "ideal":
            assert result["mae_macro_average"] == pytest.approx(0.0)
        if result["input_kind"] == "persistence" and result["fixed_kind"] == "persistence":
            assert result["per_horizon_mae"] == pytest.approx(result["per_horizon_naive_mae"])
        if result["input_kind"] == "noise":
            noisy = SHORT_HORIZONS if result["varying"] == "short" else LONG_HORIZONS
            assert any(
                abs(result["per_horizon_achieved_minus_target"][str(hours)]) > 1e-12
                for hours in noisy
            )
        assert result["seed"] is None or result["input_kind"] == "noise"
    probe = next(row for row in report["results"] if row["cell_id"] == PROBE_CELL_ID)
    manual = prepare_cell(
        _cell(manifest, cell_id=PROBE_CELL_ID),
        dev_scales=manifest["dev_scales"],
        scored_frame=bars,
    )
    assert probe["per_horizon_mae"] == pytest.approx(manual["per_horizon_mae"])
    assert probe["per_horizon_naive_mae"] == pytest.approx(manual["per_horizon_naive_mae"])

    cells_out = []
    for cell in manifest["cells"]:
        row = {
            "cell_id": cell["cell_id"],
            "varying": cell["varying"],
            "fixed": cell["fixed"],
            "fixed_kind": cell["fixed_kind"],
            "role": cell["role"],
            "input_kind": cell["input_kind"],
            "intensity": cell["intensity"],
            "seed": cell["seed"],
            "executed": cell["executed"],
            "measurement": "MEASURED" if cell["executed"] else "NOT_MEASURED",
        }
        cells_out.append(row)
    manifest_out = {
        "label": "SYNTHETIC",
        "parent_commit": "e7966f33fca9946d6b2b9e45e28bc5f8ba20f6d8",
        "b0_status": "B0_NOT_STARTED",
        "financial_utility": "NOT_CLAIMED",
        "short_versus_long_utility": "NOT_CLAIMED",
        "parameters_not_chosen_by_pnl": True,
        "equal_mae_note": manifest["equal_mae_note"],
        "noise_levels": manifest["noise_levels"],
        "paired_seeds": manifest["paired_seeds"],
        "orientations": manifest["orientations"],
        "fixed_kinds": manifest["fixed_kinds"],
        "support": manifest["support"],
        "support_id": manifest["support_id"],
        "horizons_hours": manifest["horizons_hours"],
        "horizon_protocol": manifest["horizon_protocol"],
        "horizon_protocol_is_complete": True,
        "scale_source": "DEV",
        "dev_source": manifest["dev_source"],
        "dev_scales": manifest["dev_scales"],
        "dev_origin_count": manifest["dev_origin_count"],
        "dev_consumed_count": manifest["dev_consumed_count"],
        "dev_first_utc": manifest["dev_first_utc"],
        "dev_last_utc": manifest["dev_last_utc"],
        "noise_model": manifest["noise_model"],
        "slice_rule": manifest["slice_rule"],
        "slice_executed": report["slice"],
        "cell_wall_budget_seconds": report["cell_wall_budget_seconds"],
        "one_cell_wall_seconds": report["one_cell"]["wall_seconds"],
        "one_cell_cpu_seconds": report["one_cell"]["cpu_seconds"],
        "one_cell_id": report["one_cell"]["cell_id"],
        "warmup_wall_seconds": report["warmup"]["wall_seconds"],
        "warmup_cpu_seconds": report["warmup"]["cpu_seconds"],
        "warmup_is_cell": False,
        "execute_wall_seconds": report["wall_seconds"],
        "execute_cpu_seconds": report["cpu_seconds"],
        "cells_declared": manifest["cell_count"],
        "cells_executed": len(executed),
        "cells_declared_only": len(declared_only),
        "historic_cpu_ceiling_seconds": manifest["historic_cpu_ceiling_seconds"],
        "historic_ceiling_is_b0_budget": False,
        "retained_sweep_cells_not_run": 241,
        "ratios_21_not_run": True,
        "real_market": "NOT_RUN",
        "gpu": "NOT_RUN",
        "database_service": "NOT_RUN",
        "live_broker": "NOT_RUN",
        "initial_cash": INITIAL_CASH,
        "margin_fraction": 0.05,
        "leverage": 100.0,
        "cells": cells_out,
    }
    results_out = {
        "label": "SYNTHETIC",
        "financial_utility": "NOT_CLAIMED",
        "short_versus_long_utility": "NOT_CLAIMED",
        "equal_mae_is_not_the_same_control": True,
        "long_persistence_blocking_entry_is_not_short_utility": True,
        "parameters_not_chosen_by_pnl": True,
        "slice_executed": report["slice"],
        "cells_declared": manifest["cell_count"],
        "cells_executed": len(executed),
        "cells_declared_only": [cell["cell_id"] for cell in declared_only],
        "naive_mae_by_horizon": naive,
        "naive_macro_average": report["results"][0]["naive_macro_average"],
        "cells": [
            {
                "cell_id": row["cell_id"],
                "varying": row["varying"],
                "fixed": row["fixed"],
                "fixed_kind": row["fixed_kind"],
                "role": row["role"],
                "input_kind": row["input_kind"],
                "intensity": row["intensity"],
                "seed": row["seed"],
                "n_closed_trades": row["n_closed_trades"],
                "realized_pnl": row["realized_pnl"],
                "cash": row["cash"],
                "marked_equity": row["marked_equity"],
                "pending_position_units": row["pending_position_units"],
                "initial_cash": row["initial_cash"],
                "margin_fraction": row["margin_fraction"],
                "leverage": row["leverage"],
                "shortcash": row["shortcash"],
                "commission_rate": row["commission_rate"],
                "slippage_per_side": row["slippage_per_side"],
                "swap_per_lot_per_day": row["swap_per_lot_per_day"],
                "mae_macro_average": row["mae_macro_average"],
                "naive_macro_average": row["naive_macro_average"],
                "target_macro_average": row["target_macro_average"],
                "targeted_horizon_achieved_macro": row["targeted_horizon_achieved_macro"],
                "per_horizon_mae": row["per_horizon_mae"],
                "per_horizon_naive_mae": row["per_horizon_naive_mae"],
                "per_horizon_target_mae": row["per_horizon_target_mae"],
                "per_horizon_achieved_minus_target": row["per_horizon_achieved_minus_target"],
                "per_horizon_dev_scale": row["per_horizon_dev_scale"],
                "entry_sides": row["entry_sides"],
                "trades": row["trades"],
                "fills": row["fills"],
                "cap_events": row["cap_events"],
                "order_states": row["order_states"],
                "book": row["book"],
                "cost_sums": {
                    kind: float(
                        sum(item["amount"] for item in row["ledger"] if item["kind"] == kind)
                    )
                    for kind in ("commission", "spread", "slippage", "swap")
                },
                "equity_gap": row["book"]["equity_gap"],
                "flat_cash_gap": row["book"]["flat_cash_gap"],
                "collateral": row["book"]["collateral"],
                "unrealized_pnl": row["book"]["unrealized_pnl"],
                "stop_close_obtained_fill": row["stop_close_obtained_fill"],
                "wall_seconds": row["wall_seconds"],
                "cpu_seconds": row["cpu_seconds"],
            }
            for row in report["results"]
        ],
    }
    _write_json(EVIDENCE / "MANIFEST.json", manifest_out)
    _write_json(EVIDENCE / "RESULTS.json", results_out)
