"""Timing-selection evaluator: windows, periods, the same-units owner monitor, regularization grid and cell
rule, a tiny CPU fit per arm, block bootstrap, DSR supplement (TE01-TE08)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TE = pytest.importorskip("tools.short_horizon_timing_eval")


def test_te01_windows_use_only_current_and_past_bars():
    z = np.arange(100, dtype=np.float32).reshape(100, 1)
    w = TE.windows(z, np.array([60]))
    assert w.shape == (1, 48, 1) and w[0, -1, 0] == 60 and w[0, 0, 0] == 13


def test_te02_periods_are_ordered_and_2024_is_reobserved_not_pristine():
    ts = pd.DatetimeIndex(["2022-12-31 23:00", "2023-01-01", "2023-06-30 23:00", "2023-07-01", "2024-01-01"], tz="UTC")
    assert list(TE.period_of(ts)) == ["TRAIN_2020_2022", "INNER_A_2023H1", "INNER_A_2023H1", "INNER_B_2023H2", "REOBSERVED_2024"]
    assert "pristine" not in " ".join(TE.EVAL_PERIODS).lower()


def test_te03_monitor_scores_train_and_inner_in_the_same_units_and_keeps_running_loss_apart():
    cw = np.array([1.0, 2.0, 2.0])
    y_tr, y_in = np.array([0, 1, 2]), np.array([0, 1, 2])
    mon = TE.OwnerMonitor(cw, y_tr, y_in, patience=2, min_delta=1e-8, budget=10)
    good, bad = np.eye(3) * 0.8 + 0.1 / 1.5, np.full((3, 3), 1 / 3)
    assert mon.step(0, bad, bad, lambda: 0, running=5.0) is False
    assert mon.step(1, good, good, lambda: 1, running=99.0) is False
    h = mon.history[1]
    assert h["score"] == pytest.approx(0.5 * (h["train_eval_loss"] + h["inner_a_loss"]))
    assert h["running_train_loss"] == 99.0 and h["train_eval_loss"] < 5.0           # running loss never enters the score
    assert mon.step(2, bad, bad, lambda: 2) is False and mon.step(3, bad, bad, lambda: 3) is True
    s = mon.summary()
    assert s["best_step"] == 1 and s["stop_cause"] == "PATIENCE" and s["budget"] == 10


def test_te04_declared_grid_and_primary_cells():
    assert len(TE.GRID["m1_dmo_class"]) == 6 and len(TE.GRID["m2_gru"]) == 6 and len(TE.GRID["m3_hgb"]) == 3
    assert TE.PRIMARY["m2_gru"] == {"dropout": 0.1, "weight_decay": 1e-4}
    assert TE.TRAINING["hgb_max_depth"] <= 4 and TE.TRAINING["hgb_min_samples_leaf"] >= 200


def test_te05_cell_rule_is_lowest_mean_inner_loss_then_gap_and_records_a_replacement():
    fits = {}
    for cell, loss, gap in (({"dropout": 0.1, "weight_decay": 1e-4}, 1.0, 0.1), ({"dropout": 0.2, "weight_decay": 1e-3}, 0.9, 0.3)):
        for s in (1, 2):
            fits[f"{TE.cell_id(cell)}__s{s}"] = {"cell": cell, "best_inner_a_loss": loss, "best_gap": gap}
    win, table, rule = TE.choose_cell(fits, "m2_gru")
    assert win["cell"] == {"dropout": 0.2, "weight_decay": 1e-3} and rule["replaced_primary"] is True


@pytest.mark.parametrize("arm", ["m1_dmo_class", "m2_gru", "m3_hgb"])
def test_te06_tiny_fit_per_arm_restores_the_best_checkpoint(arm):
    if arm != "m3_hgb":
        pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    n, F = 700, 3
    Z = rng.normal(size=(n, F)).astype(np.float32)
    y = ((Z[:, 0] > 0.8).astype(int) + 2 * (Z[:, 1] > 0.8).astype(int)).clip(0, 2)
    rows = np.arange(60, n)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    xa = TE.arm_inputs(arm, Z, TE.time_encoding(ts), rows)
    sel = (lambda m: [v[m] for v in xa]) if isinstance(xa, list) else (lambda m: xa[m])
    tr = np.arange(len(rows)) < 450
    saved = dict(TE.TRAINING)
    TE.TRAINING.update(max_epochs=3, hgb_max_iter=30)
    try:
        model, fit = TE.fit_arm(arm, TE.PRIMARY[arm], sel(tr), y[rows][tr], sel(~tr), y[rows][~tr], 1, F)
    finally:
        TE.TRAINING.clear()
        TE.TRAINING.update(saved)
    P = model.predict(xa, verbose=0)
    assert P.shape == (len(rows), 3) and np.allclose(P.sum(1), 1.0, atol=1e-5)
    assert fit["best_step"] >= 0 and fit["stop_cause"] in ("PATIENCE", "BUDGET_EXHAUSTED")


def test_te07_week_block_bootstrap_brackets_a_known_mean():
    idx = pd.date_range("2023-01-02", periods=24 * 7 * 20, freq="h", tz="UTC")
    f = pd.DataFrame({"d": np.random.default_rng(1).normal(0.5, 1.0, len(idx))}, index=idx)
    ci = TE.block_bootstrap(lambda g: float(g["d"].mean()), f, reps=300)
    assert ci["q025"] < 0.5 < ci["q975"] and ci["block"] == "ISO calendar week"


def test_te08_deflated_sharpe_falls_as_trials_grow():
    few = TE.deflated_sharpe(0.1, 250, 0.0, 3.0, [0.0, 0.05])
    many = TE.deflated_sharpe(0.1, 250, 0.0, 3.0, list(np.linspace(-0.1, 0.1, 40)))
    assert few["dsr"] > many["dsr"]


def test_te09_model_files_never_collide_across_cells_and_seeds(tmp_path):
    _M = dict                                                          # any picklable object stands in for a model
    names = set()
    for cell in TE.GRID["m3_hgb"]:
        for s in TE.SEEDS:
            names.add(TE.save_model(_M(v=s), tmp_path / f"{TE.cell_id(cell)}__s{s}", "m3_hgb").name)
    assert len(names) == len(TE.GRID["m3_hgb"]) * len(TE.SEEDS)
    keys = {f"{TE.cell_id(c)}__s{s}" for c in TE.GRID["m2_gru"] for s in TE.SEEDS}
    files = {(Path("x") / k).parent / (k + ".keras") for k in keys}
    assert len(files) == len(keys) == 18
    with pytest.raises(SystemExit, match="MODEL_FILE_EXISTS"):
        TE.save_model(_M(v=0), tmp_path / f"{TE.cell_id(TE.GRID['m3_hgb'][0])}__s{TE.SEEDS[0]}", "m3_hgb")
