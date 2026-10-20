"""Timing-selection evaluator: windows, periods, the same-units owner monitor, regularization grid and cell
rule, a tiny CPU fit per arm, block bootstrap, DSR supplement (TE01-TE08)."""
from __future__ import annotations

import json
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


def test_te10_expert_module_digest_must_match_the_driver():
    class _Ctx:
        h = l = np.ones(10)
    with pytest.raises(SystemExit, match="EXPERT_MODULE_DIGEST"):
        TE.setup_expert({"expert": {"module_sha256": "0" * 64, "params": {"depth": 3, "deviation_pips": 40.7, "backstep": 0}}},
                        _Ctx(), ".")


def test_te11_daily_returns_start_from_initial_capital_on_a_nonzero_first_day():
    import types
    res = types.SimpleNamespace(daily_equity=pd.Series([110.0, 121.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")))
    r = TE.daily_returns(res, initial=100.0)
    assert np.allclose(r.to_numpy(), [0.10, 0.10])


def _ev_dir(tmp_path, cap=4, depth=3):
    ev = tmp_path / "eval_input"
    ev.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2023-01-02", periods=60, freq="h", tz="UTC")
    pd.DataFrame({"timestamp": ts, "f": np.arange(60.0)}).to_parquet(ev / "features_all.parquet", index=False)
    pd.DataFrame({"timestamp": ts, "timing_pivot_k2": np.zeros(60)}).to_parquet(ev / "targets_all.parquet", index=False)
    doc = {"sets": {"a": ["f"]}, "features_sha256": TE.sha(ev / "features_all.parquet"), "targets_sha256": TE.sha(ev / "targets_all.parquet"),
           "k_bars": 2, "expert": {"params": {"depth": depth, "deviation_pips": 40.7, "backstep": 0}},
           "strategy": {"executable": True, "max_orders_per_day": cap, "k_tp": 2.0}}
    (ev / "FEATURE_SETS.json").write_text(json.dumps(doc))
    return ev


def test_te12_freeze_binding_refuses_cap_expert_and_label_drift_before_any_model_load(tmp_path, monkeypatch):
    ev = _ev_dir(tmp_path)
    frozen = {"input_identity": TE.input_identity(ev)}
    TE.check_frozen_inputs(frozen, ev)                                        # unchanged -> accepted
    for mutate in (lambda: _ev_dir(tmp_path, cap=3), lambda: _ev_dir(tmp_path, depth=4)):
        mutate()
        with pytest.raises(SystemExit, match="INPUT_IDENTITY_DRIFT"):
            TE.check_frozen_inputs(frozen, ev)
    _ev_dir(tmp_path)
    pd.DataFrame({"timestamp": pd.date_range("2023-01-02", periods=60, freq="h", tz="UTC"),
                  "timing_pivot_k2": np.ones(60)}).to_parquet(ev / "targets_all.parquet", index=False)     # labels changed
    with pytest.raises(SystemExit, match="INPUT_IDENTITY_DRIFT"):
        TE.check_frozen_inputs(frozen, ev)
    # reconfirm itself: a changed cap stops BEFORE any model is loaded
    _ev_dir(tmp_path)
    out = tmp_path / "eval"
    out.mkdir()
    fz = {"input_identity": TE.input_identity(ev), "shards": {}}
    fz["freeze_sha256"] = TE.canon(fz)
    (out / "FROZEN.json").write_text(json.dumps(fz))
    _ev_dir(tmp_path, cap=3)
    monkeypatch.setattr(TE, "load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("MODEL_LOADED")))
    a = TE.argparse.Namespace(eval_input=str(ev), out=str(out), data=None, closures=None, zigzag_run=".")
    with pytest.raises(SystemExit, match="INPUT_IDENTITY_DRIFT"):
        TE.reconfirm(a)


def test_te13_calendar_from_closures_applies_the_twelve_hour_rules():
    idx = pd.date_range("2023-01-08 22:00", "2023-01-13 21:00", freq="h", tz="UTC")
    cal = TE.calendar_from_closures([["2023-01-06 22:00:00+00:00", "2023-01-08 22:00:00+00:00"],
                                     ["2023-01-13 22:00:00+00:00", "2023-01-15 22:00:00+00:00"]], idx)
    seg = cal.segments[0]
    assert seg.operational_start == pd.Timestamp("2023-01-09 10:00", tz="UTC")
    assert seg.force_flat == pd.Timestamp("2023-01-13 10:00", tz="UTC") and seg.entry_cutoff == seg.force_flat
    assert (cal.assign(idx) == 0).all()


def test_te14_a_finite_label_on_an_ineligible_origin_is_refused():
    ts = pd.date_range("2023-01-02", periods=60, freq="h", tz="UTC")
    feats = pd.DataFrame({"f": np.arange(60.0)})
    targ = pd.DataFrame({"timing_pivot_k2": np.zeros(60)})
    elig = np.ones(60, bool)
    elig[55] = False
    with pytest.raises(SystemExit, match="ELIGIBILITY_LABEL_MISMATCH"):
        TE.common_rows(feats, targ, ts, {"sets": {"a": ["f"]}}, 2, elig)


def test_te15_train_prior_baseline_is_measured_on_the_same_rows():
    y = np.array([0, 0, 0, 1, 2] * 40)
    prior = np.bincount(y, minlength=3) / len(y)
    P = np.tile(prior, (len(y), 1))
    b = TE.classification_baselines([P, P], y, prior, np.ones(len(y), bool))
    assert abs(b["log_loss_minus_train_prior"]) < 1e-12 and b["valley_pr_baseline_base_rate"] == pytest.approx(0.2)
