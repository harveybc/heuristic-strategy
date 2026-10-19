"""Timing-selection evaluator: causal design matrix, split purge, owner's monitor, tiny CPU fit (TE01-TE04)."""
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


def test_te01_design_matrix_rows_use_only_current_and_past_bars():
    f = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 10})
    X = TE.design_matrix(f, ["a", "b"], lags=2)
    assert X.shape == (10, 6)
    assert np.isnan(X[0, 2:]).all() and np.isnan(X[1, 4:]).all()
    assert X[5].tolist() == [5.0, 50.0, 4.0, 40.0, 3.0, 30.0]
    # prefix invariance: computing on a prefix gives the same first rows
    assert np.array_equal(TE.design_matrix(f.iloc[:6], ["a", "b"], 2)[5], X[5])


def test_te02_split_rows_purges_the_k_bars_before_each_cut():
    ts = pd.date_range("2022-12-31 20:00", periods=10, freq="h", tz="UTC")
    splits = {"train": ["2020-01-01", "2023-01-01"], "inner": ["2023-01-01", "2024-01-01"], "confirm": ["2024-01-01", "2025-01-01"]}
    period, purged = TE.split_rows(ts, splits, k=2)
    assert list(period[:4]) == ["DECISION_2020_2022"] * 4 and list(period[4:]) == ["VALIDATION_2023"] * 6
    assert purged.tolist() == [False, False, True, True, False, False, False, False, False, False]


def test_te03_owner_monitor_is_the_mean_of_train_and_inner_and_restores_the_best():
    tf = pytest.importorskip("tensorflow")
    keras = tf.keras

    class _M:
        def __init__(self):
            self.w = 0
            self.stop_training = False

        def predict(self, x, verbose=0, batch_size=0):
            p = np.full((len(x), 3), 1 / 3)
            return p

        def get_weights(self):
            return self.w

    x = np.zeros((6, 2), np.float32)
    y = np.array([0, 1, 2, 0, 1, 2])
    mon = TE._OwnerMonitor(keras, x, y, np.ones(6), patience=2, min_delta=1e-8)
    m = _M()
    inner = -np.log(1 / 3)
    for epoch, train in enumerate([1.0, 0.5, 0.6, 0.7]):
        m.w = epoch
        mon.on_epoch_end(epoch, {"loss": train}, m)
    assert mon.history[1]["score"] == pytest.approx(0.5 * (0.5 + inner))
    assert mon.best_epoch == 1 and mon.best_weights == 1 and m.stop_training is True


def test_te04_tiny_fit_runs_on_cpu_and_returns_calibrated_shape():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = (x[:, 0] > 0.8).astype(int) + 2 * (x[:, 1] > 0.8).astype(int)
    y[y > 2] = 2
    TE.TRAINING["max_epochs"] = 3
    model, fit = TE.fit_mlp(x[:400], y[:400], x[400:], y[400:], seed=1)
    assert fit["epochs"] == 3 and len(fit["history"]) == 3
    P = model.predict(x[400:], verbose=0)
    assert P.shape == (200, 3) and np.allclose(P.sum(1), 1.0, atol=1e-5)
