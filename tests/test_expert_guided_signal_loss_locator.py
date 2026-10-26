"""Signal-loss locator (pre-registered 2026-10-26): pure helpers against known answers (no data, no checkpoints)."""
import numpy as np
import pytest

from tools import expert_guided_signal_loss_locator as L


def test_rolling_features_are_causal_and_skip_invalid_rows():
    C = np.arange(400, dtype=float).reshape(200, 2)
    valid = np.ones(200, bool)
    valid[150] = False
    R = L.rolling_features(C, valid)
    assert R.shape == (200, 8)
    # mean over the last 72 rows ending at row 100 (current included) uses rows 29..100 only
    assert R[100, 0] == pytest.approx(C[29:101, 0].mean())
    assert np.isnan(R[10, 0])                                # fewer than 36 valid rows yet
    # row 150 is invalid: excluded from the window ending at row 160
    w = [r for r in range(89, 161) if r != 150]
    assert R[160, 0] == pytest.approx(C[w, 0].mean())
    # changing a FUTURE row never changes a past summary
    C2 = C.copy()
    C2[180:] = 1e9
    assert np.allclose(L.rolling_features(C2, valid)[:180], R[:180], equal_nan=True)


def test_paired_drop_flags_a_real_loss_and_not_noise():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(80), 6)
    y = (rng.random(480) < 0.5).astype(float)
    good = y + rng.normal(0, 0.4, 480)
    bad = y + rng.normal(0, 3.0, 480)
    fns = {"S1": L.stage_metric("a", y, good, None), "S2": L.stage_metric("a", y, bad, None), "S2b": L.stage_metric("a", y, good + 1e-9, None)}
    draws = L.paired_bootstrap(fns, day, n=300)
    assert L.drop(draws, "S1", "S2", "auc")["flag"] == "SUSPECT_INTERNAL_DEFECT"
    assert L.drop(draws, "S1", "S2b", "auc")["flag"] == "NO_SIGNIFICANT_CHANGE"
    assert L.stage_verdict("a", {"auc": [0.53, 0.6]}) == "SIGNAL_PRESENT"
    assert L.stage_verdict("regression", {"spearman": [0.01, 0.1]}) == "ABSENT_AT_THIS_STAGE"


def test_first_vs_best_same_day():
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    rows_by_day = {0: np.array([0, 1, 2]), 1: np.array([3, 4])}
    day_of_row = {0: 0, 4: 1}
    r = L.first_vs_best(day_of_row, rows_by_day, [0, 4], score, y)
    assert r["days"] == 2 and r["precision_opened"] == 0.0 and r["precision_best_same_day"] == 1.0


def test_grid_round_and_mean_stage():
    assert L.grid_round([2.47, 0.8, 1.6], (0.75, 1.5, 2.5)).tolist() == [2.5, 0.75, 1.5]
    y = np.array([0, 1, 0, 1], float)
    m = L.mean_stage([L.stage_metric("a", y, np.array([0.1, 0.9, 0.2, 0.8]), None),
                      L.stage_metric("a", y, np.array([0.9, 0.1, 0.8, 0.2]), None)])
    assert m(np.arange(4))["auc"] == pytest.approx(0.5)
