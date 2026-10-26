"""Signal audit (pre-registered 2026-10-26): metric, bootstrap, break-even, target-builder and verdict code against
known answers. No data, no model fit beyond tiny synthetic sets."""
import math

import numpy as np
import pytest

from tools import expert_guided_signal_audit as S


def test_auc_known_answers():
    assert S.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert S.auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert S.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5            # ties -> 0.5
    assert S.auc([0, 1, 1], [0.2, 0.1, 0.3]) == 0.5                     # one of two positive pairs ordered
    assert math.isnan(S.auc([1, 1], [0.1, 0.2]))


def test_brier_logloss_skill_and_spearman_r2():
    y = np.array([0, 1, 1, 0])
    assert S.brier(y, [0, 1, 1, 0]) == 0.0
    assert S.brier(y, [0.5] * 4) == 0.25
    assert S.skill(0.0, 0.25) == 1.0 and S.skill(0.25, 0.25) == 0.0
    assert S.logloss(y, [0.5] * 4) == pytest.approx(math.log(2))
    assert S.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert S.r2([1, 2, 3], [1, 2, 3], 2.0) == 1.0
    assert S.r2([1, 2, 3], [2, 2, 2], 2.0) == 0.0


def test_top_mask_and_breakevens():
    m = S.top_mask([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.0], 0.2)
    assert m.sum() == 2 and m[1] and m[6]
    # wins +1, losses -2: p* = 2 / 3
    net = np.array([1, 1, -2, -2])
    assert S.breakeven_win_rate(net, net > 0) == pytest.approx(2 / 3)
    # (d): holding a non-SL position gains +1 over closing, an SL position loses -3: closing pays when p > 1/4
    delta = np.array([1.0, 1.0, 1.0, -3.0])
    y = np.array([0, 0, 0, 1])
    assert S.breakeven_close_precision(delta, y) == pytest.approx(0.25)


def test_bootstrap_resamples_whole_days_and_is_reproducible():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(60), 5)
    y = (rng.random(300) < 0.5).astype(int)
    s = y + rng.normal(0, 0.5, 300)              # informative score
    fn = S.classification_metrics(y, s, 0.5)
    a = S.bootstrap(fn, day, n=200, seed=1)
    b = S.bootstrap(fn, day, n=200, seed=1)
    assert a == b
    pt = fn(np.arange(300))["auc"]
    assert a["auc"][0] < pt < a["auc"][1] and a["auc"][0] > 0.6
    noise = S.classification_metrics(y, rng.normal(0, 1, 300), 0.5)
    c = S.bootstrap(noise, day, n=200, seed=1)
    assert c["auc"][0] < 0.5 < c["auc"][1]


def test_forward_sign_target_availability_and_sign():
    close = np.array([1.0, 1.1, 1.0, 1.2, 1.3, 1.3])
    row_bar = np.arange(6)
    bar_ns = np.arange(6) * 3600 * 10**9
    ok, y, move = S.forward_sign_target(close, row_bar, np.arange(6), 2, bar_ns, np.int64(5 * 3600 * 10**9), 3600 * 10**9)
    # row 0 -> bar 2 (end 3 h <= 5 h): available, 1.0 vs 1.0 -> not up; row 2 -> bar 4 (end 5 h): available, up
    # row 3 -> bar 5 (end 6 h > 5 h): unavailable; rows 4, 5: beyond the data
    assert ok.tolist() == [True, True, True, False, False, False]
    assert y[ok].tolist() == [0, 1, 1]
    assert move[0] == pytest.approx(0.0)


def test_exit_decisions_still_open_and_label():
    row_bar = np.arange(20)
    entry_rows = np.array([0, 1, 2, 3])
    hit = np.array([S.HIT_SL, S.HIT_TP, S.HIT_TIME, S.HIT_TIME])
    net = np.array([-2.0, 1.0, -0.1, 0.2])
    exit_bar = np.array([10, 3, 12, 15])
    r0, r, y = S.exit_decisions(entry_rows, row_bar, hit, net, exit_bar, 3)
    # entry 1 exits at bar 3 < its decision bar 4: closed, dropped
    assert r0.tolist() == [0, 2, 3] and r.tolist() == [3, 5, 6]
    assert y.tolist() == [1, 1, 0]                 # SL; negative forced flat; positive forced flat


def test_verdict_rule():
    ok_fold = {"hgb": {"point": {"top10_precision": 0.7}, "ci": {"auc": [0.53, 0.6]}, "breakeven": 0.6},
               "linear": {"point": {"top10_precision": 0.5}, "ci": {"auc": [0.49, 0.55]}, "breakeven": 0.6}}
    v = S.verdict("a", {"wf2022": ok_fold, "wf2023": ok_fold})
    assert v["verdict"] == "SIGNAL_PRESENT" and v["passing_models"] == ["hgb"]
    weak = {"hgb": {"point": {"top10_precision": 0.7}, "ci": {"auc": [0.51, 0.6]}, "breakeven": 0.6}}
    assert S.verdict("a", {"wf2022": ok_fold, "wf2023": weak})["verdict"] == "NO_USABLE_SIGNAL"
    uneconomic = {"hgb": {"point": {"top10_precision": 0.55}, "ci": {"auc": [0.6, 0.7]}, "breakeven": 0.6}}
    assert S.verdict("a", {"wf2022": uneconomic, "wf2023": uneconomic})["verdict"] == "NO_USABLE_SIGNAL"
    reg = {"linear": {"point": {"top10_mean_value": 0.1}, "ci": {"spearman": [0.03, 0.1]}, "breakeven": None}}
    assert S.verdict("regression", {"wf2022": reg, "wf2023": reg})["passing_models"] == ["linear"]
    reg_bad = {"linear": {"point": {"top10_mean_value": -0.1}, "ci": {"spearman": [0.03, 0.1]}, "breakeven": None}}
    assert S.verdict("regression", {"wf2022": reg, "wf2023": reg_bad})["verdict"] == "NO_USABLE_SIGNAL"
    assert S.verdict("a", {"wf2022": ok_fold})["verdict"] == "NO_USABLE_SIGNAL"      # one fold is never enough


def test_econ_metrics_c_long_top_short_bottom():
    score = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    move = np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0])
    cost = np.full(10, 0.1)
    out = S.econ_metrics({"kind": "c", "move": move, "cost": cost}, np.arange(10), score, None)
    assert out["top10_precision"] == 1.0 and out["top10_mean_value"] == pytest.approx(0.9)


def test_fit_classifier_runs_and_calibrates_on_synthetic():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1200, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 1, 1200) > 0).astype(int)
    fn, info = S.fit_classifier("linear", X[:800], y[:800], X[800:1000], y[800:1000])
    p = fn(X[1000:])
    assert 0 < p.min() and p.max() < 1 and S.auc(y[1000:], p) > 0.65 and "C" in info
