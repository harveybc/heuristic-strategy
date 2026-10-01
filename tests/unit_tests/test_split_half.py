import math

import pytest

from app.paired_backtest import HeuristicParams
from app.split_half import horizon_mae, seed_summary, split_half


def _bars(closes):
    return [{"time": f"bar-{i:05d}", "open": c, "high": c * 1.01, "low": c * 0.99, "close": c}
            for i, c in enumerate(closes)]


def test_horizon_mae_matches_a_hand_computation():
    bars = _bars([100.0, 110.0, 99.0, 99.0])
    preds = [{"time": "bar-00000", "logret_hat_h1": 0.05}, {"time": "bar-00001", "logret_hat_h1": -0.05},
             {"time": "bar-00003", "logret_hat_h1": 0.0}]  # last origin has no target: dropped everywhere
    mu = 0.001
    table = horizon_mae(bars, preds, [1], mu)[1]
    r0, r1 = math.log(110 / 100), math.log(99 / 110)
    assert table["rows"] == 2
    assert table["model_mae"] == pytest.approx((abs(0.05 - r0) + abs(-0.05 - r1)) / 2)
    assert table["zero_return_mae"] == pytest.approx((abs(r0) + abs(r1)) / 2)
    assert table["train_mean_mae"] == pytest.approx((abs(mu - r0) + abs(mu - r1)) / 2)
    assert table["naive_mae"] == min(table["zero_return_mae"], table["train_mean_mae"])
    assert table["passes"] == (table["model_mae"] < table["naive_mae"])


def test_split_half_chooses_on_the_first_half_and_scores_only_the_second():
    closes = [100 * math.exp(0.01 * ((-1) ** i)) for i in range(60)]
    bars = _bars(closes)
    preds = []
    for i in range(59):
        actual = math.log(closes[i + 1] / closes[i])
        good = actual if i < 30 else -actual  # perfect on the first half, inverted on the second
        preds.append({"time": bars[i]["time"], "logret_hat_h1": good, "close_hat_h1": closes[i] * math.exp(good)})
    out = split_half(bars, preds, [1], 0.0, HeuristicParams(profit_threshold_frac=0.005))
    assert out["first_half"]["chosen"] == [1]
    assert out["second_half"]["chosen_still_passing"] == []
    assert out["status"] == "SPLIT_HALF_DIAGNOSTIC"
    assert out["second_half"]["metrics"] is not None
    assert out["second_half"]["origins"] == 59 - 29


def test_split_half_without_a_first_half_pass_does_not_run():
    closes = [100.0 + i for i in range(20)]
    bars = _bars(closes)
    preds = [{"time": b["time"], "logret_hat_h1": -0.5, "close_hat_h1": 1.0} for b in bars[:-1]]
    out = split_half(bars, preds, [1], 0.0, HeuristicParams())
    assert out["first_half"]["chosen"] == [] and out["second_half"]["metrics"] is None


def test_seed_summary_counts_skips_and_uses_sample_sd():
    def item(label, seed, net=None, sharpe=None):
        metrics = None if net is None else {"net_return": net, "max_drawdown_fraction": 0.1, "turnover_units": 2,
                                            "trades_closed": 1, "exposure_fraction": 0.1,
                                            "sharpe": {"value": sharpe}}
        return {"label": label, "seed": seed, "metrics": metrics}
    out = seed_summary([item("a", 1, 0.1, 0.02), item("a", 2, 0.3, None), item("a", 3), item("b", 1, 0.0, 0.0)])
    assert out["a"]["ran"] == 2 and out["a"]["skipped"] == 1
    assert out["a"]["net_return"]["mean"] == pytest.approx(0.2)
    assert out["a"]["net_return"]["sd"] == pytest.approx(math.sqrt(0.02))
    assert out["a"]["sharpe"]["n"] == 1 and out["a"]["sharpe"]["undefined"] == 1
    assert out["b"]["net_return"]["sd"] is None
