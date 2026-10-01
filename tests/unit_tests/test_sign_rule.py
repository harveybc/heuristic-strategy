"""Pre-declared SIGN rule and model-free viability diagnostics (coordinator ruling, FX)."""
import json
import math
from pathlib import Path

import pytest

from app.paired_backtest import COSTS, run_episode
from app.sign_rule import load_cost_profile, sign_diagnostics, sign_targets, solve_mu_sigma

ROOT = Path(__file__).resolve().parents[2]


def _bars(closes):
    return [{"time": f"b{i:05d}", "open": c, "high": c, "low": c, "close": c} for i, c in enumerate(closes)]


def test_cost_profile_is_versioned_and_declared():
    p = load_cost_profile(ROOT / "docs/contracts/fx_profile.eurusd_ibkr_canary.v1.json")
    assert p["schema"] == "heuristic_strategy.cost_profile.v1" and len(p["sha256"]) == 64
    assert p["modelled_round_trip_spread_price"] == 0.0001 and p["spread_cap_price"] == 0.0003
    assert p["broker_fill_costs"]["status"] == "NOT_AVAILABLE"


def test_run_episode_charges_a_declared_profile_and_defaults_to_the_harness_costs():
    bars = _bars([1.10, 1.10, 1.12, 1.12])
    default = run_episode(bars, [1, 1, 0, 0])
    assert default["equity"] == run_episode(bars, [1, 1, 0, 0], costs=None)["equity"]
    profile = {"commission_fraction_per_side": 0.0, "modelled_round_trip_spread_price": 0.0002}
    spread_only = run_episode(bars, [1, 1, 0, 0], costs=profile)
    gross = run_episode(bars, [1, 1, 0, 0], costs={"commission_fraction_per_side": 0.0,
                                                   "modelled_round_trip_spread_price": 0.0})
    assert gross["equity"][-1] - spread_only["equity"][-1] == pytest.approx(0.0002)  # half per side x 2
    assert COSTS["commission"] == 0.001


def test_sign_rule_holds_for_the_horizon_and_never_overlaps():
    closes = [1.0 + 0.001 * i for i in range(12)]
    bars = _bars(closes)
    rows = {b["time"]: {"logret_hat_h3": (0.01 if i % 2 == 0 else -0.01), "n_rows_h3": 3} for i, b in enumerate(bars)}
    targets, trades = sign_targets(bars, rows, 3)
    assert trades[0] == {"origin": 0, "exit": 3, "sign": 1}
    assert [t["origin"] for t in trades] == [0, 3, 6]  # the next decision only after the hold ends
    assert targets[:3] == [1, 1, 1]


def test_diagnostics_give_edge_hit_rate_and_break_even_cost():
    closes = [1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0]
    bars = _bars(closes)
    rows = {b["time"]: {"logret_hat_h1": 1e-4 if i % 2 == 0 else -1e-4, "n_rows_h1": 1,
                        "y_hat_z_h1": 1e-4 / 0.005 if i % 2 == 0 else -1e-4 / 0.005} for i, b in enumerate(bars)}
    d = sign_diagnostics(bars, rows, 1, sigma=0.005, mu=0.0)
    assert d["trades"] == 6 and d["hit_rate"] == 1.0
    edge = sum(abs(closes[i + 1] - closes[i]) for i in range(6)) / 6
    assert d["mean_gross_edge_price"] == pytest.approx(edge)
    assert d["break_even_round_trip_price"] == pytest.approx(edge)
    assert d["mean_gross_edge_z"] == pytest.approx(sum(abs(math.log(closes[i + 1] / closes[i])) for i in range(6)) / 6 / 0.005)


def test_mu_and_sigma_are_recovered_from_two_horizons():
    sigma, mu = 0.0007, 2e-6
    row = {"y_hat_z_h1": 0.3, "logret_hat_h1": sigma * 0.3 + 1 * mu, "n_rows_h1": 1,
           "y_hat_z_h4": -0.2, "logret_hat_h4": sigma * -0.2 + 4 * mu, "n_rows_h4": 4}
    s, m = solve_mu_sigma(row, 1, 4)
    assert s == pytest.approx(sigma) and m == pytest.approx(mu)
