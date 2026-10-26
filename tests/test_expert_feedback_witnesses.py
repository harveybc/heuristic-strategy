"""Witnesses W1..W7 of ZIGZAG_EXPERT_FEEDBACK_DESIGN_2026_10_26.md on synthetic data (CPU, numpy, no market data)."""
import numpy as np
import pytest

from tools import expert_feedback_witnesses as W


def test_w1_preference_training_agrees_above_chance_and_exposes_obedience_by_type():
    r = W.W1_W2(seed=0)
    assert r["agreement"] > r["chance"] + 0.10                      # weighted per-tick agreement well above chance
    assert r["by_type"]["open"] > 0.55                              # obeys OPEN preferences where the signal exists
    assert 0.0 <= r["by_type"]["hold_or_nop"] < r["by_type"]["open"]  # and the instrument EXPOSES weaker obedience on HOLD/NOP
    assert len(r["curve_pi"]) == 5 and all(np.isfinite(r["curve_pi"]))


def test_w2_shuffled_preferences_collapse_to_chance():
    r = W.W1_W2(seed=0, shuffle=True)
    assert abs(r["agreement"] - r["chance"]) < 0.05
    assert r["by_type"]["open"] < 0.2                               # no signal -> the student cannot follow shuffled opens


def test_w3_exact_regret_identity():
    r = W.W3(seed=0)
    assert r["oracle_max_regret"] == 0.0                            # the argmax-Q policy has zero regret at EVERY tick
    assert r["random_mean_regret"] > 0.1 and r["random_min"] >= 0.0


def test_w4_availability_labels_do_not_change_when_the_future_is_perturbed():
    r = W.W4(seed=0)
    assert r["checked"] > 500 and r["diffs"] == 0


def test_w5_reward_model_inputs_carry_no_hindsight():
    r = W.W5(seed=0)
    assert r["static_ok"] and r["perturbation_invariant"]


def test_w6_every_tick_has_exactly_one_type_and_a_desired_set():
    r = W.W6(seed=0)
    assert r["total"] == r["ticks"] and r["missing_D"] == 0
    assert set(r["counts"]) == set(W.TYPES)


def test_w7_ledger_reproduces_the_engine_aggregate_exactly():
    r = W.W7(seed=0)
    assert r["equal"] and r["agent_closes_ledger"] == r["agent_closes_trades"] and r["trades"] > 50


def test_worked_example_matches_the_design_table():
    rows = W.worked_example()
    assert rows[0]["f_open"] == 1.0 and rows[0]["f_close"] == 0.0 and rows[0]["open_desired"]
    assert rows[8]["f_close"] == 0.9 and rows[8]["close_desired"]                 # f_close >= take -> CLOSE desired
    assert all(rows[i]["f_open"] >= rows[i + 1]["f_open"] - 1e-9 or rows[i + 1]["open_next"] < rows[i]["open_next"] for i in range(8))


def test_desired_and_undesired_sets_are_disjoint_everywhere():
    m = W.synth_market(n=800, seed=3); tabs = W.ladder_tables(m); th = W.Thresholds()
    for t in range(30, 790):
        ex = W.expert_tick(tabs, t, th)
        assert not (ex["D"] & ex["U"]) and ex["D"]
        for d in (1, -1):
            c = W.expert_close(tabs, t, d, th)
            assert not (c["D"] & c["U"]) and c["D"]


# ---------------------------------------------------------------------------------------------- §10 hierarchy + interventions
def test_w8_short_alarm_fires_under_injected_shocks_on_every_seed():
    for seed in (0, 1, 2, 3):
        r = W.W8_W10(seed=seed)
        for k in ("jumps_fast_3bars", "jumps_slow_6bars"):
            assert r[k]["alarms_fired"] >= 1, (seed, k)               # the witness FAILS if the alarm never fires
            assert r[k]["agent_closes_full"] >= 1, (seed, k)


def test_w9_early_close_limits_the_loss_pooled_over_seeds_for_shocks_slower_than_the_stop():
    full, noec, fast_full, fast_noec = 0.0, 0.0, 0.0, 0.0
    for seed in (0, 1, 2, 3):
        r = W.W8_W10(seed=seed)
        full += r["jumps_slow_6bars"]["shock_net_full"]; noec += r["jumps_slow_6bars"]["shock_net_no_early_close"]
        fast_full += r["jumps_fast_3bars"]["shock_net_full"]; fast_noec += r["jumps_fast_3bars"]["shock_net_no_early_close"]
    assert noec < 0 and full > noec                                   # pooled shock-window loss is smaller with early close
    assert fast_full > fast_noec - 1e-9                               # fast shocks: no worse than the stop alone (≈ equal)


def test_w10_nop_veto_reduces_trading_in_the_range_regime_and_missing_observations_block_entries():
    for seed in (0, 1, 2, 3):
        r = W.W8_W10(seed=seed)
        g = r["regime_flip"]
        assert g["trades_in_range_with_veto"] < g["trades_in_range_without_veto"]
        assert r["missing_observations"]["entries_during_gap"] == 0 and r["missing_observations"]["protections_still_run"]
        assert r["spread_blowup"]["trades_window_shocked"] >= 0


def test_hierarchy_oracles_are_consistent_with_the_leg_tables():
    m = W.synth_market(n=1200, seed=4); tabs = W.ladder_tables(m)
    seen_counter = seen_fresh = 0
    for t in range(60, 1100):
        for d in (1, -1):
            co, fi = W.coarse_oracle(tabs, t, d), W.fine_oracle(tabs, t, d)
            if np.isfinite(co["room"]):
                assert co["direction"] in (-1, 0, 1)
            seen_counter += fi["counter_within_h"]; seen_fresh += fi["fresh_favour"]
            assert fi["counter_within_h"] in (0, 1) and fi["fresh_favour"] in (0, 1)
            assert W.causal_alarm(m, t, d, t) is False                # no alarm at the entry bar itself
    assert seen_counter > 0 and seen_fresh > 0
