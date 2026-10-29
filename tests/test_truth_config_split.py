"""S6 tier-1 defect fix: under the absolute-level profile the engine TRUTH (outcome heads on the sigma cell grid), the
exploration visits and the truth-based controls run on the sigma-level engine; the strategy's own orders (and the
comparators that share its opener) run on the profile. Found by tier 1: every cell opened nothing because the cell
grid's sigma multiples were read as absolute prices by the truth engine."""
import dataclasses

import numpy as np

from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import exec_variants as XV
from tools import strategy_behaviour_matrix as BM
from tools import strategy_output_simulation as S


def test_truth_config_is_the_profile_with_sigma_levels_and_identity_otherwise():
    base = E.EnvConfig(level_policy="clip")
    f3 = S.with_variant(base, C3.FULL_F3)
    assert S.truth_config(f3) is f3 and S.truth_config(base) is base                 # sigma profiles: identity
    f4 = S.with_variant(base, C3.FULL_F4)
    t = S.truth_config(f4)
    assert f4.variant.level_mode == "absolute_pips" and t.variant.level_mode == "sigma"
    assert t.variant.sizing_mode == "risk_budget" and t.sizing.leverage == 50.0 and t.max_entries_per_day is None   # everything else of the profile stays
    assert dataclasses.replace(t.variant, level_mode="absolute_pips") == f4.variant


def test_world_rollouts_pick_the_right_config_and_the_envelope_follows_the_profile():
    import inspect
    src = {fn: inspect.getsource(getattr(BM, fn)) for fn in ("rollout_with", "rollout_oracle", "rollout_oracle_star", "rollout_policy", "rollout_cell", "build_episodes")}
    for fn in ("rollout_with", "rollout_oracle", "rollout_oracle_star"):               # truth-based controls: the sigma-level engine
        assert 'world.get("scfg_truth", world["scfg"])' in src[fn], fn
    for fn in ("rollout_policy", "rollout_cell", "build_episodes"):                    # the strategy's opener: the profile
        assert 'world["scfg"]' in src[fn] and "scfg_truth" not in src[fn], fn
    assert "_absolute_mode" in inspect.getsource(BM._envelope) and "envl is None" in src["build_episodes"]

    class Pol:
        def _absolute_mode(self):
            return True

        def absolute_envelope(self, r, si):
            return {"tp": 1.105, "sl": 1.095, "h": 6} if si == 0 else None

    assert BM._envelope(Pol(), 0, 0) == (1.105, 1.095) and BM._envelope(Pol(), 0, 1) is None
