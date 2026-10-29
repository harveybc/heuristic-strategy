"""STRATEGY_PARAMS v2.2 / C3 contract v2.4 (coordinator rulings A-D on tier 2 of 443b7a1, 2026-10-29): per-profile bounds
and defaults; the full_f3 tables, defaults and contract identities stay byte-identical (pinned below with the values
measured at 30b9b47, before the change); full_f4 widens open_rate to [0.002, 1.0], sets burst_vol_ratio 1.5 in [1.2, 2.5];
the advantage closer keeps O*'s tie rule in every profile (ruling C withdrawn, v2.3); ties-HOLD is only the declared
variant MUSASHI_TIE_HOLD."""
import hashlib
import json

import numpy as np
import pytest

from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import menu as M
from app.short_horizon.expert_guided import strategy_params as SP
from app.short_horizon.expert_guided import strategy_zz as ZZ
from tools import strategy_param_deap as D


def _h(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def test_full_f3_tables_defaults_and_contract_identities_are_byte_identical():
    assert _h(SP.BOUNDS) == "44b6bf727bd2e4dc7543c61e8ef482dab8afbc65caef91b44afe6fb724d429aa"
    assert _h(SP.DEFAULTS) == "a43517d39705a800e3db7a956837e7375582e8b5ed2f854d056bf7d722cd0676"
    assert _h(C3.Q_BOUNDS) == "5d1e3dc6e8323465aea3005ec98bc0e5c5151cb0774b7875c3aae80e2a8e5dae"
    assert SP.bounds_of("full_f3") is SP.BOUNDS and SP.default_doc("full_f3")["burst_vol_ratio"] == 2.0
    pinned = {("full_f3", "tcn"): "action_21c953427734c789", ("full_f3", "u"): "action_232890f6a24ebdf6",
              ("full_f3_12h", "tcn"): "action_d5b3b197593bccdb", ("full_f3_12h", "u"): "action_799c153615b346c9",
              ("legacy", "tcn"): "action_218b61027444dbab", ("legacy", "u"): "action_cbae635e6c5f13e1"}
    for (prof, arch), ident in pinned.items():
        d = C3.default_contract(C3.ARCH_TCN if arch == "tcn" else C3.ARCH_U, exec_profile=prof)
        assert d["identity"] == ident and d["strategy"]["genes"] == C3.Q_BOUNDS and C3.validate(d) == []
    assert C3.default_contract()["identity"] == "action_21c953427734c789"
    assert C3.FULL_F3.identity == "exec_709f9b1fa3512486" and C3.EXEC_VARIANTS["legacy"].identity == "exec_5d80473b21dc61a0"


def test_full_f4_open_rate_bound_is_widened_for_that_profile_only():
    f4 = SP.default_doc()
    assert f4["exec_profile"] == "full_f4" and SP.bounds_of("full_f4")["open_rate_l"] == [0.002, 1.0] == SP.BOUNDS_F4["open_rate_s"]
    assert SP.validate(dict(f4, open_rate_l=1.0, open_rate_s=1.0))["open_rate_l"] == 1.0
    for doc in (dict(SP.default_doc("full_f3"), open_rate_l=0.7, open_rate_s=0.7),
                {k: v for k, v in dict(f4, open_rate_l=0.7, open_rate_s=0.7).items() if k != "exec_profile"}):   # absent profile = full_f3
        with pytest.raises(SP.StrategyParamsError, match="OUT_OF_BOUNDS"):
            SP.validate(doc)
    with pytest.raises(SP.StrategyParamsError, match="OUT_OF_BOUNDS"):
        SP.validate(dict(f4, open_rate_l=1.01, open_rate_s=1.01))
    assert C3.Q_BOUNDS_F4["open_rate"] == [0.002, 1.0] and {k: v for k, v in C3.Q_BOUNDS_F4.items() if k != "open_rate"} == {k: v for k, v in C3.Q_BOUNDS.items() if k != "open_rate"}
    d = C3.default_contract(C3.ARCH_TCN, exec_profile="full_f4")
    assert d["strategy"]["genes"] == C3.Q_BOUNDS_F4 and C3.validate(d) == []
    bad = json.loads(json.dumps(d))
    bad["strategy"]["genes"] = dict(C3.Q_BOUNDS)                      # a full_f4 contract carrying the full_f3 table is refused, and vice versa
    assert any(e.startswith("C3_CONTRACT_GENES") for e in C3.validate(bad))
    f3 = json.loads(json.dumps(C3.default_contract(C3.ARCH_TCN)))
    f3["strategy"]["genes"] = dict(C3.Q_BOUNDS_F4)
    assert any(e.startswith("C3_CONTRACT_GENES") for e in C3.validate(f3))
    assert d["changelog"][0]["change"].startswith("v2.5") and d["changelog"][1]["change"].startswith("v2.4") and "v2.4" not in json.dumps(C3.default_contract(C3.ARCH_TCN)["changelog"])


def test_full_f4_burst_ratio_default_and_bounds():
    assert SP.default_doc()["burst_vol_ratio"] == 1.5 and SP.BOUNDS_F4["burst_vol_ratio"] == [1.2, 2.5] and SP.BOUNDS["burst_vol_ratio"] == [1.2, 6.0]
    with pytest.raises(SP.StrategyParamsError, match="OUT_OF_BOUNDS"):
        SP.validate(dict(SP.default_doc(), burst_vol_ratio=3.0))
    assert SP.validate(dict(SP.default_doc("full_f3"), burst_vol_ratio=3.0))["burst_vol_ratio"] == 3.0
    assert {k: v for k, v in SP.DEFAULTS_F4.items() if k != "burst_vol_ratio"} == {k: v for k, v in SP.DEFAULTS.items() if k != "burst_vol_ratio"}


def test_deap_genomes_encode_on_their_profile_table():
    b4, b3 = D.bounds(D.GENOME_F4), D.bounds(D.GENOME)
    assert b4["open_rate_l"] == [0.002, 1.0] and b4["burst_vol_ratio"] == [1.2, 2.5]
    assert b3["open_rate_l"] == [0.002, 0.50] and b3["burst_vol_ratio"] == [1.2, 6.0]
    base = SP.default_doc()
    x = [0.5] * len(D.GENOME_F4)
    x[D.GENOME_F4.index("open_rate_l")] = 1.0
    doc = D.decode(x, base)
    assert doc["open_rate_l"] == doc["open_rate_s"] == 1.0
    assert D.decode(D.encode(base, D.GENOME_F4), base, D.GENOME_F4)["open_rate_l"] == pytest.approx(0.45)
    xf = [0.5] * len(D.GENOME)
    xf[D.GENOME.index("open_rate_l")] = 1.0                           # a full-genome vector is read on the base document's profile table
    assert D.decode(xf, base)["open_rate_l"] == 1.0 and D.decode(D.project(xf, D.GENOME_F4, base), base, D.GENOME_F4)["open_rate_l"] == 1.0
    assert D.decode(xf, SP.default_doc("full_f3"))["open_rate_l"] == 0.50


def test_closer_keeps_the_ostar_tie_rule_in_every_profile_and_tie_hold_is_only_a_declared_variant():
    keep = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(z_close=1.0, closer_adv_margin=0.02)
        assert ZZ.TIE_RULE == "ostar" and ZZ.TIE_RULES == ("ostar", "musashi_tie_hold")
        for menu in (True, False):                                    # the default: O*'s rule, A - z s >= -closer_adv_margin
            for a, s_hat, want in ((0.0, 0.0, True), (-0.01, 0.0, True), (-0.02, 0.0, True), (-0.03, 0.0, False), (0.5, 0.51, True), (float("nan"), 0.0, False)):
                assert ZZ.advantage_close(a, s_hat, menu) is want, (menu, a, s_hat)
        for a, s_hat, want in ((0.0, 0.0, False), (1e-12, 0.0, True), (-0.01, 0.0, False), (0.5, 0.4, True), (0.5, 0.5, False)):
            assert ZZ.advantage_close(a, s_hat, True, tie_rule="musashi_tie_hold") is want, (a, s_hat)   # MUSASHI_TIE_HOLD (declared variant)
        assert ZZ.advantage_close(-0.01, 0.0, False, tie_rule="musashi_tie_hold") is True   # the variant acts under the menu only
        ZZ.PARAMS.update(closer_adv_margin=0.05)                      # the margin is read again (a full_f4 gene)
        assert ZZ.advantage_close(-0.04, 0.0, True) is True
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(keep)
    assert set(M.INERT_UNDER_MENU) == {"k_tp_protect", "tp_mode"} and "closer_adv_margin" in D.GENOME_F4
