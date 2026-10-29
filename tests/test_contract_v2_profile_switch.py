"""S6 re-seal and profile switch (business contract): STRATEGY_PARAMS v2 carries the contract genes and exec_profile;
exec_variant_of maps a parameter document to its execution variant (absent = full_f3, byte-identical; full_f4 with the
document's rho); the sealed C3 contracts are the full_f4 v2 with the business-contract block; the DEAP genome carries
the new genes with the two categorical options binary."""
import hashlib
import json

import pytest

from app.short_horizon.expert_guided import action_contract as ACT
from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import exec_variants as XV
from app.short_horizon.expert_guided import strategy_params as SP
from tools import strategy_param_deap as D


def test_params_v2_canonical_file_is_the_default_document_and_pinned():
    doc = SP.load(C3.STRATEGY_PARAMS_FILE)
    assert doc == SP.default_doc() and doc["schema"] == "strategy_params.v2" and doc["exec_profile"] == "full_f4"
    assert SP.sha256_of(C3.STRATEGY_PARAMS_FILE) == C3.STRATEGY_PARAMS_SHA256
    for k, v in (("a_sl", 1.0), ("b_sl", 0.0), ("buffer_pips", 1.0), ("a_tp", 1.0), ("b_tp", 0.0), ("rho", 0.005), ("ref_p75", 0.0), ("horizon_rule", 0.0)):
        assert doc[k] == v and SP.BOUNDS[k][0] <= v <= SP.BOUNDS[k][1]
    assert SP.BOUNDS["rho"] == [0.001, 0.01] and SP.BOUNDS["a_sl"] == [0.5, 2.0] and SP.BOUNDS["buffer_pips"] == [0.0, 5.0]
    bad = dict(doc, exec_profile="full_f9")
    with pytest.raises(SP.StrategyParamsError):
        SP.validate(bad)


def test_exec_variant_of_a_document_defaults_to_full_f3_and_carries_rho_under_full_f4():
    legacy_doc = {k: v for k, v in SP.default_doc().items() if k != "exec_profile"}
    assert C3.exec_variant_of(legacy_doc) is C3.FULL_F3 and C3.exec_variant_of(None) is C3.FULL_F3 and C3.FULL_F3.identity == "exec_709f9b1fa3512486"
    assert C3.exec_variant_of(dict(legacy_doc, exec_profile="full_f3")) is C3.FULL_F3
    xv = C3.exec_variant_of(dict(SP.default_doc(), rho=0.0071))
    assert xv.sizing_mode == "risk_budget" and xv.rho == 0.0071 and xv.leverage == 50.0 and xv.no_daily_cap and xv.level_mode == "absolute_pips"
    assert xv.identity != C3.FULL_F4.identity and C3.exec_variant_of(SP.default_doc()).identity == C3.FULL_F4.identity   # rho is part of the identity when it differs from the baseline


def test_sealed_c3_contracts_are_the_full_f4_v2_with_the_business_contract_block():
    for name, arch in (("ACTION_CONTRACT_C3.json", C3.ARCH_TCN), ("ACTION_CONTRACT_C3_U.json", C3.ARCH_U)):
        d = json.loads(open("docs/short_horizon/clean_round/contracts/" + name).read())
        assert d["identity"] == ACT.identity_of(d) == C3.default_contract(arch, exec_profile="full_f4")["identity"]
        assert d["exec_variant"]["name"] == "full_f4" and d["exec_variant"]["identity"] == C3.FULL_F4.identity
        bc = d["business_contract"]
        assert bc["sha256"] == "73f32e43005da04ae44bcfffb576624513e8040aa8df3f06c521be5b7495b036" and bc["profile"]["leverage"] == 50.0 and bc["profile"]["max_entries_per_day"] is None
        assert bc["levels"]["mode"] == "absolute_pips" and bc["horizons"]["members"]["mt24h"] == {"depth": 12, "backstep": 11, "hours": 24} and "INERT" in bc["horizons"]["deviation_status"]
        assert d["strategy"]["params"]["sha256"] == C3.STRATEGY_PARAMS_SHA256 and d["changelog"][0]["when"] == "2026-10-29"
    legacy = C3.default_contract(C3.ARCH_TCN)                        # the default argument stays the audited profile
    assert legacy["exec_variant"]["name"] == "full_f3" and "business_contract" not in legacy


def test_deap_genome_v2_carries_the_contract_genes_with_binary_options():
    for k in ("a_sl", "b_sl", "buffer_pips", "a_tp", "b_tp", "rho", "ref_p75", "horizon_rule"):
        assert k in D.GENOME
    assert {"ref_p75", "horizon_rule"} <= D.BINARY
    base = SP.default_doc()
    x = [0.5] * len(D.GENOME)
    x[D.GENOME.index("ref_p75")], x[D.GENOME.index("horizon_rule")], x[D.GENOME.index("rho")] = 0.9, 0.1, 0.0
    doc = D.decode(x, base)
    assert doc["ref_p75"] == 1.0 and doc["horizon_rule"] == 0.0 and doc["rho"] == 0.001 and doc["exec_profile"] == "full_f4"
    assert D.decode(D.encode(base), base)["rho"] == pytest.approx(0.005)
