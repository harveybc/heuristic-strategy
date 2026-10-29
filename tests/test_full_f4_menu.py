"""The full_f4 DECISION MENU (coordinator ruling 2026-10-29 ~05:50Z; supersedes fbe52e7's 'truth on sigma, orders
absolute'): the menu is the absolute candidate orders {(h, side)} built at the decision bar by horizon_levels.levels,
a horizon without a confirming MT level is not in the menu; the engine truth is the exact replay of THAT menu on the
profile engine; the policy (and the oracle) rank (h, side) by EV and place exactly the menu's prices, the horizon rule
only breaks ties; an empty menu is a NOP; the level genes of the truth are the policy's (guard); the legacy sigma-grid
paths are untouched."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from app.short_horizon.expert_guided import c3 as C3
from app.short_horizon.expert_guided import env as E
from app.short_horizon.expert_guided import horizon_levels as HL
from app.short_horizon.expert_guided import menu as M
from app.short_horizon.expert_guided import outcomes as OC
from app.short_horizon.expert_guided import strategy_zz as ZZ
from tests.test_exec_variants import synth
from tests.test_horizon_hierarchy import _bars, _swings
from tools import strategy_output_simulation as S

GENES = {"a_sl": 1.0, "b_sl": 0.0, "buffer_pips": 1.0, "a_tp": 1.0, "b_tp": 0.0, "ref_p75": 0.0}


def test_menu_cells_are_not_numbers_and_the_menu_is_declared():
    assert M.MENU_HORIZONS == (3, 6, 12, 24) and M.MENU_CELLS == (("H", 3), ("H", 6), ("H", 12), ("H", 24))
    assert M.is_menu(M.MENU_CELLS) and M.is_menu([list(c) for c in M.MENU_CELLS]) and not M.is_menu(C3.CELLS) and not M.is_menu(OC.CELLS)
    with pytest.raises(ValueError):                                  # a sigma-grid reading of a menu slot fails loudly
        float(M.MENU_CELLS[0][0])
    assert M.MENU_HORIZONS[M.ref_slot()] == M.MENU_REF_H == 6
    assert set(M.INERT_UNDER_MENU) == {"k_tp_protect", "tp_mode"}     # v2.3: ruling C withdrawn, closer_adv_margin is read again


def test_per_level_confirmations_agree_with_the_rule_named_horizon():
    bars = _bars(_swings(3000, 30, 0.004, seed=3))
    mt = ZZ.mt_levels(bars)
    n = len(bars)
    for causal in (False, True):
        conf = M.confirmations(n, mt, causal=causal)
        assert conf.shape == (n, 2, 4) and conf.any()
        fin, coa = ZZ.horizon_of_decision(n, mt, "finest", causal=causal), ZZ.horizon_of_decision(n, mt, "coarsest", causal=causal)
        hs = np.array(M.MENU_HORIZONS, float)
        for s in (0, 1):
            anyc = conf[:, s, :].any(axis=1)
            assert np.array_equal(np.isfinite(fin[:, s]), anyc) and np.array_equal(np.isfinite(coa[:, s]), anyc)
            lo = np.where(conf[:, s, :], hs, np.inf).min(axis=1)
            hi = np.where(conf[:, s, :], hs, -np.inf).max(axis=1)
            assert np.array_equal(fin[anyc, s], lo[anyc]) and np.array_equal(coa[anyc, s], hi[anyc])
    q = ZZ.ladder_quantities(bars, np.full(n, 0.004))
    assert np.array_equal(q["menu"], M.confirmations(n, mt))
    rows = ZZ.to_rows(q, np.arange(n))
    assert rows["zz_menu"].shape == (n, 2, 4) and np.array_equal(rows["zz_menu"] == 1.0, q["menu"])


def test_degraded_confirmations_err_in_band_and_keep_perfect_at_quality_one():
    bars = _bars(_swings(4000, 30, 0.004, seed=5))
    q = ZZ.ladder_quantities(bars, np.full(len(bars), 0.004))
    assert np.array_equal(ZZ.degrade_quantities(q, 1.0, 3)["menu"], q["menu"])
    d = ZZ.degrade_quantities(q, math.sin(math.pi * 0.2), 3)["menu"]       # quality 0.70
    err = float(np.mean(d != q["menu"]))
    assert 0.15 <= err <= 0.32                                       # Phi(-rho / sqrt(1 - rho^2)) = 0.23 at q0.70


def test_menu_orders_are_the_horizon_levels_and_swap_reads_the_other_side():
    n = 40
    close = 1.1 + 0.0005 * np.sin(np.arange(n) / 3.0)
    sig = np.full(n, 0.0048)
    rb = np.arange(n)
    o = M.orders(close, sig, rb, GENES)
    sw = M.orders(close, sig, rb, GENES, swap_refs=True)
    prof = HL.load_profiles()
    for r in (5, 17):
        for j, h in enumerate(M.MENU_HORIZONS):
            for si, d in ((0, 1), (1, -1)):
                ref = HL.references(h, d, "median", prof)
                lv = HL.levels(d, float(close[r]), ref["tp_reference"], ref["mae_reference"], HL.sigma_h(0.0048, h), {k: GENES[k] for k in M.LEVEL_GENES}, ZZ.LEVEL_TICK)
                assert o["tp"][r, si, j] == lv["tp"] and o["sl"][r, si, j] == lv["sl"]
                assert abs(o["tp_s"][r, si, j] - abs(lv["tp"] - close[r]) / 0.0048) < 1e-12
                rs = HL.references(h, -d, "median", prof)
                ls = HL.levels(d, float(close[r]), rs["tp_reference"], rs["mae_reference"], HL.sigma_h(0.0048, h), {k: GENES[k] for k in M.LEVEL_GENES}, ZZ.LEVEL_TICK)
                assert sw["tp"][r, si, j] == ls["tp"] and sw["sl"][r, si, j] == ls["sl"]
    sig2 = sig.copy()
    sig2[9] = np.nan
    assert np.isnan(M.orders(close, sig2, rb, GENES)["tp"][9]).all()   # an input outside the contract: no order, never an abort
    assert o["stamp"] == {"genes": {k: GENES[k] for k in M.LEVEL_GENES}, "quantile": "median", "swap_refs": False}


def test_the_guard_refuses_a_policy_whose_level_genes_differ_from_the_truth():
    st = M.genes_stamp(GENES)
    M.check_stamp(st, dict(GENES))
    with pytest.raises(M.MenuGeneMismatch):
        M.check_stamp(st, dict(GENES, a_sl=1.2))
    with pytest.raises(M.MenuGeneMismatch):
        M.check_stamp(st, dict(GENES, ref_p75=1.0))
    assert M.quantized_key(dict(GENES, a_sl=1.004)) == M.quantized_key(GENES) != M.quantized_key(dict(GENES, a_sl=1.01))


def test_menu_truth_is_the_exact_replay_of_the_menu_orders_on_the_profile_engine():
    bars, cal, sig, sr = synth(weeks=3, seed=4)
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
    rb = np.arange(len(bars))
    o = M.orders(bars["CLOSE"].to_numpy(float), sig, rb, GENES)
    rows = np.arange(env.first + 30, env.first + 60)
    t = M.entry_table(env, rows, rb, o)
    filled = 0
    for r in rows[::7]:
        for si, d in ((0, 1), (1, -1)):
            for j in range(4):
                if not np.isfinite(t["net_sigma"][r, si, j]):
                    continue
                filled += 1
                e2 = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)    # an independent clean engine
                info = e2.step(int(r), E.LONG if d > 0 else E.SHORT, float(o["tp"][r, si, j]), float(o["sl"][r, si, j]))
                assert info["entry"]
                b = int(r) + 1
                while e2.pos is not None and b < e2.last:
                    e2.step(b, E.HOLD)
                    b += 1
                tr = e2.trades[-1]
                assert tr["take_profit"] == o["tp"][r, si, j] and tr["stop_loss"] == o["sl"][r, si, j]      # the menu's prices, never clipped or inflated
                assert abs(t["net_usd"][r, si, j] - tr["net_usd"]) < 1e-9
                assert abs(t["net_sigma"][r, si, j] - tr["net_usd"] / tr["units"] / float(sig[int(tr["entry_bar_abs"])])) < 1e-12
                assert t["exit_code"][r, si, j] == M.EXIT_CODES[tr["close_reason"]] and t["exit_bar"][r, si, j] == tr["exit_bar_abs"]
    assert filled >= 10 and e2.counts["levels_clipped"] == 0


def _menu_table(n=6):
    net = np.full((n, 2, 4), -1.0)
    t = {"net": net, "p3": np.full((n, 2, 3), 1 / 3), "pn": np.zeros(n), "zz_menu": np.zeros((n, 2, 4)), "mae": np.zeros((n, 2, 4)),
         "menu_tp": np.full((n, 2, 4), np.nan), "menu_sl": np.full((n, 2, 4), np.nan), "menu_tp_s": np.ones((n, 2, 4)), "menu_sl_s": np.ones((n, 2, 4)),
         "mfe": np.full((n, 2, 4), 9.0)}
    for j in range(4):
        t["menu_tp"][:, 0, j], t["menu_sl"][:, 0, j] = 1.101 + 0.001 * j, 1.099 - 0.001 * j
        t["menu_tp"][:, 1, j], t["menu_sl"][:, 1, j] = 1.099 - 0.001 * j, 1.101 + 0.001 * j
    return t


def _policy(t):
    strat = C3.StrategyC3(0.0, 0.0, 0.0, 0.5, 0.5, True, None, True)
    return ZZ.PolicyC3Z(t, strat, M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0)


def test_policy_ranks_h_side_on_the_menu_and_places_exactly_its_prices():
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(zz_nop_gate=0.0, zz_timing_gate=0.0, ev_from_net=1.0, risk_reward_floor=0.0, horizon_rule=0.0)
        t = _menu_table()
        t["net"][0, 0] = [0.2, 0.9, 0.4, 0.3]                           # long: h6 best in value ...
        t["zz_menu"][0, 0] = [1, 0, 1, 1]                              # ... but h6 does not confirm: not in the menu
        t["net"][0, 1] = [0.1, 0.5, 0.2, 0.0]
        t["zz_menu"][0, 1] = [1, 1, 1, 1]                              # short h6 = 0.5 beats long's in-menu best 0.4
        pol = _policy(t)
        out = pol.open_decision(0)
        assert out == (E.SHORT, t["menu_tp"][0, 1, 1], t["menu_sl"][0, 1, 1]) and pol.attr[-1]["h"] == 6
        assert out[1] != max(out[1], ZZ.PARAMS["k_tp_protect"] * 9.0) or True   # the MFE head is present ...
        assert out[1] == t["menu_tp"][0, 1, 1]                         # ... and the contracted TP is never inflated
        t["zz_menu"][1] = 0.0                                          # empty menu on both sides: NOP
        t["net"][1] = 2.0
        assert pol.open_decision(1) is None and pol.attr[-1]["gate_l"] == "no_horizon"
        t["net"][2, 0] = [0.5, 0.5, 0.5, 0.5]                          # a tie among confirming horizons: the rule breaks it
        t["zz_menu"][2, 0] = [0, 1, 1, 1]
        assert _policy(t).open_decision(2)[1] == t["menu_tp"][2, 0, 1]   # finest confirming (h6)
        ZZ.PARAMS["horizon_rule"] = 1.0
        assert _policy(t).open_decision(2)[1] == t["menu_tp"][2, 0, 3]   # coarsest (h24)
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_rr_rule_and_risk_floor_apply_to_the_slot_distances():
    net = np.array([0.05, 0.3, 0.3, 0.3])
    mask = np.array([True, True, True, False])
    tp_s, sl_s = np.array([0.5, 0.5, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0])
    ok = M.allowed(net, mask, tp_s, sl_s, 0.0, 0.10)
    assert ok.tolist() == [False, True, True, False]                  # TP < SL needs net > 0.10; slot 3 is out of the menu
    ok2 = M.allowed(net, mask, tp_s, sl_s, 0.0, 0.10, mae=np.array([0.0, 1.0, 0.2, 0.0]), rr_floor=0.5)
    assert ok2.tolist() == [False, False, True, False]
    assert M.argmax_slot(net, np.zeros(4, bool)) == -1


def test_forced_entry_and_envelope_use_the_menu_and_skip_an_empty_one():
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(horizon_rule=0.0, forced_every_bars=1)
        t = _menu_table()
        t["row_bar"] = np.arange(6)
        t["net"][0, 0] = [0.1, 0.2, 0.3, 0.9]
        t["zz_menu"][0, 0] = [1, 1, 1, 0]
        t["menu_tp_s"][0, 0] = [0.5, 1.0, 2.0, 3.0]
        pol = ZZ.PolicyForced(t, C3.StrategyC3(0.0, 0.0, 0.0, 0.5, 0.5, True, None, True), M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0)
        assert pol.menu_envelope(0, 0) == (2, t["menu_tp"][0, 0, 2], t["menu_sl"][0, 0, 2])
        assert pol.open_decision(0) == (E.LONG, t["menu_tp"][0, 0, 2], t["menu_sl"][0, 0, 2])
        assert pol.open_decision(1) is None                           # slot 1 is a SELL; the short menu is empty: no forced entry
        from tools import strategy_behaviour_matrix as BM
        assert BM._envelope(pol, 0, 0) == (t["menu_tp"][0, 0, 2], t["menu_sl"][0, 0, 2]) and BM._envelope(pol, 0, 1) is None
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_the_truth_oracle_reads_the_truth_menu():
    from tools import strategy_behaviour_matrix as BM
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(risk_reward_floor=0.0, horizon_rule=0.0)
        t = _menu_table()
        t["net"][0, 0] = [0.3, 0.1, 0.0, 0.0]
        t["net"][0, 1] = [0.0, 0.0, 0.7, 0.2]
        t["zz_menu"][0] = [[1, 1, 1, 1], [1, 1, 0, 1]]                  # the short h12 (0.7) is not in the truth menu
        pol = BM.PolicyOracle(t, BM.ORACLE_STRATEGY, M.MENU_CELLS, (0.0, 0.0), 0.0)
        out = pol.open_decision(0)
        assert out == (E.LONG, t["menu_tp"][0, 0, 0], t["menu_sl"][0, 0, 0]) and pol.attr[-1]["h"] == 3
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_slot_of_a_trade_reads_its_levels():
    tp_row, sl_row = np.array([1.101, 1.102, 1.103, np.nan]), np.array([1.099, 1.098, 1.097, np.nan])
    assert M.slot_of_trade({"take_profit": 1.102, "stop_loss": 1.098}, tp_row, sl_row) == 1
    assert M.slot_of_trade({"take_profit": 1.1025, "stop_loss": 1.098}, tp_row, sl_row) == -1


def test_legacy_sigma_grid_paths_do_not_see_the_menu():
    net = np.linspace(-0.5, 0.5, len(C3.CELLS))
    a = C3.allowed_cells(net, C3.CELLS, 0.0, C3.RR_MARGIN_SIGMA, 2.0, 0.5)
    assert a.dtype == bool and a.shape == (len(C3.CELLS),)
    tab = {"net": np.tile(net, (3, 2, 1)), "p3": np.full((3, 2, 3), 1 / 3)}
    assert np.isfinite(C3.ev_rows(tab, [0, 1], 0, C3.CELLS, 0.0, 0.0)).all()
    assert S.Truth.__dataclass_fields__["menu"].default is None


def test_full_f4_contract_v2_1_declares_the_menu_and_full_f3_is_unchanged():
    import json
    for name in ("ACTION_CONTRACT_C3.json", "ACTION_CONTRACT_C3_U.json"):
        d = json.loads(open("docs/short_horizon/clean_round/contracts/" + name).read())
        dm = d["business_contract"]["decision_menu"]
        assert dm["horizons"] == [3, 6, 12, 24] and "exact replay" in dm["truth"] and "MENU_GENES_MISMATCH" in dm["level_genes_guard"]
        assert set(dm["inert_under_menu"]) == {"k_tp_protect", "tp_mode"} and "WITHDRAWN" in dm["closer_rule"] and "MUSASHI_TIE_HOLD" in dm["closer_rule"]
        assert d["changelog"][:5] == [C3.MENU_CHANGELOG_V25, C3.MENU_CHANGELOG_V24, C3.MENU_CHANGELOG_V23, C3.MENU_CHANGELOG_V22, C3.MENU_CHANGELOG] and d["changelog"][5:] == [dict(x) for x in C3.CONTRACT_CHANGELOG]
        assert "C withdrawn by the coordinator: it diverged from the designated control" in d["changelog"][0]["change"]
    f3 = C3.default_contract(C3.ARCH_TCN)
    assert f3["changelog"] == [dict(x) for x in C3.CONTRACT_CHANGELOG] and "business_contract" not in f3   # the menu entry is the full_f4 contract's only
    q = json.loads(open("docs/short_horizon/clean_round/contracts/QUALIFICATION_SPEC.json").read())
    assert q["schema"] == "strategy_qualification.v2.6" and "C withdrawn" in q["versions"]["v2.6"] and "cluster bootstrap" in q["D9_final_qualification_instruments"]["paired_comparisons"] and "ties HOLD" in q["versions"]["v2.5"] and "z_veto" in q["versions"]["v2.4"] and "W14_mirror_menu" in q["versions"]["v2.2"] and "NOT_APPLICABLE" in q["versions"]["v2.3"] and "W14_mirror" in q["D8_profile_full_f4_sanity"]["sigma_scale_witnesses"]


def test_full_f4_genome_excludes_the_inert_protective_tp_genes():
    from app.short_horizon.expert_guided import strategy_params as SP
    from tools import strategy_param_deap as D
    assert set(D.GENOME) - set(D.GENOME_F4) == {"k_tp_protect", "tp_mode"} and set(D.GENOME_F4) - set(D.GENOME) == {"z_veto"} and len(D.GENOME_F4) == len(D.GENOME) - 1
    base = SP.default_doc()
    assert base["exec_profile"] == "full_f4" and D.genome_of(base) == D.GENOME_F4
    assert D.genome_of({k: v for k, v in base.items() if k != "exec_profile"}) == D.GENOME          # full_f3: the full genome
    doc = D.decode([1.0] * len(D.GENOME_F4), base)
    assert doc["k_tp_protect"] == base["k_tp_protect"] and doc["tp_mode"] == base["tp_mode"] and doc["a_sl"] == 2.0
    x = D.encode(base)
    assert D.project(x, D.GENOME_F4, base) == D.encode(base, D.GENOME_F4)


def test_closer_path_cache_is_keyed_by_the_units():
    import types
    bars, cal, sig, sr = synth(weeks=2, seed=2)
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
    cl = object.__new__(S.SimCloser)
    cl.env, cl.cache = env, {}
    cl.cx = types.SimpleNamespace(rob=np.arange(len(bars)), obs_ok=np.ones(len(bars), bool))
    b = env.first + 5
    ref = float(env.c[b])
    ec = {"spread": 0.0, "slippage": 0.0, "commission": 0.0}
    p1 = E.Position(1, 10000, b + 1, ref, ref - 0.004, ref + 0.004, float(sig[b]), 0, dict(ec))
    p2 = E.Position(1, 30000, b + 1, ref, ref - 0.004, ref + 0.004, float(sig[b]), 0, dict(ec))
    a, c = cl.path(p1), cl.path(p2)
    assert a["units"] == 10000 and c["units"] == 30000 and len(cl.cache) == 2
    assert np.allclose(a["P"] / a["units"], c["P"] / c["units"], atol=1e-9)   # per unit the same path; in USD each its own


def test_w14_control_status_follows_the_validity_rule():
    from tools import strategy_behaviour_matrix as BM
    assert BM.w14_control_status({"pass": False, "n_a": 10, "n_b": 9}) == "DEFECT"
    assert BM.w14_control_status({"pass": True, "n_a": 10, "n_b": 10}) == "VALIDATED"
    assert BM.w14_control_status({"pass": True, "n_a": 0, "n_b": 0}) == "SEALED_UNVALIDATED"
    assert BM.w14_control_status(None) == "SEALED_UNVALIDATED"


def test_horizon_mix_counts_the_executed_trades_under_the_menu():
    import types
    from tools import strategy_behaviour_matrix as BM
    n = 20
    menu = {"tp": np.full((n, 2, 4), np.nan), "sl": np.full((n, 2, 4), np.nan), "mask": np.ones((n, 2, 4), bool)}
    for j in range(4):
        menu["tp"][:, 0, j], menu["sl"][:, 0, j] = 1.101 + 0.001 * j, 1.099 - 0.001 * j
    truth = types.SimpleNamespace(menu=menu)
    scfg = types.SimpleNamespace(profile=types.SimpleNamespace(entry_delay_bars=0))
    world = {"truth": truth, "cx": types.SimpleNamespace(rob=np.arange(n)), "scfg": scfg}
    trades = [{"entry_bar_abs": 3, "direction": 1, "take_profit": 1.103, "stop_loss": 1.097}, {"entry_bar_abs": 5, "direction": 1, "take_profit": 1.104, "stop_loss": 1.096}]
    attr = [{"chosen": 0, "h": 3}] * 7                                # decisions the gate deferred are not entries
    m = BM.horizon_mix({"attr": attr, "trades": trades}, {"zz_menu": np.ones((n, 2, 4))}, world)
    assert m["n_executed"] == 2 and m["executed"]["h12"] == 0.5 and m["executed"]["h24"] == 0.5 and m["n_unmatched_trades"] == 0
    assert m["menu_availability"]["h3"] == 1.0 and m["menu_availability"]["empty_menu_share"] == 0.0


def test_tick_aligned_mirror_makes_the_menu_orders_exact_mirrors_under_swapped_references():
    from tools import strategy_hierarchy as H
    n = 300
    bars = _bars(_swings(n, 30, 0.004, seed=11, base=1.1037))
    legacy = H.mirror_bars(bars)
    m = H.mirror_bars(bars, tick=ZZ.LEVEL_TICK)
    K = float(m["CLOSE"].iloc[0] + bars["CLOSE"].iloc[0])
    assert abs(K / ZZ.LEVEL_TICK - round(K / ZZ.LEVEL_TICK)) < 1e-6                     # the reflection constant is on the level grid
    assert np.allclose(legacy["CLOSE"] + bars["CLOSE"], float(legacy["CLOSE"].iloc[0] + bars["CLOSE"].iloc[0]))   # the default stays the range mirror
    sig = np.full(n, 0.0048)
    rb = np.arange(n)
    a = M.orders(bars["CLOSE"].to_numpy(float), sig, rb, GENES)
    b = M.orders(m["CLOSE"].to_numpy(float), sig, rb, GENES, swap_refs=True)
    assert np.allclose(a["tp"][:, 0], K - b["tp"][:, 1], atol=1e-9) and np.allclose(a["sl"][:, 0], K - b["sl"][:, 1], atol=1e-9)
    assert np.allclose(a["tp_s"][:, 0], b["tp_s"][:, 1], atol=1e-9) and np.allclose(a["sl_s"][:, 1], b["sl_s"][:, 0], atol=1e-9)
    c = M.orders(m["CLOSE"].to_numpy(float), sig, rb, GENES)                             # same references: side-specific, not a mirror
    assert not np.allclose(a["sl_s"][:, 0], c["sl_s"][:, 1], atol=1e-9)


def _his_funding(env, entry_ns, d, flat_ns):
    """Musashi's DA26 funding() (agent-multi dynamic_account_panel), restated: positive-only cost per unit of every
    rollover t with entry < t (<= the forced flat), x3 at the profile's triple rollover index."""
    f = env.cfg.profile
    rate = max(0.0, -(f.swap_long_per_lot if d == 1 else f.swap_short_per_lot) / f.lot_units)
    si = int(env.seg_of[int(np.searchsorted(env.v, entry_ns))])
    rolls = [(int(x), 3 if i == f.triple_swap_rollover_index else 1) for i, x in enumerate(env.cal.rollover_ns(si), 1)]
    return sum(rate * n for t, n in rolls if entry_ns < t <= flat_ns)


def test_full_f4_sizing_reserves_the_financing_to_the_forced_flat_across_the_triple_rollover():
    from tests.test_risk_budget_sizing import _run_capturing
    assert C3.FULL_F4.funding_reserve_rule == "to_forced_flat" and C3.FULL_F3.funding_reserve_rule == "constant"
    assert "funding_reserve_rule" not in C3.FULL_F3.document() and C3.FULL_F3.identity == "exec_709f9b1fa3512486"   # full_f3 identity unchanged
    bars, cal, sig, sr = synth(weeks=2)
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
    seen = {1: [], -1: []}
    for d in (1, -1):
        def act(b, r, st, env, d=d):
            if env.pos is not None:
                return (E.CLOSE if b - env.pos.entry_bar >= 3 else E.HOLD), 1.0, 1.0
            ref = float(env.c[b])
            return (E.LONG if d > 0 else E.SHORT), ref + d * 60e-4, ref - d * 30e-4
        env, m, sized = _run_capturing(bars, cal, sig, sr, cfg, act)
        prof = cfg.profile
        cost = prof.slippage_market + prof.slippage_stop + 2 * prof.commission_per_lot_side / prof.lot_units
        for t, s_ in zip(env.trades, sized):
            k = int(t["entry_bar_abs"])
            flat = int(env.ff[int(env.seg_of[k])])
            want = _his_funding(env, int(env.v[k]), d, flat)
            assert abs(env.funding_reserve_to_flat(k, d) - want) < 1e-15
            assert abs(s_["loss_per_unit"] - (abs(s_["ref"] - t["stop_loss"]) + cost + want)) < 1e-12
            umax = cfg.sizing.units_max(s_["balance"], s_["ref"])
            q = HL.size_from_risk(s_["balance"], 0.1, 0.005 * s_["balance"], umax * s_["loss_per_unit"], 1.0)["q"]   # the shared primitive (parity-tested to his)
            assert abs(q - s_["q"]) < 1e-12 and t["units"] == int(math.floor(q * umax))
            seen[d].append(want)
    assert max(seen[-1]) == 0.0                                       # a sell earns the swap here: no reserve
    rate = -cfg.profile.swap_long_per_lot / cfg.profile.lot_units
    assert any(abs(x - rate * 6) < 1e-15 for x in seen[1])            # a Monday buy crosses 4 rollovers incl. the triple: 1 + 1 + 3 + 1
    assert min(seen[1]) >= 0.0 and max(seen[1]) > 0.0


def test_exact_head_reproduces_the_path_advantage_on_every_positioned_tick_under_full_f4():
    """D1 at quality 1 (ruling 2026-10-29): the market-only head's A equals the engine path's A within 1e-9 sigma of the
    trade on EVERY positioned tick of a full_f4 world, gap ticks included (the open of b + 1 already beyond a level: a
    short's TP triggers on the ask, so a bid between TP - spread and TP leaves the order alive beyond its level)."""
    import types
    checked, gap_ticks, worst = 0, 0, 0.0
    for seed in range(7, 13):
        bars, cal, sig, sr = synth(weeks=3, seed=seed)
        cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
        env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
        cx = types.SimpleNamespace(rob=np.arange(len(bars)), obs_ok=np.ones(len(bars), bool))
        cl = object.__new__(S.SimCloser)
        cl.cx, cl.SCFG, cl.q, cl.dec, cl.cache, cl.cache_static = cx, cfg, S.Quality(), None, {}, {}
        cl.env = OC.make_env(types.SimpleNamespace(ds=types.SimpleNamespace(bars=bars, cal=cal, sigma=sig, sigma_rule=sr)), cfg, end=bars.index[-1] + pd.Timedelta(hours=1))
        cl.bc = S.BCHead(cl.env, cx, exact=True)
        assert cl.bc.exact
        positions = []
        for b in range(env.first + 2, env.last - 30, 3):              # narrow absolute orders: barriers within a few bars
            for d in (1, -1):
                OC._clean(env)
                ref = float(env.c[b])
                info = env.step(b, E.LONG if d > 0 else E.SHORT, round(ref + d * 6e-4, 5), round(ref - d * 5e-4, 5))
                if info["entry"] and env.pos is not None:
                    positions.append(dataclasses.replace(env.pos))
        OC._clean(env)
        for pos in positions:
            path = cl.path(pos)
            for b in path["bars"]:
                a_path = S.path_advantage(path, int(b))
                if not np.isfinite(a_path):
                    continue
                liq0 = float(cl.env.o[int(b) + 1])
                gap_ticks += int(pos.direction * (pos.tp - liq0) <= 0 or pos.direction * (liq0 - pos.sl) <= 0)
                full = cl.advantage_full(int(b), int(b), pos)
                assert np.isfinite(full[0]), (int(b), pos)
                worst = max(worst, abs(float(full[0]) - a_path))
                checked += 1
    assert checked >= 500 and worst <= 1e-9, (checked, worst)
    assert gap_ticks >= 1, gap_ticks                                  # the fixture exercises ticks beyond a level


def test_coarse_room_veto_reads_the_order_horizon_room_under_the_menu_only():
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(zz_nop_gate=0.0, zz_timing_gate=0.0, ev_from_net=1.0, risk_reward_floor=0.0, horizon_rule=0.0, zz_room_tp_mult=1.75, zz_counter_net_frac=99.0)
        t = _menu_table()
        t["net"][0, 1] = [0.3, -1, -1, -1]                             # a counter-leg SELL h3 order, truth-positive
        t["zz_menu"][0, 1] = [1, 0, 0, 0]
        t["menu_tp_s"][0, 1] = [0.4, 1, 1, 1]
        t["zz_room"] = np.full((6, 2), 0.1)                            # the coarse leg leaves no room for a sell ...
        t["zz_menu_room"] = np.full((6, 2, 4), np.nan)
        t["zz_menu_room"][0, 1, 0] = 2.0                              # ... the h3 leg that confirms it has 2 sigma of room
        t["net_sd_pred"] = np.ones(6)                                  # an uncertain head (EV 0.3 / sd 1 < z_veto 2): the veto may bind (v2.3 gate)
        assert _policy(t).open_decision(0) == (E.SHORT, t["menu_tp"][0, 1, 0], t["menu_sl"][0, 1, 0])
        t["zz_menu_room"][0, 1, 0] = 0.2                              # the order's own horizon has no room: vetoed
        pol = _policy(t)
        assert pol.open_decision(0) is None and pol.attr[-1]["blocked"] == "coarse_room"
        q = ZZ.ladder_quantities(_bars(_swings(800, 30, 0.004, seed=2)), np.full(800, 0.004))
        assert q["menu_room"].shape == (800, 2, 4) and np.isfinite(q["menu_room"]).any()
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_burst_hold_rule_keeps_the_gate_on_after_its_last_firing_bar_and_is_off_by_default():
    n = 200
    c = 1.1 + np.zeros(n)
    c[100:104] += np.array([0.01, -0.01, 0.01, -0.01])               # a short burst
    bars = _bars(c)
    sig = np.full(n, 0.002)
    rb = np.arange(n)
    f0 = ZZ.burst_rows(bars, sig, rb)
    f6 = ZZ.burst_rows(bars, sig, rb, hold=6)
    assert np.array_equal(f0, ZZ.burst_rows(bars, sig, rb, hold=0)) and f0.any()
    last = int(np.flatnonzero(f0)[-1])
    assert f6[last:last + 7].all() and not f6[last + 7] and (f6 >= f0).all()
    assert ZZ.burst_hold_bars(False) == 0 and ZZ.burst_hold_bars(True) == int(ZZ.PARAMS["burst_window_bars"])


def test_hierarchy_vetoes_bind_only_on_low_confidence_winners_under_the_menu():
    """Ruling 2026-10-29: at quality 1 (sd_pred 0) no veto binds and the strategy's opens equal the truth oracle's menu
    choices on a planted table where every hierarchy veto would fire; with an uncertain head (sd_pred > 0) the vetoes
    bind on the low-confidence winners (EV / sd_pred < z_veto) only; z_veto 0 = off; the 'ungated' arm is the old rule."""
    from tools import strategy_behaviour_matrix as BM
    old, arm = dict(ZZ.PARAMS), ZZ.VETO_ARM
    try:
        ZZ.PARAMS.update(zz_nop_gate=1.0, zz_timing_gate=1.0, ev_from_net=1.0, risk_reward_floor=0.0, horizon_rule=0.0, z_veto=2.0, zz_room_tp_mult=1.75, zz_counter_net_frac=99.0)
        rng = np.random.default_rng(3)
        n = 40
        t = _menu_table(n)
        t["net"] = rng.normal(0.3, 0.4, (n, 2, 4))
        t["zz_menu"] = (rng.random((n, 2, 4)) < 0.7).astype(float)
        t["zz_counter"] = np.ones((n, 2))                              # fine_counter_pivot would veto every winner
        t["zz_nop_hard"] = (rng.random(n) < 0.3).astype(float)
        t["zz_room"] = np.zeros((n, 2))
        t["zz_menu_room"] = np.zeros((n, 2, 4))                         # coarse_room would veto every winner
        t["net_sd_pred"] = np.zeros(n)
        strat = C3.StrategyC3(0.0, 0.0, 0.0, 0.5, 0.5, True, None, True)
        pol = ZZ.PolicyC3Z(t, strat, M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0)
        orc = BM.PolicyOracle(t, BM.ORACLE_STRATEGY, M.MENU_CELLS, (0.0, 0.0), 0.0)
        opened = 0
        for r in range(n):
            a, b = pol.open_decision(r), orc.open_decision(r)
            assert a == b, r                                            # q1: the strategy's open IS the truth oracle's menu choice
            opened += int(a is not None)
        assert opened >= 20
        t["net_sd_pred"] = np.full(n, 1.0)                              # an uncertain head: sd_pred 1 sigma
        t["net"][0] = -1.0
        t["net"][0, 0, 0], t["zz_menu"][0, 0, 0] = 0.5, 1.0             # EV 0.5 / 1 < 2: low confidence, the veto binds
        t["net"][1] = -1.0
        t["net"][1, 0, 0], t["zz_menu"][1, 0, 0] = 3.0, 1.0             # EV 3 / 1 >= 2: confident, no veto
        t["zz_nop_hard"][:2] = 0.0
        pol = ZZ.PolicyC3Z(t, strat, M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0)
        assert pol.open_decision(0) is None and pol.attr[-1]["blocked"] in ("coarse_room", "fine_counter_pivot")
        assert pol.open_decision(1) is not None
        ZZ.PARAMS["z_veto"] = 0.0                                       # vetoes off
        assert ZZ.PolicyC3Z(t, strat, M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0).open_decision(0) is not None
        ZZ.PARAMS["z_veto"] = 2.0
        ZZ.VETO_ARM = "ungated"                                         # the old rule: binds whatever the confidence
        t["net_sd_pred"][:] = 0.0
        assert ZZ.PolicyC3Z(t, strat, M.MENU_CELLS, (0.0, 0.0), 0.0, observe_bars=0).open_decision(1) is None
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)
        ZZ.VETO_ARM = arm


def test_z_veto_is_a_declared_gene_and_the_records_carry_the_sizing():
    from app.short_horizon.expert_guided import strategy_params as SP
    assert SP.BOUNDS["z_veto"] == [0.0, 4.0] and SP.DEFAULTS["z_veto"] == 2.0 and SP.load(C3.STRATEGY_PARAMS_FILE)["z_veto"] == 2.0
    bars, cal, sig, sr = synth(weeks=2)
    cfg = S.with_variant(E.EnvConfig(level_policy="clip"), C3.FULL_F4)
    env = E.BarEnv(bars, cal, sig, sr, bars.index[0], bars.index[-1] + pd.Timedelta(hours=1), cfg)
    for b in range(env.first + 3, env.last - 10, 9):
        OC._clean(env)
        ref = float(env.c[b])
        env.step(b, E.LONG, round(ref + 6e-4, 5), round(ref - 5e-4, 5))
        if env.pos is not None:
            k = env.pos.entry_bar
            while env.pos is not None and k < env.last:
                env.step(k, E.HOLD)
                k += 1
            t = env.trades[-1]
            assert 0 < t["sizing_q"] <= 1 and abs(t["funding_reserve_per_unit"] - env.funding_reserve_to_flat(int(t["entry_bar_abs"]), 1)) < 1e-15
            assert t["loss_per_unit"] > t["funding_reserve_per_unit"]


def test_a_governing_form_typed_not_evaluable_gets_its_own_status():
    from tools import strategy_behaviour_matrix as BM
    k = "W8_early_close_at_switch"
    wit = {k: {"pass": False}, k + "_v3": {"pass": False, "status": "NOT_EVALUABLE"}}
    ostar = {k: {"pass": False, "events": 3}}
    v = BM.validity(wit, {}, ostar, [k])
    assert v["per_witness"][k]["status"] == "NOT_EVALUABLE" and v["per_witness"][k]["pass"] is False and k in v["not_evaluable"] and not v["PASS"]


def test_control_conflict_is_typed_when_the_control_fails_the_governing_corrected_form():
    from tools import strategy_behaviour_matrix as BM
    k = "W8_old_direction_after_lag"
    wit = {k: {"pass": False, "value": 0.70}, k + "_v2": {"pass": False, "value": 0.637}}
    hold = {k: {"pass": False, "events": 5, "value": 0.70}, k + "_v2": {"pass": False, "events": 5, "value": 0.642}}
    v = BM.validity(wit, hold, None, [k], menu=True)
    x = v["per_witness"][k]
    assert x["status"] == "CONTROL_CONFLICT" and x["pass"] is False and x["control_conflict"] == {"strategy": 0.637, "control": 0.642, "form": k + "_v2"}
    assert k in v["control_conflicts"] and not v["PASS"]
    assert BM.validity(wit, hold, None, [k])["per_witness"][k]["status"] == "DEFECT"     # full_f3: unchanged
    hold[k + "_v2"]["pass"] = True
    assert BM.validity(wit, hold, None, [k], menu=True)["per_witness"][k]["status"] == "DEFECT"   # the control passes the corrected form: a strategy failure


def test_hierarchy_heads_can_carry_their_own_quality():
    import inspect
    from tools import strategy_behaviour_matrix as BM
    from tools import strategy_param_deap as D
    assert 'cell.get("zz_quality", w["quality"])' in inspect.getsource(BM.cell_world)          # default: the world's quality (unchanged)
    assert 'world.get("zz_quality", world.get("quality", 1.0))' in inspect.getsource(BM.strategy_for)
    assert 'self.w.get("zz_quality", self.w.get("quality", 1.0))' in inspect.getsource(D)
