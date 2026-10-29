"""S4 horizons in the hierarchy (business contract): the four MetaTrader members carry Musashi's TRAIN champions
(deviation a declared inert constant), are mirror-symmetric, name a decision's horizon by the finest confirming level
(HORIZON_RULE, one test per horizon on planted paths), none confirming = no horizon; in the absolute-level profile the
policy emits exactly horizon_levels.levels(h, side, quote, sigma_h, genes) prices; the legacy ladder is byte-identical."""
import math

import numpy as np
import pandas as pd

from app.short_horizon import zigzag as Z
from app.short_horizon.expert_guided import horizon_levels as HL
from app.short_horizon.expert_guided import strategy_zz as ZZ


def _idx(n):
    return pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC")


def _bars(c):
    c = np.asarray(c, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"OPEN": o, "HIGH": np.maximum(o, c) + 1e-4, "LOW": np.minimum(o, c) - 1e-4, "CLOSE": c}, index=_idx(len(c)))


def _swings(n, period, amp, seed=1, base=1.1):
    """A path whose pivots are spaced ~period bars apart with amplitude amp (plus small noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return base + amp * np.sign(np.sin(2 * np.pi * t / period)) * (1 - np.exp(-((t % period) + 1) / (period / 4))) + rng.normal(0, amp / 40, n)


def test_mt_members_carry_the_champions_and_are_mirror_symmetric():
    assert ZZ.MT_HORIZONS == {"mt3h": (3, 1, 3), "mt6h": (4, 2, 6), "mt12h": (7, 6, 12), "mt24h": (12, 11, 24)} and ZZ.MT_DEVIATION_PIPS == 5.0
    assert "mt3h" not in ZZ.DEFAULT_LEVELS                           # the legacy ladder quantities stay byte-identical
    bars = _bars(_swings(2000, 40, 0.004))
    mt = ZZ.mt_levels(bars)
    assert all(len(mt[k]["pivots"]) > 10 for k in mt) and all(c == -1 or c > p[0] for k in mt for p, c in zip(mt[k]["pivots"], mt[k]["conf"]))
    sig = np.full(len(bars), 0.004)
    h = ZZ.horizon_of_decision(len(bars), mt)
    mirror = bars.copy()
    mirror["OPEN"], mirror["HIGH"], mirror["LOW"], mirror["CLOSE"] = 2.2 - bars["OPEN"], 2.2 - bars["LOW"], 2.2 - bars["HIGH"], 2.2 - bars["CLOSE"]
    hm = ZZ.horizon_of_decision(len(bars), ZZ.mt_levels(mirror))
    assert np.array_equal(h[:, 0], hm[:, 1], equal_nan=True) and np.array_equal(h[:, 1], hm[:, 0], equal_nan=True)
    q = ZZ.ladder_quantities(bars, sig)
    assert q["horizon"].shape == (len(bars), 2) and np.array_equal(q["horizon"], h, equal_nan=True)
    rows = ZZ.to_rows(q, np.arange(len(bars)))
    assert rows["zz_h"].shape == (len(bars), 2) and np.allclose(rows["zz_sigma"], 0.004)


def _reference_rule(n, mt, rule, causal=False):
    """An independent per-bar reading of HORIZON_RULE (loops, no vectorisation)."""
    out = np.full((n, 2), np.nan)
    for b in range(n):
        for side, kind_wanted in ((0, Z.VALLEY), (1, Z.PEAK)):
            cands = []
            for name, (_, _, h) in ZZ.MT_HORIZONS.items():
                m = mt[name]
                ev = sorted(m["causal_events"]) if causal else sorted((c, k) for (_, k, _), c in zip(m["pivots"], m["conf"]) if c >= 0)
                past = [(c, k) for c, k in ev if c <= b]
                if past and past[-1][1] == kind_wanted and b - past[-1][0] <= int(round(ZZ.HORIZON_CONFIRM_WINDOW * h)):
                    cands.append(h)
            if cands:
                out[b, side] = min(cands) if rule == "finest" else max(cands)
    return out


def test_the_declared_rules_name_the_horizon_and_none_means_no_horizon():
    n = 400
    flat = _bars(np.full(n, 1.1))
    assert np.isnan(ZZ.horizon_of_decision(n, ZZ.mt_levels(flat))).all()
    bars = _bars(_swings(900, 30, 0.004, seed=3))
    mt = ZZ.mt_levels(bars)
    for rule in ("finest", "coarsest"):
        for causal in (False, True):
            got = ZZ.horizon_of_decision(len(bars), mt, rule, causal=causal)
            assert np.array_equal(got, _reference_rule(len(bars), mt, rule, causal), equal_nan=True), (rule, causal)
    fin, coa = ZZ.horizon_of_decision(len(bars), mt, "finest"), ZZ.horizon_of_decision(len(bars), mt, "coarsest")
    both = np.isfinite(fin) & np.isfinite(coa)
    assert both.any() and np.all(fin[both] <= coa[both]) and np.any(fin[both] < coa[both])
    assert ZZ._rule_name(0.0) == "finest" and ZZ._rule_name(1.0) == "coarsest"


def test_mt_confirmations_pass_the_d2_ladder_and_the_causal_control_differs_from_the_ideal():
    """Coordinator condition (1): at q0.70 the per-bar confirmation error of the MT heads is in the ladder's band
    (Phi(-rho / sqrt(1 - rho^2)) = 0.23 at rho 0.588), and the causal-confirmed control differs from the ideal."""
    bars = _bars(_swings(3000, 30, 0.004, seed=5))
    sig = np.full(len(bars), 0.004)
    q = ZZ.ladder_quantities(bars, sig)
    rho = math.sin(math.pi * (0.70 - 0.5))
    d = ZZ.degrade_quantities(q, rho, 11)
    ideal, noisy = np.isfinite(q["horizon"]), np.isfinite(d["horizon"])
    err = float(np.mean(ideal != noisy))
    assert 0.15 <= err <= 0.35, err                                   # the ladder's band around 0.23
    assert np.array_equal(d["horizon"][ideal & noisy], q["horizon"][ideal & noisy], equal_nan=True)   # a true confirmation keeps its horizon
    assert np.isfinite(d["horizon"][~ideal & noisy]).all()           # a false confirmation carries a horizon
    assert np.array_equal(ZZ.degrade_quantities(q, 1.0, 11)["horizon"], q["horizon"], equal_nan=True)   # quality 1: untouched
    c = ZZ.causal_quantities(bars, sig)
    assert c["horizon"].shape == q["horizon"].shape and not np.array_equal(c["horizon"], q["horizon"], equal_nan=True)
    assert np.isfinite(c["horizon"]).mean() >= np.isfinite(q["horizon"]).mean() * 0.5


def test_legacy_ladder_quantities_are_byte_identical_with_the_mt_members_present():
    """Coordinator condition (3): DEFAULT_LEVELS unchanged; every legacy ladder quantity equals the one computed from a
    cache without the MT members."""
    bars = _bars(_swings(1500, 40, 0.004, seed=7))
    sig = np.full(len(bars), 0.004)
    lv = ZZ.ladder_levels(bars, sig)
    legacy_cache = {k: v for k, v in lv.items() if k != "mt"}
    a, b = ZZ.ladder_quantities(bars, sig, cached=legacy_cache), ZZ.ladder_quantities(bars, sig, cached=lv)
    for k in ("room", "direction", "tau_coarse", "flip_age", "counter", "fresh", "tau_fine", "fstar", "votes", "nop_hard", "nop_graded"):
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
    assert a["levels"] == b["levels"] == ZZ.DEFAULT_LEVELS and "mt3h" not in ZZ.DEFAULT_LEVELS
    ca, cb = ZZ.causal_quantities(bars, sig, cached=legacy_cache), ZZ.causal_quantities(bars, sig, cached=lv)
    for k in ("direction", "flip_age", "counter", "fresh"):
        assert np.array_equal(ca[k], cb[k], equal_nan=True), k


def _planted_horizon(h_target):
    """A path whose swings are tuned so that the finest confirming level at the probe bar has horizon h_target."""
    period = {3: 10, 6: 24, 12: 48, 24: 96}[h_target]
    return _bars(_swings(1500, period, 0.003 * (period / 10) ** 0.5, seed=h_target))


def test_each_horizon_is_reachable_on_a_tuned_path():
    seen = set()
    for h_target in (3, 6, 12, 24):
        bars = _planted_horizon(h_target)
        h = ZZ.horizon_of_decision(len(bars), ZZ.mt_levels(bars))
        vals = h[np.isfinite(h)]
        assert len(vals) > 0
        seen |= set(np.unique(vals).tolist())
    assert seen >= {3.0, 6.0, 12.0, 24.0}                            # every horizon occurs across the tuned paths


class _Env:
    def __init__(self, c, mode):
        self.c = np.asarray(c, float)
        self.cfg = type("Cfg", (), {"variant": type("V", (), {"level_mode": mode})()})()


def test_policy_emits_exact_absolute_prices_and_nop_without_a_horizon():
    import types
    n = 50
    c = np.linspace(1.1, 1.102, n)
    pol = object.__new__(ZZ.PolicyC3Z)
    pol.t, pol.attr, pol.log = {"zz_h": np.full((n, 2), np.nan), "zz_sigma": np.full(n, 0.0048)}, [{"blocked": "", "chosen": 0}], [(0, 1.0, 1.0, True, False, 0, False, 0, 0)]
    pol.t["zz_h"][10, 0] = 24.0
    pol._ctx = (10, _Env(c, "absolute_pips"))
    old = dict(ZZ.PARAMS)
    try:
        ZZ.PARAMS.update(a_sl=1.0, b_sl=0.25, buffer_pips=1.0, a_tp=1.0, b_tp=0.1, ref_p75=0.0)
        assert pol._absolute_mode()
        lv = pol.absolute_envelope(10, 0)
        ref = HL.references(24, 1, "median")
        want = HL.levels(1, float(c[10]), ref["tp_reference"], ref["mae_reference"], HL.sigma_h(0.0048, 24), {k: ZZ.PARAMS[k] for k in ("a_sl", "b_sl", "buffer_pips", "a_tp", "b_tp")}, ZZ.LEVEL_TICK)
        assert lv["h"] == 24 and lv["tp"] == want["tp"] and lv["sl"] == want["sl"] and lv["quantile"] == "median"
        assert abs(round(lv["tp"] / ZZ.LEVEL_TICK) - lv["tp"] / ZZ.LEVEL_TICK) < 1e-6 and abs(round(lv["sl"] / ZZ.LEVEL_TICK) - lv["sl"] / ZZ.LEVEL_TICK) < 1e-6   # on the tick grid
        assert pol.absolute_envelope(10, 1) is None                     # the SELL side has no horizon: NOP
        ZZ.PARAMS["ref_p75"] = 1.0
        pol._refs = None
        assert pol.absolute_envelope(10, 0)["tp"] > lv["tp"]            # the p75 option widens the TP
        pol._ctx = (10, _Env(c, "sigma"))
        assert not pol._absolute_mode()                                 # the legacy profile never enters the absolute path
    finally:
        ZZ.PARAMS.clear()
        ZZ.PARAMS.update(old)


def test_horizon_mix_record_and_the_dominance_flag():
    from tools import strategy_behaviour_matrix as BM
    attr = [{"chosen": 0, "h": 3}] * 19 + [{"chosen": 1, "h": 24}] + [{"chosen": -1, "h": 6}]
    view = {"zz_h": np.array([[3.0, np.nan]] * 95 + [[24.0, 24.0]] * 5), "zz_h_alt": np.array([[3.0, 3.0]] * 96 + [[np.nan, 24.0]] * 4)}
    m = BM.horizon_mix({"attr": attr}, view)
    assert m["n_executed"] == 20 and abs(m["executed"]["h3"] - 0.95) < 1e-12 and m["dominant"] == "h3" and m["flag_one_horizon_dominates_both_rules"]
    view["zz_h_alt"] = np.array([[24.0, 24.0]] * 100)
    assert not BM.horizon_mix({"attr": attr}, view)["flag_one_horizon_dominates_both_rules"]
    assert BM.horizon_mix({"attr": []}, {})["dominant"] is None
