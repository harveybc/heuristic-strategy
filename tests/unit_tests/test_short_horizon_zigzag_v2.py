"""ZigZag NSGA-II redo (owner + Musashi audit, 2026-10-19). Written before the code existed.

- the pinned signal_blocks path stays byte-identical (fills at b + 2 under entry_delay_bars = 1, no
  no-reopen rule); the additive executable path decides after the completed bar b, fills at the open of
  b + 1 and refuses entries in the first hours after a session reopens (the calendar's entry delay);
- the fast confirmation bar equals a brute-force replay of the knowable rule on every prefix;
- the hindsight pivots are unchanged by the refactor;
- log-scale genes, Backstep < Depth repair, bound detection, exact hypervolume, the convergence rule
  and the read ledger.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.short_horizon import execution as X
from app.short_horizon import zigzag as Z
from app.short_horizon import zigzag_opt as ZO
from tests.unit_tests.test_short_horizon import HOUR, calendar_for, synthetic_bars

OLD_BLOCKS_DIGEST = "177848238c47bb01759e015f57c9562d14838d4ad2888416bf07464c5f44c6fa"
OLD_PIVOTS_DIGEST = "c62364d1689af75cc834e92852fdf2d91c66f3014f941273f7e33fcb45d41dc5"


def _digest(bl):
    h = hashlib.sha256()
    for a in (bl.bar_lo, bl.bar_hi, bl.ref_bar, bl.segment):
        h.update(np.asarray(a, np.int64).tobytes())
    h.update(np.nan_to_num(np.asarray(bl.ref, float), nan=-1).tobytes())
    h.update("|".join(map(str, bl.status)).encode())
    h.update(np.array([s.value for s in bl.start] + [e.value for e in bl.end], np.int64).tobytes())
    return h.hexdigest()


def _case():
    bars = synthetic_bars(seed=12, segments=3, hours=240, start="2020-01-05 21:00")
    cal = calendar_for(bars, delay=12, flat=12, cutoff=0)
    p = Z.Params(depth=6, deviation_pips=3, backstep=2)
    piv = Z.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), p)
    return bars, cal, p, piv


def test_pinned_paths_are_byte_identical():
    bars, cal, p, piv = _case()
    h = hashlib.sha256(json.dumps([(int(i), int(k), round(float(v), 8)) for i, k, v in piv]).encode()).hexdigest()
    assert h == OLD_PIVOTS_DIGEST
    bl = Z.signal_blocks(bars, cal, Z.signals_from_pivots(piv))
    assert _digest(bl) == OLD_BLOCKS_DIGEST


def test_executable_path_fills_at_the_next_open_and_respects_the_reopen_delay():
    bars, cal, p, piv = _case()
    sig = Z.signals_from_pivots(piv)
    old = Z.signal_blocks(bars, cal, sig)
    new = Z.signal_blocks(bars, cal, sig, executable=True)
    assert len(new) == len(old)
    v = bars.index
    for k in range(len(new)):
        seg = cal.segments[new.segment[k]]
        if new.status[k] == "OK":
            assert new.bar_lo[k] == sig[k][0] + 1
            assert v[new.bar_lo[k]] >= seg.operational_start
        if old.status[k] == "OK" and v[old.bar_lo[k]] < cal.segments[old.segment[k]].operational_start:
            assert new.status[k] == "BEFORE_OPERATIONAL_START"
    sigma = Z.causal_range_sigma(bars)
    sig_v = Z.causal_sigma(bars)
    o_old = Z.reversal_orders(old, sig, piv, sig_v, mode="causal", k_tp=1.0, sigma_rule=sigma)
    o_new = Z.reversal_orders(new, sig, piv, sig_v, mode="causal", k_tp=1.0, sigma_rule=sigma)
    split = (bars.index[0], bars.index[-1] + HOUR)
    sb = lambda b: sigma[np.minimum(b.bar_lo, len(bars) - 1)]  # noqa: E731
    r_old = X.run_backtest(bars, old, o_old, X.RealisticProfile(), cal, 10_000.0, split, profile=X.RealisticProfile(),
                           order_rules=X.OrderRules(), sigma_block=sb(old), sizing=X.Sizing())
    prof0 = ZO.executable_profile()
    assert prof0.entry_delay_bars == 0
    r_new = X.run_backtest(bars, new, o_new, prof0, cal, 10_000.0, split, profile=prof0,
                           order_rules=X.OrderRules(), sigma_block=sb(new), sizing=X.Sizing())
    assert r_old.trades and r_new.trades
    assert all(t["entry_bar_abs"] == old.bar_lo[t["block"]] + 1 for t in r_old.trades)       # b + 2
    assert all(t["entry_bar_abs"] == new.bar_lo[t["block"]] for t in r_new.trades)           # b + 1
    assert all(t["entry_bar_abs"] == sig[t["block"]][0] + 1 for t in r_new.trades)


def _brute_confirmation(high, low, p, i, kind):
    """First t at which the knowable rule holds on the prefix [:t+1]: candidate maps computed on the prefix,
    entries younger than Backstep masked (not yet final), selection run, pivot i present and followed by an
    opposite pivot."""
    n = len(high)
    for t in range(i + 1, n):
        lm, hm = Z.candidate_maps(high[:t + 1], low[:t + 1], p)
        cut = t - p.backstep
        lm, hm = lm.copy(), hm.copy()
        lm[max(cut + 1, 0):] = 0.0
        hm[max(cut + 1, 0):] = 0.0
        sel = Z.select_pivots(lm, hm)
        kinds = {b: k for b, k, _ in sel}
        if kinds.get(i) == kind and any(b > i and k == -kind for b, k, _ in sel):
            return t
    return -1


@pytest.mark.parametrize("seed,p", [(3, Z.Params(4, 2.0, 2)), (5, Z.Params(6, 3.0, 3)), (8, Z.Params(3, 1.0, 1))])
def test_fast_confirmation_equals_the_brute_force_prefix_rule(seed, p):
    bars = synthetic_bars(seed=seed, segments=1, hours=260, start="2020-01-05 21:00")
    h, l = bars.HIGH.to_numpy(), bars.LOW.to_numpy()
    piv = Z.hindsight_pivots(h, l, p)
    conf = Z.confirmation_bars_fast(h, l, piv, p)
    assert len(conf) == len(piv) and len(piv) >= 6
    for (i, kind, _), c in zip(piv, conf):
        assert c == _brute_confirmation(h, l, p, i, kind)
    # never later than the v1 definition (next final pivot + Backstep)
    for j, ((i, kind, _), c) in enumerate(zip(piv, conf)):
        if j + 1 < len(piv) and c >= 0:
            assert c <= piv[j + 1][0] + p.backstep


def test_genes_are_log_scaled_repaired_and_bounds_are_detected():
    sp = ZO.SPACE_A
    lo, hi = sp.lower(), sp.upper()
    assert lo[0] == pytest.approx(np.log(2)) and hi[0] == pytest.approx(np.log(200))
    assert lo[1] == pytest.approx(np.log(1)) and hi[1] == pytest.approx(np.log(300))
    g = [np.log(5), np.log(20), 12.0]
    p, g2 = sp.decode(g)
    assert p["depth"] == 5 and p["backstep"] == 4 and g2[2] == pytest.approx(4.0)        # repaired, written back
    assert p["deviation_points"] == pytest.approx(p["deviation_pips"] * 10)
    p, _ = sp.decode([np.log(200), np.log(300), 30.0])
    assert sp.on_bound(p) == ["depth=upper", "deviation_pips=upper", "backstep=upper"]
    p, _ = sp.decode([np.log(50), np.log(30), 5.0])
    assert sp.on_bound(p) == []
    pb, _ = ZO.SPACE_B.decode([np.log(50), np.log(30), 5.0, 1.5])
    assert pb["k_tp"] == pytest.approx(1.5)


def test_hypervolume_is_exact_in_two_and_three_dimensions():
    assert ZO.hypervolume([[1, 2], [2, 1]], [3, 3]) == pytest.approx(3.0)
    assert ZO.hypervolume([[1, 2], [2, 1], [2.5, 2.5]], [3, 3]) == pytest.approx(3.0)
    assert ZO.hypervolume([[0, 0, 0]], [1, 2, 3]) == pytest.approx(6.0)
    assert ZO.hypervolume([[0, 0, 1], [1, 1, 0]], [2, 2, 2]) == pytest.approx(4 + 1 * 1 * 2 - 1 * 1 * 1)
    assert ZO.hypervolume([[5, 5]], [3, 3]) == 0.0


def test_convergence_rule_needs_the_minimum_generations_and_a_flat_last_window():
    hv = list(np.linspace(1, 2, 60)) + [2.0] * 30
    assert ZO.converged(hv, min_generations=80, window=20, tol=0.01) is True
    assert ZO.converged(hv[:70], min_generations=80, window=20, tol=0.01) is False
    assert ZO.converged(list(np.linspace(1, 3, 100)), min_generations=80, window=20, tol=0.01) is False


def test_read_ledger_appends_one_line_per_read(tmp_path):
    led = tmp_path / "READ_LEDGER.jsonl"
    ZO.ledger(led, "VALIDATION_2023", "JOINT_DEVELOPMENT", "front validation A cap 4", n_points=7)
    ZO.ledger(led, "OBSERVED_2024", "PREVIOUSLY_OBSERVED_DEVELOPMENT", "chosen point", n_points=1)
    rows = [json.loads(x) for x in led.read_text().splitlines()]
    assert [r["period"] for r in rows] == ["VALIDATION_2023", "OBSERVED_2024"]
    assert rows[1]["status"] == "PREVIOUSLY_OBSERVED_DEVELOPMENT" and "confirmation" not in json.dumps(rows).lower()
