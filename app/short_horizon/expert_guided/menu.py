"""The full_f4 DECISION MENU (coordinator ruling 2026-10-29, ~05:50Z; supersedes fbe52e7's "truth on sigma, orders absolute").

Under the absolute-level profile the policy values exactly the orders it places. At a decision bar b the menu is the
set of absolute candidate orders {(h, side): h in MENU_HORIZONS, side in (BUY, SELL)}, each built by
horizon_levels.levels(side, quote = close[b], the TRAIN references of (h, side), sigma_h = sigma(24 h) x sqrt(h / 24),
the PARAMS level genes, tick LEVEL_TICK), plus NOP. A horizon whose MT level does not CONFIRM the side at b
(strategy_zz.HORIZON_RULE, per level) is not in the menu at that bar. The engine truth under full_f4 is the exact replay
of these orders on the PROFILE engine (sizing, costs, calendar); the ideal heads, the quality ladder, the truth oracle,
O*, the hold oracle, the forced-entry episodes and the exploration visits all use the same menu; EV ranks (h, side);
``horizon_rule`` is the arbiter's tie-break / prior among the confirming horizons, not the selector. The sigma-grid
truth stays only under full_f3 / LEGACY (byte-identical).

The menu's slots are the CELLS of a full_f4 world: MENU_CELLS = (("H", 3), ("H", 6), ("H", 12), ("H", 24)). They are
deliberately NOT numbers: any code path that would read a slot as a sigma multiple (cells[ci][0] as a TP multiple) fails
loudly instead of trading a meaningless level.

Declared choices (ours, not the ruling's): the reference slot of the 3-class opener head / E[net | TIME] is h = 6
(MENU_REF_H; unused by the canonical ev_from_net = 1 rule); the exploration visits draw a slot uniformly among the four
horizons (the order geometry the strategy can place, confirmation-free); a contracted TP is never inflated (the
protective k_tp_protect x MFE rule and its tp_mode switch are INERT under full_f4).
"""
from __future__ import annotations

import math

import numpy as np

from .. import zigzag as Z
from . import horizon_levels as HL

MENU_HORIZONS = (3, 6, 12, 24)
MENU_CELLS = tuple(("H", h) for h in MENU_HORIZONS)
MENU_REF_H = 6
LEVEL_GENES = ("a_sl", "b_sl", "buffer_pips", "a_tp", "b_tp")
GENE_RESOLUTION = {"a_sl": 0.01, "b_sl": 0.01, "buffer_pips": 0.1, "a_tp": 0.01, "b_tp": 0.01}   # the declared resolution of the level genes (truth cache key)
INERT_UNDER_MENU = {"k_tp_protect": "a contracted TP is never inflated: the protective k x MFE envelope is off under full_f4",
                    "tp_mode": "selects the protective TP envelope, off under full_f4"}


class MenuGeneMismatch(ValueError):
    pass


def is_menu(cells) -> bool:
    """True when the world's cells are the full_f4 menu slots (("H", h), ...)."""
    try:
        return len(cells) > 0 and str(cells[0][0]) == "H"
    except (TypeError, IndexError, KeyError):
        return False


def slot_h(ci: int) -> int:
    return int(MENU_HORIZONS[int(ci)])


def ref_slot() -> int:
    return MENU_HORIZONS.index(MENU_REF_H)


def level_genes(params: dict) -> dict:
    return {k: float(params[k]) for k in LEVEL_GENES}


def quantile_of(params: dict) -> str:
    return "p75" if float(params.get("ref_p75", 0.0)) >= 0.5 else "median"


def genes_stamp(params: dict, swap_refs: bool = False) -> dict:
    """What a menu truth is a function of beyond the world: the level genes, the reference quantile, the reference swap."""
    return {"genes": level_genes(params), "quantile": quantile_of(params), "swap_refs": bool(swap_refs)}


def check_stamp(stamp: dict, params: dict):
    """The guard: a policy whose level genes (or reference quantile) differ from those the menu truth was built with is
    refused (never silent); the reference swap is a property of the world, not of the policy."""
    want = {"genes": level_genes(params), "quantile": quantile_of(params)}
    got = {"genes": {k: float(v) for k, v in (stamp.get("genes") or {}).items()}, "quantile": stamp.get("quantile")}
    if got != want:
        raise MenuGeneMismatch(f"MENU_GENES_MISMATCH:truth={got}:policy={want}")


def quantized_key(params: dict) -> tuple:
    """The cache key of a menu truth: the level genes on their declared resolution grid, the quantile option."""
    return tuple(int(round(float(params[k]) / GENE_RESOLUTION[k])) for k in LEVEL_GENES) + (quantile_of(params),)


# ----------------------------------------------------------------------------- the confirmations (per level)
def confirmations(n: int, mt: dict, causal: bool = False) -> np.ndarray:
    """(n, 2, 4) bool: level h confirms side s at bar b (strategy_zz.HORIZON_RULE, per level: its latest confirmation
    <= b is a VALLEY for a buy / PEAK for a sell, within HORIZON_CONFIRM_WINDOW x h bars). ``causal``: every candidate
    at i + backstep (the causal-confirmed control). Consistent with strategy_zz.horizon_of_decision: the finest
    (coarsest) confirming level is its 'finest' ('coarsest') horizon (test)."""
    from . import strategy_zz as ZZ
    out = np.zeros((n, 2, len(MENU_HORIZONS)), bool)
    by_h = {v[2]: k for k, v in ZZ.MT_HORIZONS.items()}
    for j, h in enumerate(MENU_HORIZONS):
        m = mt[by_h[h]]
        win = int(round(ZZ.HORIZON_CONFIRM_WINDOW * float(h)))
        events = sorted(m["causal_events"]) if causal else sorted((c, kind) for (_, kind, _), c in zip(m["pivots"], m["conf"]) if c >= 0)
        if not events:
            continue
        conf_bars = np.array([c for c, _ in events], np.int64)
        kinds = np.array([k for _, k in events], np.int64)
        idx = np.searchsorted(conf_bars, np.arange(n), side="right") - 1
        has = idx >= 0
        latest_conf = np.where(has, conf_bars[np.clip(idx, 0, len(conf_bars) - 1)], -1)
        latest_kind = np.where(has, kinds[np.clip(idx, 0, len(kinds) - 1)], 0)
        recent = has & (np.arange(n) - latest_conf <= win)
        out[:, 0, j] = recent & (latest_kind == Z.VALLEY)
        out[:, 1, j] = recent & (latest_kind == Z.PEAK)
    return out


def degrade_confirmations(conf: np.ndarray, rho: float, rng, ar1) -> np.ndarray:
    """The menu confirmations at quality rho: each (side, level) flag misclassified through the latent
    z = rho (+-1) + sqrt(1 - rho^2) u, u an AR(1) stream (phi 0.9, the same ladder as the rule-named horizon)."""
    if rho >= 1.0:
        return conf.copy()
    n = conf.shape[0]
    k = math.sqrt(max(1.0 - rho * rho, 0.0))
    out = np.zeros_like(conf)
    for s in (0, 1):
        for j in range(conf.shape[2]):
            y = conf[:, s, j].astype(float)
            out[:, s, j] = (rho * (2 * y - 1) + k * ar1(rng, n, 0.9)) > 0
    return out


# ----------------------------------------------------------------------------- the orders
def orders(close, sigma, row_bar, params: dict, profiles: dict | None = None, swap_refs: bool = False, tick: float | None = None) -> dict:
    """Per dataset row r (decision bar b = row_bar[r]) and side and slot: the absolute order (tp, sl PRICES, NaN when the
    inputs are outside the contract or a distance is <= 0) and its distances in sigma of the decision (tp_s, sl_s).
    ``swap_refs``: the BUY order reads the SELL references and vice versa (the W14_menu mirror control)."""
    from . import strategy_zz as ZZ
    tick = ZZ.LEVEL_TICK if tick is None else float(tick)
    prof = profiles if profiles is not None else HL.load_profiles()
    q = quantile_of(params)
    genes = level_genes(params)
    rb = np.asarray(row_bar, np.int64)
    c, sg = np.asarray(close, float), np.asarray(sigma, float)
    n = len(rb)
    out = {k: np.full((n, 2, len(MENU_HORIZONS)), np.nan) for k in ("tp", "sl", "tp_s", "sl_s")}
    refs = {}
    for j, h in enumerate(MENU_HORIZONS):
        for si, d in ((0, 1), (1, -1)):
            refs[(j, si)] = HL.references(h, -d if swap_refs else d, q, prof)
    for r in np.flatnonzero((rb >= 0) & (rb < len(c))):
        b = int(rb[r])
        quote, s24 = float(c[b]), float(sg[b])
        if not (np.isfinite(quote) and np.isfinite(s24) and s24 > 0):
            continue
        for j, h in enumerate(MENU_HORIZONS):
            for si, d in ((0, 1), (1, -1)):
                ref = refs[(j, si)]
                try:
                    lv = HL.levels(d, quote, ref["tp_reference"], ref["mae_reference"], HL.sigma_h(s24, h), genes, tick)
                except HL.LevelInputError:
                    continue
                if lv["raw_tp_distance"] <= 0 or lv["raw_sl_distance"] <= 0:
                    continue
                out["tp"][r, si, j], out["sl"][r, si, j] = lv["tp"], lv["sl"]
                out["tp_s"][r, si, j] = abs(lv["tp"] - quote) / s24
                out["sl_s"][r, si, j] = abs(quote - lv["sl"]) / s24
    out["stamp"] = genes_stamp(params, swap_refs)
    return out


def rows_of_bars(conf_bars: np.ndarray, row_bar) -> np.ndarray:
    """(n_rows, 2, 4) bool: the per-bar confirmations read at each row's decision bar (False where the row has no bar)."""
    rb = np.asarray(row_bar, np.int64)
    ok = (rb >= 0) & (rb < conf_bars.shape[0])
    out = np.zeros((len(rb), 2, conf_bars.shape[2]), bool)
    out[ok] = conf_bars[rb[ok]]
    return out


# ----------------------------------------------------------------------------- the allowed slots and the arbiter
def allowed(net, mask, tp_s, sl_s, base: float = 0.0, rr_margin: float = 0.10, mae=None, rr_floor: float | None = None) -> np.ndarray:
    """The declared slot rule of one (row, side): in the menu, executable (finite net and levels), a TP shorter than its
    SL only when the net clears ``base`` + rr_margin (the C3 RR rule on the order's own distances), and, when a MAE is
    given, net >= rr_floor x MAE. No sigma band: absolute levels are never clipped."""
    net, tp_s, sl_s = (np.asarray(x, float) for x in (net, tp_s, sl_s))
    ok = np.asarray(mask, bool) & np.isfinite(net) & np.isfinite(tp_s) & np.isfinite(sl_s)
    ok &= (tp_s >= sl_s) | (net > float(base) + float(rr_margin))
    if mae is not None and rr_floor is not None:
        m = np.asarray(mae, float)
        ok &= np.where(np.isfinite(m), net >= float(rr_floor) * m, True)
    return ok


TIE_EPS = 1e-9


def tie_order(rule) -> tuple:
    """The arbiter's prior among confirming horizons: finest first (rule 0 / 'finest') or coarsest first (1 / 'coarsest')."""
    coarsest = (rule == "coarsest") if isinstance(rule, str) else float(rule) >= 0.5
    idx = tuple(range(len(MENU_HORIZONS)))
    return tuple(reversed(idx)) if coarsest else idx


def argmax_slot(values, ok, rule=0.0) -> int:
    """The allowed slot with the highest value; a tie (within TIE_EPS) goes to the slot the horizon rule ranks first;
    -1 when none is allowed."""
    v = np.where(np.asarray(ok, bool), np.asarray(values, float), -np.inf)
    m = float(np.max(v)) if len(v) else -np.inf
    if not np.isfinite(m):
        return -1
    for j in tie_order(rule):
        if v[j] >= m - TIE_EPS:
            return int(j)
    return -1


def slot_of_trade(t: dict, tp_row, sl_row, tick: float = 1e-5) -> int:
    """The menu slot whose levels the executed order carries (|delta| <= half a tick on both), -1 if none."""
    for j in range(len(MENU_HORIZONS)):
        a, b = float(tp_row[j]), float(sl_row[j])
        if np.isfinite(a) and np.isfinite(b) and abs(float(t["take_profit"]) - a) <= 0.5 * tick and abs(float(t["stop_loss"]) - b) <= 0.5 * tick:
            return j
    return -1


# ----------------------------------------------------------------------------- the engine truth of the menu (exact replay)
EXIT_CODES = {"TP": 0, "SL": 1, "FORCED_FLAT": 2, "EARLY": 3}


def entry_table(env, rows, row_bar, menu_orders: dict) -> dict:
    """Exact replay of every menu order on a clean account of the PROFILE engine (its sizing, costs, calendar): per
    (row, side, slot) the net in sigma per unit and in USD, the first barrier (hit 0 TP / 1 SL / 2 TIME), MAE / MFE in
    sigma, duration, exit bar, censoring, exit reason; NaN / -1 where not eligible or refused (refusals counted)."""
    from . import outcomes as OC
    n, ns = len(row_bar), len(MENU_HORIZONS)
    t = {k: np.full((n, 2, ns), np.nan) for k in ("net_sigma", "net_usd", "mae", "mfe", "duration")}
    t["hit"] = np.full((n, 2, ns), -1, np.int8)
    t["exit_code"] = np.full((n, 2, ns), -1, np.int8)
    t["exit_bar"] = np.full((n, 2, ns), -1, np.int64)
    t["censored"] = np.zeros((n, 2, ns), bool)
    t["eligible"] = np.zeros(n, bool)
    refused = {}
    for r in np.asarray(rows, np.int64):
        b = int(row_bar[r])
        t["eligible"][r] = True
        for si, d in enumerate(OC.SIDES):
            for j in range(ns):
                tp, sl = float(menu_orders["tp"][r, si, j]), float(menu_orders["sl"][r, si, j])
                if not (np.isfinite(tp) and np.isfinite(sl)):
                    refused["LEVELS_INVALID"] = refused.get("LEVELS_INVALID", 0) + 1
                    continue
                o = OC.simulate_entry(env, b, d, tp, sl)
                if o["refused"]:
                    refused[o["refused"]] = refused.get(o["refused"], 0) + 1
                    continue
                tr = env.trades[-1]
                for k in ("net_sigma", "mae", "mfe", "duration", "hit", "exit_bar", "censored"):
                    t[k][r, si, j] = o[k]
                t["net_usd"][r, si, j] = float(tr["net_usd"])
                t["exit_code"][r, si, j] = EXIT_CODES.get(tr["close_reason"], 3)
    t["refused"] = refused
    return t
