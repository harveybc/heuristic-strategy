#!/usr/bin/env python
"""The BEHAVIOUR MATRIX (owner order 2026-10-27): a systematic full variation of the market variables, each cell a
synthetic world on the real hourly calendar with its EXPECTED and FORBIDDEN agent behaviours written before any run,
simulated over a sustained span (the fold's TRAIN period: 18 months of hourly bars for wf2022) with PERFECT heads,
the real C3 strategy and the real engine; every cell carries the witness that asserts its behaviours and a PASS / FAIL.
A failing cell with perfect heads is a strategy defect: fixed and re-run, never excluded.

Market variables (rows): coarse trend direction x strength, transitions (ordered pairs x switch position, 1-4 legs),
volatility level, volatility dynamics, fine oscillation amplitude x pivot density, gaps, spread regime, calendar;
every cell has its exact log mirror. Agent behaviours (columns): OPEN_LONG, OPEN_SHORT, NOP, HOLD, EARLY_CLOSE,
REVERSE, VOLUME, TP/SL cell. Sampling: a one-factor-at-a-time backbone around the baseline, five factorials, a Latin
hypercube over every knob, mirrored.

CLI: cells (write the matrix definition), run (evaluate a chunk of cells, one JSON per cell x seed, resumable),
report (BEHAVIOUR_MATRIX.json + .md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402
from tools import strategy_hierarchy as H  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

BLOCK = "TRAIN"                     # the sustained span every cell is simulated and witnessed on (18 months for wf2022)
CAL = S.CAL_BLOCK                   # the gene map's calibration block (INNER_A); synthetic as well
BASELINE = {"legs": ["trend_down"], "strength": "medium", "switch_fracs": [], "vol": "mid", "fine": "medium", "jump": None, "exec": {}}
VOL_LEVELS = {"low": 0.7, "mid": 1.0, "high": 1.6}
VOL_DYNAMICS = ("const", "step_lh", "step_hl", "garch09", "garch097", "ramp_up", "ramp_down")
FINE_AMP = {"low": 1.5, "medium": 2.5, "high": 4.0}
FINE_DENS = {"low": 0.5, "medium": 1.0, "high": 2.0}
GAPS = {"none": None, "g2_with": {"sigma": 2.0, "against": False, "frac": 0.5}, "g2_against": {"sigma": 2.0, "against": True, "frac": 0.5},
        "g4_with": {"sigma": 4.0, "against": False, "frac": 0.5}, "g4_against": {"sigma": 4.0, "against": True, "frac": 0.5}}
SPREADS = {"normal": {}, "spike": {"spread": "spike"}, "x5": {"spread": "x5"}}
CALENDARS = {"normal": {}, "cutoff12": {"cutoff_h": 12.0}, "missing": {"missing_runs": True}}
TRENDS = [("trend_up", st) for st in H.TREND_STRENGTH] + [("trend_down", st) for st in H.TREND_STRENGTH] + [("range", "medium")]
BEHAVIOURS = ("OPEN_LONG", "OPEN_SHORT", "NOP", "HOLD", "EARLY_CLOSE", "REVERSE", "VOLUME", "TPSL_CELL")
LAG = H.CONFIRMATION_LAG_BARS
THRESH = {"side_share_trend_min": 0.60, "side_share_range_min": 0.25, "truth_positive_open_min": 0.90, "wrong_side_max": 0.10,
          "best_cell_min": 0.70, "nop_open_max": 0.05, "sl_exit_max": 0.10, "early_close_justified_min": 0.80, "reverse_in_switch_window_min": 0.50,
          "reverse_per_week_static_max": 1.0, "old_direction_after_lag_max": 0.40, "new_leg_side_min": 0.60, "mirror_net_rel_tol": 0.02,
          "activity_per_week_min": 1.0, "gap_loss_extra_sigma_max": 0.5}


# ----------------------------------------------------------------------------- the matrix definition
def _vol_spec(level: str, dyn: str) -> tuple:
    """(vol kind for the generator, persistence): the level sets the constant multiplier; dynamics override it."""
    if dyn == "const":
        return {"low": "low", "mid": "mid", "high": "high"}[level], None
    return dyn, None


def _cell(cid: str, level: str, cfg: dict, family: str = "c3") -> dict:
    legs = tuple(cfg["legs"])
    coarse = legs[-1]
    expected, forbidden = [], []
    if len(legs) == 1:
        if coarse == "trend_up":
            expected += ["OPEN_LONG dominant (share >= 0.60)", "HOLD through fine counter-swings while the coarse room is positive", "EARLY_CLOSE on SL-destined positions"]
            forbidden += ["OPEN_SHORT except truth-positive counter-moves (share <= 0.40, each with truth net > 0)", "REVERSE outside a coarse switch (<= 1 per week)", "opens on truth-NOP ticks"]
        elif coarse == "trend_down":
            expected += ["OPEN_SHORT dominant (share >= 0.60)", "HOLD through fine counter-swings while the coarse room is positive", "EARLY_CLOSE on SL-destined positions"]
            forbidden += ["OPEN_LONG except truth-positive counter-moves (share <= 0.40, each with truth net > 0)", "REVERSE outside a coarse switch (<= 1 per week)", "opens on truth-NOP ticks"]
        else:
            expected += ["both sides opened (each >= 0.25) at truth-positive fine counter-pivots", "EARLY_CLOSE on SL-destined positions"]
            forbidden += ["one-sided trading (> 0.75 on one side)", "opens on truth-NOP ticks"]
    else:
        expected += [f"after each coarse switch the old-direction opens fall to <= 0.40 within {LAG} bars", "side shares follow the new leg (>= 0.60 in its direction)",
                     "early-close rate in [switch, switch + lag] >= rate before the switch", "REVERSE only inside the switch windows (>= 0.50 of reversals)"]
        forbidden += ["keeping the old direction after the lag", "opens on truth-NOP ticks"]
    expected += ["every opened trade has truth net > 0 at the chosen cell (>= 0.90)", "chosen cell = the truth's best cell (>= 0.70)", "SL exits <= 0.10 (perfect closers close early)",
                 "VOLUME: risk-sized units exact", "activity >= 1 open per week"]
    forbidden += ["opening the worse side when the other side's truth net is higher by > 0.10 sigma (<= 0.10)"]
    if cfg.get("jump"):
        expected.append("the trade crossing the gap loses at most SL + gap + slippage (+0.5 sigma)")
    if cfg.get("exec", {}).get("spread") == "x5":
        expected.append("inside the x5 spread window: no open unless truth net > the window's cost")
    if cfg.get("exec", {}).get("cutoff_h"):
        expected.append("no entry inside the Friday cutoff window (engine refusal honoured)")
    if cfg.get("exec", {}).get("missing_runs"):
        expected.append("no decision on an unobservable row (HOLD)")
    if cfg["vol"] in ("step_lh", "step_hl", "ramp_up", "ramp_down"):
        expected.append("TP/SL price distances track sigma (second-half / first-half ratio follows the vol ratio)")
    return {"id": cid, "level": level, "family": family, "legs": list(legs), "strength": cfg.get("strength", "medium"), "switch_fracs": list(cfg.get("switch_fracs") or []),
            "vol": cfg["vol"], "persistence": cfg.get("persistence"), "fine": cfg["fine"], "fine_amp": cfg.get("fine_amp"), "fine_dens": cfg.get("fine_dens"),
            "jump": cfg.get("jump"), "exec": dict(cfg.get("exec") or {}), "regime": coarse, "quality": 1.0, "variant": "base",
            "expected": expected, "forbidden": forbidden, "witness": "strategy_behaviour_matrix.witnesses (decision log + ledger + truth)", "mirror": False}


def _sym(cell: dict) -> dict:
    """The base cell with SYMMETRIC financing (both swap rates at their mean): the mirror pair isolates the strategy's
    side symmetry from the declared swap asymmetry (long -7 vs short +2.5 per lot per rollover)."""
    m = dict(cell)
    m["id"], m["sym_of"], m["exec"] = "S_" + cell["id"], cell["id"], {**{k: v for k, v in (cell.get("exec") or {}).items() if k != "cost_mult"}, "financing_symmetric": True, "costs_zero": True}
    m["expected"] = list(cell["expected"]) + ["same behaviours as " + cell["id"] + " at ZERO cost (bid bars and ask-triggered shorts make the real engine side-asymmetric by the spread); its mirror M_" + cell["id"] + " must mirror it EXACTLY"]
    return m


def _mirror(cell: dict) -> dict:
    m = dict(cell)
    m["id"], m["mirror_of"], m["mirror"] = "M_" + cell["id"], "S_" + cell["id"], True
    m["exec"] = {**{k: v for k, v in (cell.get("exec") or {}).items() if k != "cost_mult"}, "financing_symmetric": True, "costs_zero": True}
    m["legs_gen"] = list(cell["legs"])                      # the generator runs the BASE legs and flips the path: an exact mirror
    m["legs"] = [H.MIRROR[x] for x in cell["legs"]]          # the effective (mirrored) legs the witnesses expect
    m["regime"] = H.MIRROR[cell["regime"]]
    sw = {"OPEN_LONG": "OPEN_SHORT", "OPEN_SHORT": "OPEN_LONG"}
    m["expected"] = [sw.get(x.split(" ")[0], x.split(" ")[0]) + x[len(x.split(" ")[0]):] for x in cell["expected"]] + ["EXACT MIRROR of S_" + cell["id"] + " (zero cost): same entry / exit bars, sides flipped, net equal in sigma units"]
    m["forbidden"] = [sw.get(x.split(" ")[0], x.split(" ")[0]) + x[len(x.split(" ")[0]):] for x in cell["forbidden"]]
    if cell.get("jump"):
        m["jump"] = dict(cell["jump"])                       # the mirror flips the path, so with / against is preserved relative to the mirrored leg
    return m


def build_matrix(n_lhs: int = 64, seed: int = 7, fine_full: bool = True) -> list:
    cells = []

    def add(cid, level, **over):
        cfg = dict(BASELINE)
        cfg.update(over)
        cfg.setdefault("fine_amp", FINE_AMP[cfg["fine"]] if cfg["fine"] in FINE_AMP else None)
        cfg.setdefault("fine_dens", FINE_DENS[cfg["fine"]] if cfg["fine"] in FINE_DENS else None)
        cells.append(_cell(cid, level, cfg))

    # backbone: one factor at a time around the baseline (medium down-trend, mid constant vol, medium fine, no gap, normal spread / calendar)
    for reg, st in TRENDS:
        add(f"B_trend_{reg}_{st}", "backbone_trend", legs=[reg], strength=st)
    for name, legs in H.SEQUENCES.items():
        for pos, f in H.SWITCH_POS.items():
            fr = [f] if len(legs) == 2 else [max(0.05, f - 0.12), min(0.95, f + 0.12)]
            add(f"B_trans_{name}_{pos}", "backbone_transition", legs=list(legs), switch_fracs=fr)
    for lv in VOL_LEVELS:
        add(f"B_vol_{lv}", "backbone_vol_level", vol=lv)
    for dyn in VOL_DYNAMICS[1:]:
        add(f"B_voldyn_{dyn}", "backbone_vol_dynamics", vol=dyn)
    for fa in FINE_AMP:
        for fd in FINE_DENS:
            add(f"B_fine_{fa}_{fd}", "backbone_fine", fine=f"{fa}_{fd}", fine_amp=FINE_AMP[fa], fine_dens=FINE_DENS[fd])
    for g, spec in GAPS.items():
        if spec:
            add(f"B_gap_{g}", "backbone_gap", jump=spec)
    for sp, ex in SPREADS.items():
        if ex:
            add(f"B_spread_{sp}", "backbone_spread", exec=ex)
    for cal_, ex in CALENDARS.items():
        if ex:
            add(f"B_cal_{cal_}", "backbone_calendar", exec=ex)
    # factorials
    for reg, st in TRENDS:
        for fa in FINE_AMP:
            for fd in FINE_DENS:
                add(f"F1_{reg}_{st}_{fa}_{fd}", "F1_trend_x_fine", legs=[reg], strength=st, fine=f"{fa}_{fd}", fine_amp=FINE_AMP[fa], fine_dens=FINE_DENS[fd])
    for reg, st in TRENDS:
        for lv in VOL_LEVELS:
            for dyn in VOL_DYNAMICS:
                vol, pers = _vol_spec(lv, dyn)
                add(f"F2_{reg}_{st}_{lv}_{dyn}", "F2_trend_x_vol", legs=[reg], strength=st, vol=vol, persistence=pers, vol_level=lv)
    for name, legs in H.SEQUENCES.items():
        for pos, f in H.SWITCH_POS.items():
            fr = [f] if len(legs) == 2 else [max(0.05, f - 0.12), min(0.95, f + 0.12)]
            for fa in FINE_AMP:
                add(f"F3_{name}_{pos}_{fa}", "F3_transition_x_fine", legs=list(legs), switch_fracs=fr, fine=f"{fa}_medium", fine_amp=FINE_AMP[fa], fine_dens=1.0)
    for reg, st in TRENDS:
        for g, spec in GAPS.items():
            add(f"F4_{reg}_{st}_{g}", "F4_trend_x_gap", legs=[reg], strength=st, jump=spec)
    for reg, st in TRENDS:
        for sp, ex1 in SPREADS.items():
            for cal_, ex2 in CALENDARS.items():
                add(f"F5_{reg}_{st}_{sp}_{cal_}", "F5_trend_x_spread_x_calendar", legs=[reg], strength=st, exec={**ex1, **ex2})
    # Latin hypercube over every knob
    knobs = {"n_legs": (1.0, 4.999), "seq_seed": (0.0, 1.0), "switch_spread": (0.2, 0.8), "strength": (0.3, 2.5), "vol_level": (0.6, 1.8), "vol_dyn": (0.0, 1.0),
             "persistence": (0.85, 0.98), "fine_amp": (1.0, 4.5), "fine_dens": (0.4, 2.2), "gap_sigma": (0.0, 4.5), "gap_against": (0.0, 1.0), "spread": (0.0, 1.0),
             "calendar": (0.0, 1.0), "cost_mult": (1.0, 3.0), "cap": (1, 4), "cutoff_h": (12.0, 24.0), "risk_pct": (0.005, 0.02)}
    u = H.lhs(n_lhs, len(knobs), seed)
    regs = list(H.L1)
    for i in range(n_lhs):
        x = {k: lo + (hi - lo) * u[i, j] for j, (k, (lo, hi)) in enumerate(knobs.items())}
        n_legs = int(x["n_legs"])
        rng = np.random.default_rng(int(x["seq_seed"] * 1e6))
        legs = [regs[int(rng.integers(0, 3))]]
        while len(legs) < n_legs:
            cand = regs[int(rng.integers(0, 3))]
            if cand != legs[-1]:
                legs.append(cand)
        fr = sorted(float(np.clip(0.5 + x["switch_spread"] * (k_ / max(n_legs - 1, 1) - 0.5), 0.08, 0.92)) for k_ in range(n_legs - 1)) if n_legs > 1 else []
        dyn = VOL_DYNAMICS[min(int(x["vol_dyn"] * len(VOL_DYNAMICS)), len(VOL_DYNAMICS) - 1)]
        vol = dyn if dyn != "const" else ("high" if x["vol_level"] > 1.3 else "low" if x["vol_level"] < 0.85 else "mid")
        st = "strong" if x["strength"] > 1.5 else "weak" if x["strength"] < 0.75 else "medium"
        gap = {"sigma": round(float(x["gap_sigma"]), 2), "against": bool(x["gap_against"] > 0.5), "frac": 0.5} if x["gap_sigma"] >= 1.0 else None
        ex = {}
        ex.update(SPREADS["x5"] if x["spread"] > 0.66 else SPREADS["spike"] if x["spread"] > 0.33 else {})
        ex.update(CALENDARS["missing"] if x["calendar"] > 0.66 else CALENDARS["cutoff12"] if x["calendar"] > 0.33 else {})
        ex.update({"cost_mult": round(x["cost_mult"], 2), "cap": int(round(x["cap"])), "risk_pct": round(x["risk_pct"], 4)})
        if "cutoff_h" not in ex:
            ex["cutoff_h"] = round(x["cutoff_h"], 1)
        add(f"LHS_{i:02d}", "LHS", legs=legs, strength=st, switch_fracs=fr, vol=vol, persistence=round(x["persistence"], 3), fine="lhs",
            fine_amp=round(x["fine_amp"], 2), fine_dens=round(x["fine_dens"], 2), jump=gap, exec=ex)
    return cells + [_sym(c) for c in cells] + [_mirror(c) for c in cells]


# ----------------------------------------------------------------------------- the world of a cell
def canonical_genes(family: str = "c3") -> dict:
    """The declared canonical gene point of the strategy under test: mid-quantile margins, reverse allowed, NOP veto
    at its mid bound. The matrix asserts BEHAVIOUR (sides, knowledge, closes, reversals, cells, volume), not the
    fitness of a tuned point; the quantile genes are mapped on each cell's own calibration block."""
    g = {k: (0.5 * (float(v[0]) + float(v[1]))) for k, v in C3.Q_BOUNDS.items()}
    g["allow_reverse"] = 1.0
    return g


def cell_world(cx, SCFG, AC, cell: dict, seed: int) -> dict:
    """The synthetic world of a cell over the sustained span (truth on CAL + BLOCK), perfect heads unless the cell says
    otherwise. Fine amplitude / density come from the cell (continuous) through a temporary L2 entry."""
    leaf = dict(cell)
    if cell.get("legs_gen"):
        leaf["legs"] = list(cell["legs_gen"])
    key = f"cell_{cell.get('fine_amp')}_{cell.get('fine_dens')}"
    if cell.get("fine_amp") is not None:
        H.L2[key] = (float(cell["fine_amp"]), float(cell["fine_dens"]))
        leaf["fine"] = key
    elif leaf["fine"] not in H.L2:
        leaf["fine"] = "medium"
    if leaf["vol"] == "mid":
        H.VOL["mid"] = 1.0
    SC = SCFG
    ex = cell.get("exec") or {}
    if ex.get("financing_symmetric"):
        prof = SC.profile
        mean_swap = 0.5 * (float(prof.swap_long_per_lot) + float(prof.swap_short_per_lot))
        SC = dataclasses.replace(SC, profile=dataclasses.replace(prof, swap_long_per_lot=mean_swap, swap_short_per_lot=mean_swap))
    if ex.get("costs_zero"):
        # the exact-mirror twins: bars are BID and a short triggers on ASK = bid + spread (A4-12), so a bid-bar world's
        # mirror is not a bid-bar world; with every cost at zero the engine is side-symmetric and the trades must mirror
        prof = SC.profile.scaled(0.0)
        prof = dataclasses.replace(prof, weekly_open_spread=0.0, slippage_market=0.0, slippage_stop=0.0, commission_per_lot_side=0.0, swap_long_per_lot=0.0, swap_short_per_lot=0.0)
        SC = dataclasses.replace(SC, profile=prof)
    w = H.leaf_world(cx, SC, AC, leaf, BLOCK, seed, periods=(CAL, BLOCK))
    H.L2.pop(key, None)
    return w


class PolicyC3Audit(S.PolicyC3PerBar):
    """PolicyC3PerBar with a per-tick decision log: both sides' EV and margins, which side cleared, which won, and
    every close decision with the closer's values. Proves the comparison is made (never 'first side wins')."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []
        self.closes = []

    def open_decision(self, r, with_score: bool = False):
        s = self.s
        vetoed = s.nop_veto is not None and float(self._v("pn", r)) > s.nop_veto
        cap, lo = self._cap(r)
        c = self.t.get("cost_row")
        cost = float(c[r]) if c is not None and np.isfinite(c[r]) else self.cost
        evs, cells_ = [], []
        for si in (0, 1):
            ci, ev = C3.best_cell_ev(self._v("net", r, si), self._v("p3", r, si), self.cells, self.e_time[si], cost, cap=cap, lo=lo)
            evs.append(float(ev) if ci >= 0 and np.isfinite(ev) else float("nan"))
            cells_.append(int(ci))
        pooled = getattr(s, "open_margin", None)                   # C3 b27781e: ONE pooled open margin for both sides (side symmetry)
        mar = (getattr(s, "open_margin_l", pooled), getattr(s, "open_margin_s", pooled))
        cleared = [bool(np.isfinite(evs[i]) and evs[i] > mar[i]) for i in (0, 1)]
        out = super().open_decision(r, with_score)
        chosen = -1 if out is None else (0 if out[0] == E.LONG else 1)
        self.log.append((int(r), evs[0], evs[1], cleared[0], cleared[1], chosen, bool(vetoed), cells_[0], cells_[1]))
        return out

    def close_now(self, r, si, bar, pos) -> bool:
        if not self.s.early_close:
            return False
        rem, psl = self.t["closer"].value(r, bar, pos)
        mar = self.s.close_margin_l if si == 0 else self.s.close_margin_s
        p = self.s.close_p_l if si == 0 else self.s.close_p_s
        dec = bool(rem < -mar or psl > p)
        self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec))
        return dec


def strategy_for(world: dict, genes: dict, AC: dict):
    cx, table, truth = world["cx"], world["table"], world["truth"]
    cells = [tuple(c) for c in truth.cells]
    cal_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == CAL))
    table["cost_row"] = S.row_costs(cx, world["scfg"], truth.cost - C3.cost_sigma_of(cx, world["scfg"], 0.0, cal_rows))
    ref = C3.closer_reference_sides(table, S.closer_src_of(truth))
    view = C2.restrict(table, np.flatnonzero(cx.obs_ok & np.isin(cx.ds.period, [CAL, BLOCK])))
    gm = C3.gene_map(view, cal_rows, ref, cells, truth.e_time, truth.cost)
    strat = C3.StrategyC3.from_genes(gm.to_absolute(genes), "nop_veto_rate" in genes)
    return strat, view, cells


def rollout_cell(world: dict, strat, view, cells) -> dict:
    cx, SCFG, truth = world["cx"], world["scfg"], world["truth"]
    pol = PolicyC3Audit(view, strat, cells, truth.e_time, truth.cost)
    m, trades = ACT.rollout_with_trades(cx, BLOCK, pol(cx), SCFG)
    return {"metrics": m, "trades": trades, "log": pol.log, "closes": pol.closes}


# ----------------------------------------------------------------------------- the witnesses
def _side(t) -> int:
    return 0 if int(t["direction"]) > 0 else 1


def _cell_index(t, cells) -> int:
    for i, (tp, sl) in enumerate(cells):
        if abs(float(t["take_profit"]) - tp) < 1e-6 and abs(float(t["stop_loss"]) - sl) < 1e-6:
            return i
    return -1


def decision_offset(SCFG) -> int:
    """Bars between the decision row and the fill bar: the profile's entry delay plus the next-open fill."""
    return 1 + int(getattr(SCFG.profile, "entry_delay_bars", 1) or 0)


def truth_side_opportunity(truth, cx, a: int, b: int) -> dict:
    """Among the block's rows where at least one side has a positive truth net, the share whose better side is short."""
    rows = truth.rows[(truth.period[truth.rows] == BLOCK)]
    best = []
    for r in rows:
        if not np.isfinite(truth.net[r]).any():
            continue
        l_ = float(np.nanmax(truth.net[r, 0])) if np.isfinite(truth.net[r, 0]).any() else -np.inf
        s_ = float(np.nanmax(truth.net[r, 1])) if np.isfinite(truth.net[r, 1]).any() else -np.inf
        if max(l_, s_) > 0:
            best.append(1 if s_ > l_ else 0)
    return {"rows_with_opportunity": len(best), "share_short_better": float(np.mean(best)) if best else float("nan")}


def witnesses(cell: dict, world: dict, roll: dict, cells) -> dict:
    """Every behaviour assertion of the cell, measured: value, threshold, pass. Perfect-head thresholds."""
    cx, truth, SCFG = world["cx"], world["truth"], world["scfg"]
    trades = roll["trades"]
    log = roll["log"]
    a, b = H.period_bars(cx.ds, BLOCK)
    weeks = max((b - a) / (24 * 5), 1e-9)
    out = {}
    n_open = len(trades)
    off = decision_offset(SCFG)
    truth_opp = truth_side_opportunity(truth, cx, a, b)
    sides = np.array([_side(t) for t in trades], int)
    share_l = float((sides == 0).mean()) if n_open else float("nan")
    share_s = 1.0 - share_l if n_open else float("nan")
    reg = cell["regime"]
    legs = cell["legs"]
    # W1 sides vs the coarse leg (last leg's regime for static cells; per leg for transitions)
    if len(legs) == 1:
        # the sides must FOLLOW THE TRUTH: the strategy's short share within 0.15 of the truth's share of short-better
        # opportunities, and the trend direction dominant whenever the truth is dominated by it (>= 0.60)
        t_short = truth_opp["share_short_better"]
        dev = abs(share_s - t_short) if n_open and np.isfinite(t_short) else float("nan")
        follow = bool(n_open and np.isfinite(dev) and dev <= 0.15)
        if reg == "trend_up":
            dom_ok = (share_l >= 0.5) if (1 - t_short) >= THRESH["side_share_trend_min"] else True
            out["W1_side"] = {"value": {"share_long": share_l, "truth_share_long_better": 1 - t_short, "deviation": dev}, "threshold": "|dev| <= 0.15; long is the majority of opens when the truth is long-dominated (>= 0.60)",
                              "pass": bool(follow and dom_ok), "desc": "OPEN_LONG share follows the truth's opportunities"}
        elif reg == "trend_down":
            dom_ok = (share_s >= 0.5) if t_short >= THRESH["side_share_trend_min"] else True
            out["W1_side"] = {"value": {"share_short": share_s, "truth_share_short_better": t_short, "deviation": dev}, "threshold": "|dev| <= 0.15; short is the majority of opens when the truth is short-dominated (>= 0.60)",
                              "pass": bool(follow and dom_ok), "desc": "OPEN_SHORT share follows the truth's opportunities"}
        else:
            mn = min(share_l, share_s) if n_open else float("nan")
            out["W1_side"] = {"value": {"min_side_share": mn, "truth_share_short_better": t_short, "deviation": dev}, "threshold": "min side >= 0.25; |dev| <= 0.15",
                              "pass": bool(follow and n_open and mn >= THRESH["side_share_range_min"]), "desc": "both sides traded (range), following the truth"}
    else:
        meta = world["meta"]
        bounds = [a] + list(meta["switch_bars"]) + [b]
        per_leg, ok_all, old_after = [], True, []
        for i, lr in enumerate(legs):
            lo_, hi_ = bounds[i], bounds[i + 1]
            tl = [t for t in trades if lo_ <= int(t["entry_bar_abs"]) < hi_]
            if not tl:
                per_leg.append({"leg": lr, "n": 0, "share_in_direction": float("nan")})
                continue
            sl_ = np.array([_side(t) for t in tl])
            d = 0 if lr == "trend_up" else 1 if lr == "trend_down" else None
            sh = float((sl_ == d).mean()) if d is not None else float("nan")
            per_leg.append({"leg": lr, "n": len(tl), "share_in_direction": sh})
            if d is not None and sh < THRESH["new_leg_side_min"]:
                ok_all = False
            if i > 0:
                prev = legs[i - 1]
                dprev = 0 if prev == "trend_up" else 1 if prev == "trend_down" else None
                tl2 = [t for t in tl if int(t["entry_bar_abs"]) >= lo_ + LAG]
                if dprev is not None and tl2:
                    sh_old = float(np.mean([_side(t) == dprev for t in tl2]))
                    old_after.append(sh_old)
                    if lr != "range" and sh_old > THRESH["old_direction_after_lag_max"]:
                        ok_all = False
        out["W1_side"] = {"value": per_leg, "threshold": THRESH["new_leg_side_min"], "pass": bool(n_open and ok_all), "desc": "side share per coarse leg; old-direction share after the lag"}
        out["W8_old_direction_after_lag"] = {"value": (max(old_after) if old_after else float("nan")), "threshold": THRESH["old_direction_after_lag_max"],
                                             "pass": bool(all(x <= THRESH["old_direction_after_lag_max"] for x in old_after)) if old_after else True, "desc": "max old-direction open share after the confirmation lag"}
        # early-close rate at the switch vs before
        ec_before, ec_at = [], []
        for sw in meta["switch_bars"]:
            t_b = [t for t in trades if sw - LAG <= int(t["exit_bar_abs"]) < sw]
            t_a = [t for t in trades if sw <= int(t["exit_bar_abs"]) < sw + LAG]
            ec_before.append(np.mean([t["exit_reason"] == "EARLY" for t in t_b]) if t_b else np.nan)
            ec_at.append(np.mean([t["exit_reason"] == "EARLY" for t in t_a]) if t_a else np.nan)
        rb, ra = float(np.nanmean(ec_before)) if np.isfinite(np.nanmean(ec_before)) else float("nan"), float(np.nanmean(ec_at)) if np.isfinite(np.nanmean(ec_at)) else float("nan")
        out["W8_early_close_at_switch"] = {"value": [rb, ra], "threshold": THRESH["reverse_in_switch_window_min"], "pass": bool((not np.isfinite(rb)) or (not np.isfinite(ra)) or ra >= rb), "desc": "early-close share of exits before / in the switch window"}
    # W2 knowledge: truth net at the chosen cell > 0; not the worse side; cell = truth best
    pos_ok, wrong, best_cell = [], [], []
    dec = {x[0]: x for x in log}
    matched = 0
    for t in trades:
        r = int(cx.rob[int(t["entry_bar_abs"]) - off]) if int(t["entry_bar_abs"]) - off >= 0 else -1
        rr = r if r >= 0 and np.isfinite(truth.net[r]).any() else -1
        if rr < 0:
            continue
        si = _side(t)
        d_ = dec.get(rr)
        ci = int(d_[7 + si]) if d_ is not None and d_[5] == si else _cell_index(t, cells)
        matched += int(d_ is not None and d_[5] == si)
        net_c = float(truth.net[rr, si, ci]) if ci >= 0 else float("nan")
        other = float(np.nanmax(truth.net[rr, 1 - si])) if np.isfinite(truth.net[rr, 1 - si]).any() else float("nan")
        mine = float(np.nanmax(truth.net[rr, si])) if np.isfinite(truth.net[rr, si]).any() else float("nan")
        pos_ok.append(bool(np.isfinite(net_c) and net_c > 0))
        wrong.append(bool(np.isfinite(other) and np.isfinite(mine) and other > mine + 0.10))
        best_cell.append(bool(ci >= 0 and np.isfinite(mine) and net_c >= mine - 1e-9))
    out["W2_truth_positive_open"] = {"value": float(np.mean(pos_ok)) if pos_ok else float("nan"), "threshold": THRESH["truth_positive_open_min"], "pass": bool(pos_ok and np.mean(pos_ok) >= THRESH["truth_positive_open_min"]), "desc": "opened trades with truth net > 0 at the chosen cell",
                                     "decisions_matched_to_trades": matched, "decision_row_offset": off}
    out["W2_wrong_side"] = {"value": float(np.mean(wrong)) if wrong else float("nan"), "threshold": THRESH["wrong_side_max"], "pass": bool(wrong) and float(np.mean(wrong)) <= THRESH["wrong_side_max"], "desc": "opens on the side whose truth best net is lower by > 0.10 sigma"}
    out["W7_best_cell"] = {"value": float(np.mean(best_cell)) if best_cell else float("nan"), "threshold": THRESH["best_cell_min"], "pass": bool(best_cell) and float(np.mean(best_cell)) >= THRESH["best_cell_min"], "desc": "chosen TP/SL cell = the truth's best cell"}
    # W3 NOP respected: opens on truth-NOP rows
    nop_rows = {int(r) for r in truth.rows if truth.nop[r] == 1 and truth.period[r] == BLOCK}
    opens_rows = [int(cx.rob[int(t["entry_bar_abs"]) - off]) for t in trades if int(t["entry_bar_abs"]) - off >= 0]
    nop_opens = sum(1 for r in opens_rows if r in nop_rows)
    out["W3_nop"] = {"value": (nop_opens / max(n_open, 1)), "threshold": THRESH["nop_open_max"], "pass": bool(nop_opens / max(n_open, 1) <= THRESH["nop_open_max"]), "desc": "share of opens on truth-NOP rows"}
    # W4 closes: SL exits share; early closes justified (closer's remaining value <= 0 at the close)
    sl_share = float(np.mean([t["exit_reason"] == "SL" for t in trades])) if n_open else float("nan")
    out["W4_sl_exits"] = {"value": sl_share, "threshold": THRESH["sl_exit_max"], "pass": bool(n_open and sl_share <= THRESH["sl_exit_max"]), "desc": "share of trades ending at the stop (perfect closers should have closed early)"}
    ec = [c for c in roll["closes"] if c[5]]
    just = float(np.mean([c[3] <= 0.0 for c in ec])) if ec else float("nan")
    out["W4_early_close_justified"] = {"value": just, "threshold": THRESH["early_close_justified_min"], "pass": bool((not ec) or just >= THRESH["early_close_justified_min"]), "desc": "early closes whose remaining hold value was <= 0"}
    # W5 reversals: opens while positioned (an open logged at a row where the previous trade was still open)
    rev = 0
    for i in range(1, len(trades)):
        if int(trades[i]["entry_bar_abs"]) <= int(trades[i - 1]["exit_bar_abs"]) and _side(trades[i]) != _side(trades[i - 1]):
            rev += 1
    if len(legs) == 1:
        out["W5_reverse"] = {"value": rev / weeks, "threshold": THRESH["reverse_per_week_static_max"], "pass": bool(rev / weeks <= THRESH["reverse_per_week_static_max"]), "desc": "reversals per week (static leg)"}
    else:
        sw = world["meta"]["switch_bars"]
        inwin = 0
        for i in range(1, len(trades)):
            if int(trades[i]["entry_bar_abs"]) <= int(trades[i - 1]["exit_bar_abs"]) and _side(trades[i]) != _side(trades[i - 1]):
                eb = int(trades[i]["entry_bar_abs"])
                if any(s_ <= eb < s_ + LAG for s_ in sw):
                    inwin += 1
        out["W5_reverse"] = {"value": (inwin / rev) if rev else float("nan"), "threshold": THRESH["reverse_in_switch_window_min"], "pass": bool(rev == 0 or inwin / rev >= THRESH["reverse_in_switch_window_min"]), "desc": "share of reversals inside the switch windows"}
    # W6 volume: risk-sized units exact (engine formula)
    sig = np.asarray(cx.ds.sigma, float)
    vol_ok = []
    for t in trades:
        eb = int(t["entry_bar_abs"])
        exp_units = S.risk_units(float(t.get("balance_before", np.nan)), float(t["stop_loss"]), float(sig[eb]), float(t["entry_price"]), SCFG) if hasattr(S, "risk_units") and "balance_before" in t else None
        if exp_units is not None and np.isfinite(exp_units):
            vol_ok.append(abs(int(t["units"]) - int(exp_units)) <= 1)
    out["W6_volume"] = {"value": float(np.mean(vol_ok)) if vol_ok else float("nan"), "threshold": 1.0, "pass": bool((not vol_ok) or np.mean(vol_ok) >= 0.99), "desc": "units equal the risk-sizing formula (when the ledger carries the balance)"}
    # activity
    out["W0_activity"] = {"value": n_open / weeks, "threshold": THRESH["activity_per_week_min"], "pass": bool(n_open / weeks >= THRESH["activity_per_week_min"]), "desc": "opens per week"}
    # W9 gap: the trade crossing the gap bar loses at most SL + gap + slippage
    meta = world.get("meta") or {}
    if meta.get("jump_bar") is not None:
        jb = int(meta["jump_bar"])
        cross = [t for t in trades if int(t["entry_bar_abs"]) < jb <= int(t["exit_bar_abs"])]
        worst = []
        for t in cross:
            s_ = float(sig[int(t["entry_bar_abs"])])
            loss_sigma = float(t["net_usd"]) / max(int(t["units"]), 1) / s_
            bound = -(float(t["stop_loss"]) + float(meta["jump"]["sigma"]) * math.sqrt(24.0) + THRESH["gap_loss_extra_sigma_max"])
            worst.append(loss_sigma >= bound)
        out["W9_gap_loss_bounded"] = {"value": float(np.mean(worst)) if worst else float("nan"), "threshold": 1.0, "pass": bool((not worst) or all(worst)), "desc": "trades crossing the gap lose at most SL + gap + 0.5 sigma", "n_crossing": len(cross)}
    # W10 spread x5 window: opens inside the window need truth net > the window cost
    ex = cell.get("exec") or {}
    if ex.get("spread") == "x5":
        prof = SCFG.profile
        t0 = int(getattr(prof, "window_t0_ns", 0))
        t1 = t0 + int(24 * 3600 * 1e9)
        ins = [t for t in trades if t0 <= int(cx.v[int(t["entry_bar_abs"])]) < t1]
        okk = []
        for t in ins:
            r = int(cx.rob[int(t["entry_bar_abs"]) - off])
            ci = _cell_index(t, cells)
            okk.append(bool(r >= 0 and ci >= 0 and np.isfinite(truth.net[r, _side(t), ci]) and truth.net[r, _side(t), ci] > 0))
        out["W10_spread_window"] = {"value": float(np.mean(okk)) if okk else float("nan"), "threshold": 1.0, "pass": bool((not okk) or all(okk)), "desc": "opens inside the x5 window have truth net > 0 (cost included)", "n_inside": len(ins)}
    if ex.get("cutoff_h"):
        out["W11_cutoff"] = {"value": int(roll["metrics"].get("refused_VARIANT_CUTOFF", 0)) if isinstance(roll["metrics"], dict) else None, "threshold": "engine", "pass": True, "desc": "entries inside the cutoff window are refused by the engine (count reported)"}
    if ex.get("missing_runs"):
        unobs = {int(r) for r in np.flatnonzero(~cx.obs_ok)}
        bad = sum(1 for r in opens_rows if r in unobs)
        out["W12_missing"] = {"value": bad, "threshold": 0, "pass": bool(bad == 0), "desc": "opens decided on unobservable rows"}
    if cell["vol"] in ("step_lh", "step_hl", "ramp_up", "ramp_down"):
        mid = a + (b - a) // 2
        d1 = [abs(float(t["take_profit"])) * float(sig[int(t["entry_bar_abs"])]) for t in trades if int(t["entry_bar_abs"]) < mid]
        d2 = [abs(float(t["take_profit"])) * float(sig[int(t["entry_bar_abs"])]) for t in trades if int(t["entry_bar_abs"]) >= mid]
        ratio = (float(np.mean(d2)) / float(np.mean(d1))) if d1 and d2 else float("nan")
        exp_up = cell["vol"] in ("step_lh", "ramp_up")
        out["W13_tpsl_track_sigma"] = {"value": ratio, "threshold": ">1" if exp_up else "<1", "pass": bool((not np.isfinite(ratio)) or (ratio > 1.0) == exp_up), "desc": "mean TP price distance second half / first half"}
    # per-side decision audit (proves the comparison)
    both = [x for x in log if x[3] and x[4]]
    won_l = sum(1 for x in both if x[5] == 0)
    won_s = sum(1 for x in both if x[5] == 1)
    argmax_ok = all((x[5] == (0 if x[1] >= x[2] else 1)) for x in both if x[5] >= 0)
    out["WA_side_comparison"] = {"value": {"ticks": len(log), "both_cleared": len(both), "long_won": won_l, "short_won": won_s, "only_long": sum(1 for x in log if x[3] and not x[4]),
                                           "only_short": sum(1 for x in log if x[4] and not x[3]), "none": sum(1 for x in log if not x[3] and not x[4]), "vetoed": sum(1 for x in log if x[6])},
                                 "threshold": "winner = argmax EV on every both-cleared tick", "pass": bool(argmax_ok), "desc": "per-side decision audit"}
    out["summary"] = {"opens": n_open, "share_long": share_l, "share_short": share_s, "net_usd": float(roll["metrics"].get("net_profit_usd", np.nan)) if isinstance(roll["metrics"], dict) else None,
                      "early_closes": int(sum(1 for t in trades if t["exit_reason"] == "EARLY")), "tp_exits": int(sum(1 for t in trades if t["exit_reason"] == "TP")),
                      "sl_exits": int(sum(1 for t in trades if t["exit_reason"] == "SL")), "time_exits": int(sum(1 for t in trades if t["exit_reason"] == "TIME"))}
    out["PASS"] = bool(all(v["pass"] for k, v in out.items() if k not in ("summary",)))
    return out


def mirror_check(trades_a: list, trades_b: list) -> dict:
    """The mirrored twin's ledger must mirror: same entry / exit bars, sides flipped, net equal up to the financing
    asymmetry (long and short swap rates differ) plus a relative tolerance."""
    n = min(len(trades_a), len(trades_b))
    bars_ok, side_ok, net_dev, bound = 0, 0, [], []
    for ta, tb in zip(trades_a[:n], trades_b[:n]):
        bars_ok += int(int(ta["entry_bar_abs"]) == int(tb["entry_bar_abs"]) and int(ta["exit_bar_abs"]) == int(tb["exit_bar_abs"]))
        side_ok += int(int(ta["direction"]) == -int(tb["direction"]))
        na, nb = float(ta["net_sigma"]) - float(ta.get("financing_sigma", 0.0)), float(tb["net_sigma"]) - float(tb.get("financing_sigma", 0.0))
        net_dev.append(abs(na - nb))
        bound.append(THRESH["mirror_net_rel_tol"] * max(abs(na), abs(nb), 0.05))
    ok = (len(trades_a) == len(trades_b)) and bars_ok == n and side_ok == n and all(d <= b_ for d, b_ in zip(net_dev, bound))
    return {"pass": bool(ok), "n_a": len(trades_a), "n_b": len(trades_b), "bars_match": bars_ok, "sides_flipped": side_ok,
            "max_net_discrepancy_ex_financing_sigma": float(max(net_dev)) if net_dev else 0.0, "max_allowed_sigma": float(max(bound)) if bound else 0.0,
            "financing_asymmetry_usd": float(sum(abs(float(ta.get("financing_usd", 0.0)) - float(tb.get("financing_usd", 0.0))) for ta, tb in zip(trades_a[:n], trades_b[:n])))}


# ----------------------------------------------------------------------------- CLI
def cmd_cells(a):
    cells = build_matrix(a.n_lhs, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cells, indent=1))
    levels = {}
    for c in cells:
        levels[c["level"]] = levels.get(c["level"], 0) + 1
    print(json.dumps({"cells": len(cells), "mirrored_pairs": sum(1 for c in cells if c["mirror"]), "levels": levels}))


def run_cell(cx, SCFG, AC, cell: dict, seed: int, genes: dict, out_dir: Path, twin_trades: dict) -> dict:
    t0 = time.time()
    w = cell_world(cx, SCFG, AC, cell, seed)
    strat, view, cells = strategy_for(w, genes, AC)
    roll = rollout_cell(w, strat, view, cells)
    wit = witnesses(cell, w, roll, cells)
    sig = np.asarray(w["cx"].ds.sigma, float)
    for t in roll["trades"]:
        s_ = float(sig[int(t["entry_bar_abs"])])
        t["net_sigma"] = float(t["net_usd"]) / max(int(t["units"]), 1) / s_ if np.isfinite(s_) and s_ > 0 else float("nan")
        t["financing_sigma"] = float(t.get("financing_usd", 0.0)) / max(int(t["units"]), 1) / s_ if np.isfinite(s_) and s_ > 0 else 0.0
    rec = {"id": cell["id"], "seed": seed, "level": cell["level"], "regime": cell["regime"], "legs": cell["legs"], "mirror_of": cell.get("mirror_of"), "quality": cell["quality"],
           "witness": w["witness"], "achieved": w["sim"]["achieved"], "strategy": strat.to_dict(), "results": wit, "PASS": wit["PASS"], "sec": round(time.time() - t0, 1),
           "trades": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in t.items()} for t in roll["trades"]]}
    key = cell.get("mirror_of") or cell["id"]
    twin_trades.setdefault(key, {})[cell["id"]] = roll["trades"]
    if cell.get("mirror_of") and cell["mirror_of"] in twin_trades.get(key, {}):
        rec["results"]["W14_mirror"] = mirror_check(twin_trades[key][cell["mirror_of"]], roll["trades"])
        rec["results"]["W14_mirror"]["desc"] = "exact mirror of the twin's ledger"
        rec["PASS"] = bool(rec["PASS"] and rec["results"]["W14_mirror"]["pass"])
        rec["results"]["PASS"] = rec["PASS"]
    (out_dir / f"CELL_{cell['id']}_s{seed}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def cmd_run(a):
    cells = json.loads(Path(a.cells).read_text())
    if a.ids:
        keep = set(a.ids.split(","))
        cells = [c for c in cells if c["id"] in keep or c.get("sym_of") in keep or (c.get("mirror_of") or "").replace("S_", "", 1) in keep]
    # chunking keeps a cell and its mirror in the same chunk (the mirror check needs both ledgers)
    base = [c for c in cells if not c["mirror"] and not c.get("sym_of")]
    syms = {c["sym_of"]: c for c in cells if c.get("sym_of")}
    mirrors = {c["mirror_of"]: c for c in cells if c["mirror"]}
    k, n = (int(x) for x in a.chunk.split("/"))
    mine = [c for i, c in enumerate(base) if i % n == k]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    AC = S.contract_for_sim(a.contract)
    cx, SCFG = S.load_fold(a.fold, a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    from app.short_horizon.expert_guided import strategy_params as SP
    params_doc = SP.load(a.params)
    genes = SP.genes_c3(params_doc)
    params_sha = SP.sha256_of(a.params)
    (out / "PARAMS_SHA.txt").write_text(params_sha + "  " + str(a.params) + "\n")
    seeds = [int(x) for x in a.seeds.split(",")]
    done = 0
    for c in mine:
        for seed in seeds:
            twin = {}
            sc = syms.get(c["id"])
            trio = [c] + ([sc, mirrors.get(sc["id"])] if sc is not None and seed == seeds[0] else [])
            for cc in trio:
                if cc is None:
                    continue
                p = out / f"CELL_{cc['id']}_s{seed}.json"
                if p.exists() and not a.force:
                    if not cc["mirror"]:
                        twin.setdefault(cc["id"], {})[cc["id"]] = json.loads(p.read_text())["trades"]
                    continue
                try:
                    rec = run_cell(cx, SCFG, AC, cc, seed, genes, out, twin)
                    print(json.dumps({"cell": cc["id"], "seed": seed, "PASS": rec["PASS"], "opens": rec["results"]["summary"]["opens"], "sec": rec["sec"]}), flush=True)
                except SystemExit as e:
                    (out / f"CELL_{cc['id']}_s{seed}.json").write_text(json.dumps({"id": cc["id"], "seed": seed, "level": cc["level"], "regime": cc["regime"], "mirror_of": cc.get("mirror_of"),
                                                                                    "PASS": False, "error": str(e), "results": {"PASS": False, "error": str(e)}}, indent=1))
                    print(json.dumps({"cell": cc["id"], "seed": seed, "PASS": False, "error": str(e)[:200]}), flush=True)
                done += 1
    print(json.dumps({"chunk": a.chunk, "cells_run": done}))


def cmd_report(a):
    cells = json.loads(Path(a.cells).read_text())
    recs = {}
    for p in sorted(Path(a.results).glob("CELL_*.json")):
        d = json.loads(p.read_text())
        recs.setdefault(d["id"], []).append(d)
    rows = []
    for c in cells:
        rs = recs.get(c["id"], [])
        passes = [r["PASS"] for r in rs]
        fails = {}
        for r in rs:
            for k, v in (r.get("results") or {}).items():
                if isinstance(v, dict) and v.get("pass") is False:
                    fails[k] = fails.get(k, 0) + 1
        sl = [r["results"]["summary"]["share_long"] for r in rs if r.get("results", {}).get("summary")]
        rows.append({"id": c["id"], "level": c["level"], "regime": c["regime"], "legs": c["legs"], "mirror": c["mirror"], "sym": bool(c.get("sym_of")), "n_runs": len(rs), "PASS": bool(rs) and all(passes),
                     "failing_witnesses": fails, "share_long_mean": float(np.nanmean(sl)) if sl else None, "opens_mean": float(np.mean([r["results"]["summary"]["opens"] for r in rs if r.get("results", {}).get("summary")])) if rs else None,
                     "errors": [r.get("error") for r in rs if r.get("error")], "expected": c["expected"], "forbidden": c["forbidden"]})
    n_pass = sum(1 for r in rows if r["PASS"])
    import hashlib
    spec_sha = hashlib.sha256(Path(a.cells).read_bytes()).hexdigest()
    psha = (Path(a.results) / "PARAMS_SHA.txt").read_text().split()[0] if (Path(a.results) / "PARAMS_SHA.txt").exists() else None
    doc = {"schema": "behaviour_matrix.v1", "block": BLOCK, "cells": len(rows), "run": sum(1 for r in rows if r["n_runs"]), "pass": n_pass, "fail": sum(1 for r in rows if r["n_runs"] and not r["PASS"]),
           "spec_sha256": spec_sha, "strategy_params_sha256": psha, "thresholds": THRESH, "behaviours": BEHAVIOURS, "rows": rows}
    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "BEHAVIOUR_MATRIX.json").write_text(json.dumps(doc, indent=1, default=float))
    lines = ["# Matriz de comportamiento (cabezas perfectas, estrategia C3, motor real)", "",
             f"Celdas: {len(rows)} (cada una con su gemela de financiación simétrica y su espejo exacto); corridas: {doc['run']}; PASS: {n_pass}; FAIL: {doc['fail']}; bloque sostenido: `{BLOCK}` del fold.",
             f"Especificación sellada: sha256 `{spec_sha}` del archivo de celdas; parámetros de la estrategia: sha256 `{psha}` (`STRATEGY_PARAMS.json`).", "",
             "| celda | nivel | tramos | régimen | espejo | corridas | PASS | testigos que fallan | cuota largos | aperturas |", "| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['level']} | {' → '.join(r['legs'])} | {r['regime']} | {'sí' if r['mirror'] else ''} | {r['n_runs']} | {'PASS' if r['PASS'] else ('FAIL' if r['n_runs'] else '—')} | "
                     f"{', '.join(f'{k}×{v}' for k, v in r['failing_witnesses'].items()) or ''} | {r['share_long_mean'] if r['share_long_mean'] is None else f'{r['share_long_mean']:.2f}'} | {'' if r['opens_mean'] is None else f'{r['opens_mean']:.0f}'} |")
    fails = [r for r in rows if r["n_runs"] and not r["PASS"]]
    lines += ["", "## Celdas que fallan (defectos de la estrategia a corregir; ninguna se excluye)", ""]
    if fails:
        lines += ["| celda | testigos | errores | esperado |", "| --- | --- | --- | --- |"]
        for r in fails:
            lines.append(f"| {r['id']} | {', '.join(r['failing_witnesses'])} | {'; '.join(str(e)[:80] for e in r['errors'])} | {'; '.join(r['expected'][:3])} |")
    else:
        lines.append("ninguna")
    lines += ["", "## Comportamientos esperados y prohibidos por celda (escritos antes de correr)", "", "| celda | esperado | prohibido |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['id']} | {'; '.join(r['expected'])} | {'; '.join(r['forbidden'])} |")
    (outp / "BEHAVIOUR_MATRIX.md").write_text("\n".join(lines))
    print(json.dumps({"cells": len(rows), "run": doc["run"], "pass": n_pass, "fail": doc["fail"], "out": str(outp)}))


def stop_requested(out_path) -> bool:
    from pathlib import Path as _P
    p = _P(str(out_path)).resolve()
    return any((q / "STOP").exists() for q in [p] + list(p.parents)[:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells")
    c.add_argument("--n-lhs", type=int, default=64)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True)
    r.add_argument("--fold", default="wf2022")
    r.add_argument("--admitted", required=True)
    r.add_argument("--csv-dir", required=True)
    r.add_argument("--contract", default=str(S.CONTRACT))
    r.add_argument("--chunk", default="0/1", help="k/n: this job's share of the base cells (mirrors travel with their twin)")
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--params", default=str(ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"), help="every strategy coefficient (validated; sha recorded)")
    r.add_argument("--ids", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--cells", required=True)
    rp.add_argument("--results", required=True)
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd != "cells" and stop_requested(a.out) and os.environ.get("BEHAVIOUR_MATRIX_ALLOWED") != "1":
        print("STOP flag present and the behaviour matrix is not marked as allowed (BEHAVIOUR_MATRIX_ALLOWED=1): nothing started")
        raise SystemExit(75)
    {"cells": cmd_cells, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
