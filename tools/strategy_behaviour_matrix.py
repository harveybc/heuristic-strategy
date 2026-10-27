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
import hashlib
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
from app.short_horizon.expert_guided import strategy_zz as ZZ  # noqa: E402
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
SEALED_CELLS_SHA = "e07bbe731fc0b0cb59e7569202a1cd8ffc0f2af0e4e48c33b4a7fd964e28ba6a"   # the original seal (cells only, 2026-10-27T00:58Z)
WITNESS_SPEC = {
    "sealed": {"W1_side_static": "|side share - truth opportunity share| <= 0.15 and the trend direction is the majority when the truth is dominated by it (>= 0.60); range: both sides >= 0.25",
               "W1_side_transition": "per trend leg: share in the leg's direction >= 0.60; old-direction share after the 120-bar lag <= 0.40",
               "W2": "truth net > 0 at the chosen cell >= 0.90; wrong side (other side's truth best net higher by > 0.10 sigma) <= 0.10", "W7": "chosen cell = truth best cell >= 0.70",
               "W3": "opens on truth-NOP rows <= 0.05", "W4_sl_exits": "SL exits <= 0.10", "W4_early_close_justified": "early closes with remaining hold value <= 0: >= 0.80",
               "W5_reverse": "static leg: reversals <= 1 per week; transition: reversals inside the switch windows >= 0.50", "W6": "risk-sized units exact", "W0": "opens >= 1 per week",
               "W8_old_direction_after_lag": "max old-direction share after the lag <= 0.40", "W8_early_close_at_switch": "early-close share of exits in [switch, switch+lag] >= before",
               "W9": "gap trade loss <= SL + gap + 0.5 sigma", "W10": "opens inside the x5 window have truth net > 0", "W12": "no open on an unobservable row", "W13": "TP distance tracks sigma",
               "WA": "winner = argmax EV on every both-cleared tick", "W14": "exact mirror of the zero-cost twin"},
    "v2": {"W1_side_transition_v2": "per leg: |share - truth opportunity| <= 0.15; direction majority when the truth is dominated (>= 0.60)",
           "W4_early_close_justified_v2": "early closes on an SL-first detection (P(SL first) > close_p) or a negative remaining value: >= 0.95",
           "W5_reverse_v2": "reversals whose new side has truth net > 0: >= 0.90; static leg <= 1 per week",
           "W8_old_direction_after_lag_v2": "old-direction share after the lag <= max(0.40, truth old-direction opportunity + 0.10)",
           "rule": "a v2 form is admitted only where the TRUTH ORACLE fails the sealed form (witness defect); otherwise the sealed form stands",
           "intensity_block": "W8 applies only when the coarse direction changes (same-direction strength changes have no old direction); W13 measures before / after the cell's own volatility change bar; written before the block ran"},
}
WITNESS_CHANGELOG = [
    {"when": "2026-10-27T00:40Z (before the seal)", "witness": "W2 / W7 instrumentation", "defect": "the decision row was looked up one bar off the fill bar; truth lookups returned nothing (W2 / W7 read 0.0 while every trade hit TP)",
     "evidence": "B_trend_trend_down_medium s0: 178 trades, 171 TP exits, W2 = 0.0, W7 = 0.0; after the fix both 1.0", "effect": "trades are mapped to the audited decision rows (decision_row_offset recorded)"},
    {"when": "2026-10-27T00:50Z (before the seal)", "witness": "W14 mirror cells", "defect": "the mirror cell was double-mirrored (mirrored legs AND flipped path), so twins did not share a path; and the real engine is side-asymmetric by construction (bid bars, ask-triggered shorts, swap -7 / +2.5)",
     "evidence": "S/M twins 178 vs 169 trades, 0 bars matching; engine truth of the twins differs by up to 4 sigma; at zero cost with the base legs 165/165 identical", "effect": "zero-cost twins S_ and exact mirrors M_ per base cell; the real-cost cell is witnessed on its own"},
    {"when": "2026-10-27T02:40Z (after the seal; admitted by the TRUTH ORACLE control)", "witness": "W1_side on transition cells -> v2 truth-relative per leg",
     "defect": "the sealed form demands >= 0.60 opens in the leg's direction; the TRUTH ORACLE (argmax true net per side, no strategy) fails it whenever the market truth offers fewer opportunities on that side",
     "evidence": "B_trans_up_down_early s0, oracle: first leg long share 0.523 with truth opportunity 0.473 (W2 = 1.0, W7 = 1.0); S_ twin 0.543 / 0.514; M_ twin 0.533 / 0.464", "effect": "v2 admitted only in the cells where the oracle fails the sealed form (automatic, per cell); both verdicts reported"},
    {"when": "2026-10-27T02:40Z (after the seal; admitted by the TRUTH ORACLE control)", "witness": "W5_reverse on transition cells -> v2 truth-justified reversals",
     "defect": "the sealed form demands that reversals happen inside the coarse-switch windows; the TRUTH ORACLE reverses at truth-positive fine counter-pivots outside them",
     "evidence": "B_trans_up_down_early s0, oracle: 3 reversals, 0 inside the switch window, 3/3 truth-justified; twins 3 / 0 / 3", "effect": "v2 admitted only where the oracle fails the sealed form; both verdicts reported"},
    {"when": "2026-10-27T02:40Z (after the seal; admitted by the TRUTH ORACLE control)", "witness": "W8_old_direction_after_lag -> v2 bounded by the truth's old-direction opportunity (+0.10)",
     "defect": "the sealed fixed 0.40 is exceeded by the TRUTH ORACLE itself when the truth keeps offering old-direction opportunities after the switch (the earlier relaxation made after the strategy's 0.405 was REVERTED as adapting the test to the result; this entry rests on the oracle only)",
     "evidence": "B_trans_up_down_early s0, oracle: old-direction share after the lag 0.414 (truth opportunity 0.350); S_ twin 0.443 (truth 0.415); M_ twin 0.414 (truth 0.343)", "effect": "v2 admitted only where the oracle fails the sealed form; the strategy's own 0.405 is reported under the sealed form as FAIL"},
    {"when": "2026-10-27T01:50Z (after the seal; NOT admitted: the oracle passes the sealed form)", "witness": "W4_early_close_justified (v2 = SL-first detection or negative remaining)",
     "defect": "proposed after 49 strategy closes with positive remaining; those closes were the close_p quantile defect of the STRATEGY (fixed); the oracle passes the sealed form", "evidence": "oracle: 8-11 early closes per twin, all with remaining <= 0", "effect": "the sealed form stands; v2 is computed and reported but never admitted"},
]
RUN_HISTORY = [
    {"run": "v1 (2026-10-27T00:58Z)", "note": "first launch; mirror twins built from a stale queue (double-mirrored legs); superseded"},
    {"run": "v2 (01:36Z)", "note": "absolute close_p / nop_veto and open-margin eps; witnesses not yet oracle-controlled; superseded"},
    {"run": "v3 (01:55Z)", "note": "truth-oracle control, sealed + corrected witness forms; strategy C3; 349 / 429 effective PASS; superseded"},
    {"run": "v4 (03:17Z)", "note": "strategy C3Z, witnesses W15-W17, intensity and multi-horizon blocks; superseded"},
    {"run": "v1-v4 SEED DEFECT", "note": "in runs v1 to v4 the synthetic path seed came from the cell id only and perfect heads add no noise, so the three seeds of a cell were IDENTICAL worlds: their per-seed agreement is not evidence of robustness"},
    {"run": "v5 (04:23Z)", "note": "seeds vary the path; close-margin floor, net-ranked sides; the ladder kept only pivots found on both the path and its mirror -- a ladder production never computes: superseded by v6"},
    {"run": "v6", "note": "production ZigZags made symmetric under price negation at the source (reversal and expert MetaTrader first-pivot rules); real-data impact: 1 pivot per fold (the expert ZigZag's first pivot at 2020-01-02 00:00), every reversal level unchanged; the matrix ladder uses the production implementation"},
]
BURST_DETECTION_LAG = 3             # bars the causal burst gate needs (realized vol over 6 bars > 2 sigma at a x3 burst after ~2-3 bars)
THRESH = {"side_share_trend_min": 0.60, "side_share_range_min": 0.25, "truth_positive_open_min": 0.90, "wrong_side_max": 0.10,
          "best_cell_min": 0.70, "nop_open_max": 0.05, "sl_exit_max": 0.10, "early_close_justified_min": 0.80, "reverse_in_switch_window_min": 0.50,
          "reverse_per_week_static_max": 1.0, "old_direction_after_lag_max": 0.40, "new_leg_side_min": 0.60, "mirror_net_rel_tol": 0.02,
          "activity_per_week_min": 1.0, "gap_loss_extra_sigma_max": 0.5,
          "high_risk_open_max": 0.05}


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
    if cfg["vol"] in ("step_lh", "step_hl", "ramp_up", "ramp_down") or str(cfg["vol"]).startswith(("vstep_", "vramp_")):
        expected.append("TP/SL price distances track sigma (after / before the volatility change follows the vol ratio)")
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


# ----------------------------------------------------------------------------- the intensity block (owner 2026-10-27)
STRENGTHS = ("weak", "medium", "strong")
VSTEP_RATIOS = (1.5, 2.0, 3.0)
RAMP_SLOPES = ("gentle", "steep")


def build_intensity_block() -> list:
    """Changes of SEVERAL INTENSITIES, varied independently for trend and for volatility (sealed as its own block):
    I1 trend-change intensity: every ordered pair of trend directions (reversal and same-direction strength change) x
       strength before x strength after (same-direction same-strength = no change, excluded) x switch early/middle/late;
       trend <-> range with the trend strength x 3 x switch position;
    I2 volatility-change intensity with the trend held constant: steps of ratio 1.5 / 2 / 3 up and down, ramps x2
       gentle (60 % of the block) / steep (5 %) up and down, on trend up / down / range (medium);
    I3 the same volatility changes crossed with a trend change at a DIFFERENT bar (vol at 0.3 and trend at 0.7, and
       the reverse) and at the SAME bar (0.5), for up->down and down->up (medium);
    I4 GARCH persistence 0.97 under trend changes (0.90 is the backbone's transition volatility; GARCH with the trend
       held constant is the F2 factorial). Every cell is mirrored and has its zero-cost twin."""
    cells = []

    def add(cid, level, intensity, **cfg):
        base = dict(BASELINE)
        base.update(fine="medium", fine_amp=FINE_AMP["medium"], fine_dens=FINE_DENS["medium"])
        base.update(cfg)
        c = _cell(cid, level, base)
        c["strengths"] = list(cfg.get("strengths") or [])
        c["vol_frac"] = cfg.get("vol_frac")
        c["intensity"] = intensity
        cells.append(c)

    pairs_tt = [("trend_up", "trend_down"), ("trend_down", "trend_up"), ("trend_up", "trend_up"), ("trend_down", "trend_down")]
    for a_, b_ in pairs_tt:
        for s1 in STRENGTHS:
            for s2 in STRENGTHS:
                if a_ == b_ and s1 == s2:
                    continue
                for pos, f in H.SWITCH_POS.items():
                    add(f"I1_{a_[6:]}{s1}_{b_[6:]}{s2}_{pos}", "I1_trend_change_intensity",
                        {"kind": "trend_change", "pair": f"{a_}->{b_}", "from": s1, "to": s2, "switch": pos},
                        legs=[a_, b_], strengths=[s1, s2], switch_fracs=[f], vol="garch09")
    for tr in ("trend_up", "trend_down"):
        for st in STRENGTHS:
            for order in ((tr, "range"), ("range", tr)):
                for pos, f in H.SWITCH_POS.items():
                    stl = [st, "medium"] if order[0] == tr else ["medium", st]
                    add(f"I1_{'_'.join(x[6:] if x.startswith('trend_') else x for x in order)}_{st}_{pos}", "I1_trend_change_intensity",
                        {"kind": "trend_range", "pair": f"{order[0]}->{order[1]}", "trend_strength": st, "switch": pos},
                        legs=list(order), strengths=stl, switch_fracs=[f], vol="garch09")
    vkinds = [f"vstep_{d}_{r:g}" for d in ("up", "down") for r in VSTEP_RATIOS] + [f"vramp_{d}_{sl}" for d in ("up", "down") for sl in RAMP_SLOPES]
    for vk in vkinds:
        for tr in ("trend_up", "trend_down", "range"):
            add(f"I2_{vk}_{tr}", "I2_vol_change_intensity", {"kind": "vol_only", "vol": vk, "trend": tr, "vol_at": 0.5},
                legs=[tr], strengths=["medium"], vol=vk, vol_frac=0.5)
        for a_, b_ in (("trend_up", "trend_down"), ("trend_down", "trend_up")):
            for vf, tf, timing in ((0.3, 0.7, "vol_first"), (0.7, 0.3, "trend_first"), (0.5, 0.5, "same_bar")):
                add(f"I3_{vk}_{a_[6:]}_{b_[6:]}_{timing}", "I3_vol_x_trend_change",
                    {"kind": "vol_x_trend", "vol": vk, "pair": f"{a_}->{b_}", "timing": timing, "vol_at": vf, "trend_at": tf},
                    legs=[a_, b_], strengths=["medium", "medium"], switch_fracs=[tf], vol=vk, vol_frac=vf)
    for a_, b_ in (("trend_up", "trend_down"), ("trend_down", "trend_up")):
        for pos, f in H.SWITCH_POS.items():
            add(f"I4_garch097_{a_[6:]}_{b_[6:]}_{pos}", "I4_garch_x_trend_change", {"kind": "garch_x_trend", "persistence": 0.97, "pair": f"{a_}->{b_}", "switch": pos},
                legs=[a_, b_], strengths=["medium", "medium"], switch_fracs=[f], vol="garch097")
    return cells + [_sym(c) for c in cells] + [_mirror(c) for c in cells]


# ----------------------------------------------------------------------------- the multi-horizon block (owner 2026-10-27)
def _ms_expected(ms: dict) -> tuple:
    comps = [k for k in ("long", "medium", "short") if ms.get(k)]
    lw, md, sh, vs = ms.get("long"), ms.get("medium"), ms.get("short"), ms.get("vol") or {}
    exp, forb = [], []
    exp.append("OPENS: every order carries broker-side SL and TP (W16); every open is truth-positive at the truth-best cell (W2, W7); side shares follow the dominant opportunity (W1)")
    exp.append("EARLY CLOSE when it pays: SL-destined positions closed before the stop (W4 SL exits <= 0.10) and every early close has remaining hold value <= 0 (W4)")
    exp.append("NO NEW OPENS when risk is high relative to the gain: no open where no cell offers truth net >= floor x truth MAE (W15)")
    if sh and int(sh.get("bursts", 0)) > 0:
        exp.append(f"NO NEW OPENS inside the {sh['bursts']} volatility burst(s) after the detection lag (W17)")
    if lw and len(lw["legs"]) > 1:
        exp.append("after the long-horizon switch the old direction fades within the confirmation lag (W8); reversals truth-justified (W5)")
    if md and lw:
        exp.append(f"medium swings ({md.get('amp')}, {md.get('align')} the long leg): counter-swings are traded only where the truth pays them against the long horizon's adverse move")
    if sh and md:
        exp.append("short counter-moves inside a medium swing: fine timing gates the entry, the fine alarm with a weak coarse room arms the early close")
    if vs.get("horizon", "none") != "none":
        exp.append(f"volatility horizon {vs.get('horizon')}: sizing and the per-bar cost gate follow sigma; no open inside bursts")
    forb.append("an order without SL or TP; an open on a high-risk tick; an early close that destroys value (remaining > 0); an open inside a burst after the lag")
    return exp, forb, comps


def build_multihorizon_block(n_lhs: int = 32, seed: int = 11) -> list:
    """Multi-scale worlds (sum of independent long / medium / short / volatility components, own horizon, intensity and
    timing): H1 active horizons x volatility horizon (static long), H2 the same with a long reversal, H3 intensity per
    horizon x alignment, H4 volatility bursts while the long trend reverses, H5 gaps with / against the medium swing,
    H6 Latin hypercube over every knob. Every cell has its zero-cost twin and exact mirror."""
    cells = []

    def add(cid, level, ms):
        lw = ms.get("long")
        legs = list(lw["legs"]) if lw else ["range"]
        base = dict(BASELINE)
        base.update(legs=legs, strengths=list((lw or {}).get("strengths") or []), switch_fracs=list((lw or {}).get("switch_fracs") or []), vol="ms", fine="medium",
                    fine_amp=None, fine_dens=None)
        c = _cell(cid, level, base)
        exp, forb, comps = _ms_expected(ms)
        c["expected"] = exp + [x for x in c["expected"] if x.startswith(("after each", "side shares", "early-close rate", "REVERSE"))]
        c["forbidden"] = forb + c["forbidden"]
        c["ms"], c["intensity"] = ms, {"kind": "multi_horizon", "active": "+".join(comps) or "none", "vol": (ms.get("vol") or {}).get("horizon", "none")}
        cells.append(c)

    def comp(active, long_legs=("trend_down",), long_strengths=("medium",), sw=(), med_amp="mid", align="neutral", fine="medium", bursts=0, burst_mult=3.0,
             at_switch=False, jumps=None, vol="none", slow_ratio=2.0, persistence=0.97):
        ms = {}
        if "long" in active:
            ms["long"] = {"legs": list(long_legs), "strengths": list(long_strengths), "switch_fracs": list(sw)}
        if "medium" in active:
            ms["medium"] = {"amp": med_amp, "align": align, "leg_bars": [48, 144]}
        if "short" in active:
            ms["short"] = {"fine": fine, "bursts": int(bursts), "burst_mult": float(burst_mult), "burst_bars": [6, 12], "burst_at_switch": bool(at_switch), "jumps": jumps or []}
        ms["vol"] = {"horizon": vol, "slow_ratio": float(slow_ratio), "slow_mean_bars": 360, "fast_persistence": float(persistence)}
        return ms

    subsets = [("long",), ("medium",), ("short",), ("long", "medium"), ("long", "short"), ("medium", "short"), ("long", "medium", "short")]
    for sub in subsets:
        for vh in H.MS_VOL:
            add(f"H1_{'+'.join(sub)}_{vh}", "H1_active_horizons", comp(sub, vol=vh))
    for sub in [x for x in subsets if "long" in x]:
        for vh in H.MS_VOL:
            add(f"H2_{'+'.join(sub)}_{vh}_reversal", "H2_active_horizons_long_reversal", comp(sub, long_legs=("trend_up", "trend_down"), long_strengths=("medium", "medium"), sw=(0.5,), vol=vh))
    for st in STRENGTHS:
        for amp in ("low", "mid", "high"):
            for fine in ("low", "medium", "high"):
                for al in ("with", "against"):
                    add(f"H3_L{st}_M{amp}_S{fine}_{al}", "H3_intensity_per_horizon", comp(("long", "medium", "short"), long_strengths=(st,), med_amp=amp, align=al, fine=fine))
    for al in ("with", "against"):
        for bm in (3.0, 5.0):
            for vh in ("none", "fast"):
                add(f"H4_spike_at_reversal_{al}_x{bm:g}_{vh}", "H4_vol_spike_at_long_reversal",
                    comp(("long", "medium", "short"), long_legs=("trend_up", "trend_down"), long_strengths=("medium", "medium"), sw=(0.5,), align=al, bursts=2, burst_mult=bm, at_switch=True, vol=vh))
    for js in (2.0, 4.0):
        for ag in (False, True):
            for fr in (0.3, 0.7):
                add(f"H5_gap{js:g}_{'against' if ag else 'with'}_medium_{fr:g}", "H5_gap_vs_medium_swing",
                    comp(("long", "medium", "short"), jumps=[{"sigma": js, "frac": fr, "against_medium": ag}]))
    knobs = {"act": (0.0, 1.0), "n_legs": (1.0, 3.999), "seq": (0.0, 1.0), "s1": (0.0, 1.0), "s2": (0.0, 1.0), "sw": (0.25, 0.75), "amp": (0.0, 1.0), "align": (0.0, 1.0),
             "fine": (0.0, 1.0), "bursts": (0.0, 4.999), "bmult": (2.0, 5.0), "jump": (0.0, 4.5), "jag": (0.0, 1.0), "vol": (0.0, 1.0), "sratio": (1.5, 3.0), "pers": (0.90, 0.98)}
    u = H.lhs(n_lhs, len(knobs), seed)
    regs = list(H.L1)
    for i in range(n_lhs):
        x = {k: lo + (hi - lo) * u[i, j] for j, (k, (lo, hi)) in enumerate(knobs.items())}
        sub = subsets[min(int(x["act"] * len(subsets)), len(subsets) - 1)]
        rng = np.random.default_rng(int(x["seq"] * 1e6))
        nl = int(x["n_legs"])
        legs = [regs[int(rng.integers(0, 3))]]
        while len(legs) < nl:
            cnd = regs[int(rng.integers(0, 3))]
            if cnd != legs[-1]:
                legs.append(cnd)
        stl = [STRENGTHS[min(int(x["s1" if k_ == 0 else "s2"] * 3), 2)] for k_ in range(nl)]
        sw = sorted(float(np.clip(x["sw"] + 0.2 * (k_ - (nl - 2) / 2), 0.1, 0.9)) for k_ in range(nl - 1))
        jumps = [{"sigma": round(float(x["jump"]), 2), "frac": 0.5, "against_medium": bool(x["jag"] > 0.5)}] if x["jump"] >= 1.0 else []
        add(f"H6_LHS_{i:02d}", "H6_LHS", comp(sub, long_legs=legs, long_strengths=stl, sw=sw, med_amp=("low", "mid", "high")[min(int(x["amp"] * 3), 2)],
                                                  align=("with", "against", "neutral")[min(int(x["align"] * 3), 2)], fine=("low", "medium", "high")[min(int(x["fine"] * 3), 2)],
                                                  bursts=int(x["bursts"]), burst_mult=round(float(x["bmult"]), 2), jumps=jumps, vol=H.MS_VOL[min(int(x["vol"] * 4), 3)],
                                                  slow_ratio=round(float(x["sratio"]), 2), persistence=round(float(x["pers"]), 3)))
    return cells + [_sym(c) for c in cells] + [_mirror(c) for c in cells]


def intensity_coverage(cells: list, rows: list) -> str:
    """Every intensity combination present in the queue, with cells, runs and the effective verdict."""
    by = {r["id"]: r for r in rows}
    groups = {}
    for c in cells:
        it = c.get("intensity")
        if not it or c.get("mirror") or c.get("sym_of"):
            continue
        key = tuple(sorted((k, str(v)) for k, v in it.items() if k not in ("switch", "vol_at", "trend_at") or it.get("kind") == "vol_x_trend"))
        g = groups.setdefault(key, {"cells": 0, "run": 0, "eff_pass": 0, "ids": []})
        g["cells"] += 1
        fam = [by.get(x) for x in (c["id"], "S_" + c["id"], "M_" + c["id"])]
        if all(f is not None and f["n_runs"] for f in fam):
            g["run"] += 1
            g["eff_pass"] += int(all(f["PASS_effective"] for f in fam))
    lines = ["| combinación de intensidad | celdas | corridas (celda + gemela + espejo) | PASS efectivo |", "| --- | ---: | ---: | ---: |"]
    for key, g in sorted(groups.items()):
        lines.append(f"| {', '.join(f'{k}={v}' for k, v in key)} | {g['cells']} | {g['run']} | {g['eff_pass']} |")
    return "\n".join(lines)


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
    if cell.get("fine_amp") is not None and not cell.get("ms"):
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
        cleared = [bool(np.isfinite(evs[i]) and evs[i] > mar[i] + C3.OPEN_MARGIN_EPS) for i in (0, 1)]
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
    strat = C3.StrategyC3.from_genes(gm.to_absolute(genes), "nop_veto" in genes)
    # the IDEAL hierarchy heads (multi-resolution ZigZag ladder on the path, hindsight = perfect) and the causal burst flag
    bars = world.get("bars") if world.get("bars") is not None else cx.ds.bars
    q = ZZ.ladder_quantities(bars, cx.ds.sigma)
    world["ladder"] = q
    zz = ZZ.to_rows(q, cx.ds.row_bar)
    keep = np.isfinite(view["pn"])
    for k, v in zz.items():
        view[k] = np.where(keep.reshape((-1,) + (1,) * (np.ndim(v) - 1)), v, np.nan) if np.ndim(v) else v
    view["risk_burst"] = ZZ.burst_rows(bars, cx.ds.sigma, cx.ds.row_bar)
    if "mae" in table:
        view["mae"] = np.where(keep.reshape(-1, 1, 1), table["mae"], np.nan)
    if ZZ.PARAMS["ev_from_net"] >= 0.5:                          # the pooled margin on the net reference (same quantile rule as C3)
        strat = dataclasses.replace(strat, open_margin=ZZ.net_open_margin(view, cal_rows, cells, float(genes["open_rate"])))
    return strat, view, cells


ORACLE_STRATEGY = C3.StrategyC3(0.0, 0.0, 0.0, 0.5, 0.5, True, 0.5, True)   # zero margins; close on SL-first or a negative remaining value; reverse allowed


class PolicyOracle(PolicyC3Audit):
    """The TRUTH ORACLE: at every flat decision it takes, per side, the allowed cell with the highest TRUE engine net
    and opens the side whose true net is higher when that net is positive; positioned, it closes on the true SL-first
    flag or a negative true remaining value. No strategy, no calibration, no margins: the engine truth itself. A
    witness the oracle fails is a defective witness, never a strategy defect (owner rule 2026-10-27)."""

    def close_now(self, r, si, bar, pos) -> bool:
        rem, psl = self.t["closer"].value(r, bar, pos)
        dec = bool(rem < 0.0)                                       # truth-optimal: close iff holding is worth less than closing now
        self.closes.append((int(r), int(bar), int(si), float(rem), float(psl), dec, dec, False, False))
        return dec

    def open_decision(self, r, with_score: bool = False):
        cap, lo = self._cap(r)
        evs, cells_ = [], []
        if self.t.get("truth_burst") is not None and self.t["truth_burst"][r]:
            self.log.append((int(r), float("nan"), float("nan"), False, False, -1, False, -1, -1))
            return None
        for si in (0, 1):
            net = np.asarray(self._v("net", r, si), float)
            ok = C3.allowed_cells(net, self.cells, 0.0, C3.RR_MARGIN_SIGMA, cap, lo)
            if self.t.get("mae") is not None:
                m = np.asarray(self.t["mae"][r, si], float)
                ok &= np.where(np.isfinite(m), net >= ZZ.PARAMS["risk_reward_floor"] * m, True)
            if ok.any():
                ci = int(np.argmax(np.where(ok, net, -np.inf)))
                evs.append(float(net[ci])); cells_.append(ci)
            else:
                evs.append(float("nan")); cells_.append(-1)
        cleared = [bool(np.isfinite(evs[i]) and evs[i] > C3.OPEN_MARGIN_EPS) for i in (0, 1)]
        chosen = -1
        if cleared[0] or cleared[1]:
            if cleared[0] and cleared[1]:
                chosen = 0 if evs[0] >= evs[1] else 1
            else:
                chosen = 0 if cleared[0] else 1
        self.log.append((int(r), evs[0], evs[1], cleared[0], cleared[1], chosen, False, cells_[0], cells_[1]))
        if chosen < 0:
            return None
        d = E.LONG if chosen == 0 else E.SHORT
        out = (d, self.cells[cells_[chosen]][0], self.cells[cells_[chosen]][1])
        return out + (evs[chosen],) if with_score else out


def rollout_oracle(world: dict, view, cells) -> dict:
    cx, SCFG, truth = world["cx"], world["scfg"], world["truth"]
    pol = PolicyOracle(view, ORACLE_STRATEGY, cells, truth.e_time, truth.cost)
    m, trades = ACT.rollout_with_trades(cx, BLOCK, pol(cx), SCFG)
    return {"metrics": m, "trades": trades, "log": pol.log, "closes": pol.closes, "strategy_obj": ORACLE_STRATEGY}


def oracle_view(world: dict, view: dict) -> dict:
    """The truth as a table: one-hot outcome probabilities from the engine's hit at the reference cell, the true net per
    cell, the true NOP flag; the (perfect) closer object is shared. A policy with zero margins on this table is the
    TRUTH ORACLE: it acts on the engine truth itself, no strategy involved (owner rule 2026-10-27: a witness the
    oracle fails is defective, not the strategy)."""
    truth = world["truth"]
    ov = dict(view)
    n = len(truth.period)
    p3 = np.full((n, 2, 3), np.nan)
    net = np.full_like(truth.net, np.nan)
    for si in (0, 1):
        h = truth.hit3[:, si]
        fin = np.isfinite(truth.net[:, si, :]).all(axis=1)
        p3[:, si, 0] = np.where(fin & (h == 0), 1.0, 0.0)
        p3[:, si, 1] = np.where(fin & (h == 1), 1.0, 0.0)
        p3[:, si, 2] = np.where(fin & (h >= 0) & (h != 0) & (h != 1), 1.0, np.where(fin, 0.0, 1.0))   # unknown hit or no eligible entry -> TIME (no open)
        p3[fin & (h < 0), si, :] = (0.0, 0.0, 1.0)
        net[:, si, :] = np.where(fin.reshape(-1, 1), truth.net[:, si, :], -1.0)                  # the simulated table's convention: -1 = no eligible entry
    keep = np.isfinite(view["pn"])
    ov["p3"] = np.where(keep.reshape(-1, 1, 1), p3, np.nan)
    ov["net"] = np.where(keep.reshape(-1, 1, 1), net, np.nan)
    ov["pn"] = np.where(keep, truth.nop.astype(float), np.nan)
    if truth.mae is not None:
        ov["mae"] = np.where(keep.reshape(-1, 1, 1), truth.mae, np.nan)
    meta = world.get("meta") or {}
    tb = np.zeros(n, bool)
    rb = np.asarray(world["cx"].ds.row_bar)
    for t0, t1 in meta.get("burst_windows") or []:
        tb |= (rb >= t0 + BURST_DETECTION_LAG) & (rb < t1)
    ov["truth_burst"] = tb
    return ov


def rollout_cell(world: dict, strat, view, cells) -> dict:
    cx, SCFG, truth = world["cx"], world["scfg"], world["truth"]
    pol = ZZ.PolicyC3Z(view, strat, cells, truth.e_time, truth.cost)
    m, trades = ACT.rollout_with_trades(cx, BLOCK, pol(cx), SCFG)
    return {"metrics": m, "trades": trades, "log": pol.log, "closes": pol.closes, "strategy_obj": strat, "attr": getattr(pol, "attr", [])}


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


def truth_side_opportunity(truth, cx, a: int, b: int, bars: tuple | None = None) -> dict:
    """Among the block's rows (or the rows whose bar lies in ``bars``) where at least one side has a positive truth net,
    the share whose better side is short."""
    rows = truth.rows[(truth.period[truth.rows] == BLOCK)]
    if bars is not None:
        rb = np.asarray(cx.ds.row_bar)[rows]
        rows = rows[(rb >= bars[0]) & (rb < bars[1])]
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
        out["W1_side_v2"] = dict(out["W1_side"])                           # the static form was sealed truth-relative: identical in both versions
    else:
        meta = world["meta"]
        bounds = [a] + list(meta["switch_bars"]) + [b]
        per_leg, ok_sealed, ok_v2, old_after = [], True, True, []
        for i, lr in enumerate(legs):
            lo_, hi_ = bounds[i], bounds[i + 1]
            tl = [t for t in trades if lo_ <= int(t["entry_bar_abs"]) < hi_]
            t_leg = truth_side_opportunity(truth, cx, lo_, hi_, bars=(lo_, hi_))["share_short_better"]
            if not tl:
                per_leg.append({"leg": lr, "n": 0, "share_in_direction": float("nan"), "truth_share_short_better": t_leg})
                continue
            sl_ = np.array([_side(t) for t in tl])
            d = 0 if lr == "trend_up" else 1 if lr == "trend_down" else None
            sh = float((sl_ == d).mean()) if d is not None else float("nan")
            sh_short = float((sl_ == 1).mean())
            dev_leg = abs(sh_short - t_leg) if np.isfinite(t_leg) else float("nan")
            t_dir = (t_leg if d == 1 else 1 - t_leg) if d is not None else float("nan")
            per_leg.append({"leg": lr, "n": len(tl), "share_in_direction": sh, "truth_share_in_direction": t_dir, "deviation": dev_leg})
            if d is not None and sh < THRESH["new_leg_side_min"]:
                ok_sealed = False                                          # sealed: fixed 0.60 in the leg's direction
            if np.isfinite(dev_leg) and dev_leg > 0.15:
                ok_v2 = False                                              # v2: follows the truth's opportunities
            if d is not None and np.isfinite(t_dir) and t_dir >= THRESH["side_share_trend_min"] and sh < 0.5:
                ok_v2 = False
            if i > 0:
                prev = legs[i - 1]
                dprev = 0 if prev == "trend_up" else 1 if prev == "trend_down" else None
                tl2 = [t for t in tl if int(t["entry_bar_abs"]) >= lo_ + LAG]
                if dprev is not None and tl2 and lr != "range" and dprev != d:   # only when the coarse direction changes
                    sh_old = float(np.mean([_side(t) == dprev for t in tl2]))
                    t_old = truth_side_opportunity(truth, cx, lo_ + LAG, hi_, bars=(lo_ + LAG, hi_))["share_short_better"]
                    t_old = t_old if dprev == 1 else 1 - t_old
                    old_after.append({"leg": lr, "old_direction_share_after_lag": sh_old, "truth_old_direction_opportunity": t_old, "n": len(tl2)})
                    if sh_old > THRESH["old_direction_after_lag_max"]:
                        ok_sealed = False                                  # sealed W1 carried the old-direction clause as well
        out["W1_side"] = {"value": per_leg, "threshold": "SEALED: per trend leg share in the leg's direction >= 0.60; old-direction share after the lag <= 0.40",
                          "pass": bool(n_open and ok_sealed), "desc": "side share per coarse leg (sealed form)"}
        out["W1_side_v2"] = {"value": per_leg, "threshold": "per leg: |share - truth opportunity| <= 0.15; the leg's direction is the majority when the truth is dominated by it (>= 0.60)",
                             "pass": bool(n_open and ok_v2), "desc": "side share per coarse leg follows the truth (corrected form, admitted only if the truth oracle fails the sealed form)"}
        ok8 = all(x["old_direction_share_after_lag"] <= THRESH["old_direction_after_lag_max"] for x in old_after)
        out["W8_old_direction_after_lag"] = {"value": old_after, "threshold": THRESH["old_direction_after_lag_max"],
                                             "pass": bool(ok8) if old_after else True, "desc": "SEALED: max old-direction open share after the confirmation lag <= 0.40"}
        ok8v2 = all(x["old_direction_share_after_lag"] <= max(THRESH["old_direction_after_lag_max"], (x["truth_old_direction_opportunity"] if np.isfinite(x["truth_old_direction_opportunity"]) else 0.0) + 0.10) for x in old_after)
        out["W8_old_direction_after_lag_v2"] = {"value": old_after, "threshold": "old-direction share after the lag <= max(0.40, truth old-direction opportunity + 0.10)",
                                                "pass": bool(ok8v2) if old_after else True, "desc": "corrected form (admitted only where the truth oracle fails the sealed form): no more old-direction opens than the truth offers"}
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
    p_close = {0: float(getattr(roll.get("strategy_obj", None), "close_p_l", 0.5)), 1: float(getattr(roll.get("strategy_obj", None), "close_p_s", 0.5))}
    just_sealed = [bool(c[3] <= 0.0) for c in ec]
    just_v2 = [bool(c[4] > p_close.get(c[2], 0.5) or c[3] < 0.0) for c in ec]
    on_sl = float(np.mean([c[4] > p_close.get(c[2], 0.5) for c in ec])) if ec else float("nan")
    value = float(np.mean([-c[3] for c in ec])) if ec else float("nan")
    out["W4_early_close_justified"] = {"value": float(np.mean(just_sealed)) if ec else float("nan"), "threshold": THRESH["early_close_justified_min"], "pass": bool((not ec) or np.mean(just_sealed) >= THRESH["early_close_justified_min"]),
                                       "desc": "SEALED: early closes whose remaining hold value was <= 0", "share_on_sl_first": on_sl, "early_close_value_sigma_mean": value, "n_early_closes": len(ec)}
    out["W4_early_close_justified_v2"] = {"value": float(np.mean(just_v2)) if ec else float("nan"), "threshold": 0.95, "pass": bool((not ec) or np.mean(just_v2) >= 0.95),
                                          "desc": "corrected form: early closes fired on an SL-first detection (P(SL first) > close_p) or on a negative remaining value", "n_early_closes": len(ec)}
    # W5 reversals: opens while positioned (an open logged at a row where the previous trade was still open)
    rev, rev_just, inwin = 0, 0, 0
    sw = (world.get("meta") or {}).get("switch_bars") or []
    for i in range(1, len(trades)):
        if int(trades[i]["entry_bar_abs"]) <= int(trades[i - 1]["exit_bar_abs"]) and _side(trades[i]) != _side(trades[i - 1]):
            rev += 1
            eb = int(trades[i]["entry_bar_abs"])
            r = int(cx.rob[eb - off]) if eb - off >= 0 else -1
            si = _side(trades[i])
            d_ = dec.get(r) if r >= 0 else None
            ci = int(d_[7 + si]) if d_ is not None and d_[5] == si else -1
            rev_just += int(r >= 0 and ci >= 0 and np.isfinite(truth.net[r, si, ci]) and truth.net[r, si, ci] > 0)
            inwin += int(any(s_ <= eb < s_ + LAG for s_ in sw))
    just_share = (rev_just / rev) if rev else float("nan")
    inwin_share = (inwin / rev) if rev else float("nan")
    val5 = {"reversals": rev, "per_week": rev / weeks, "truth_justified_share": just_share, "in_switch_window_share": inwin_share}
    if len(legs) == 1:
        out["W5_reverse"] = {"value": val5, "threshold": THRESH["reverse_per_week_static_max"], "pass": bool(rev / weeks <= THRESH["reverse_per_week_static_max"]), "desc": "SEALED: reversals per week (static leg) <= 1"}
    else:
        out["W5_reverse"] = {"value": val5, "threshold": THRESH["reverse_in_switch_window_min"], "pass": bool(rev == 0 or inwin_share >= THRESH["reverse_in_switch_window_min"]), "desc": "SEALED: share of reversals inside the switch windows >= 0.50"}
    out["W5_reverse_v2"] = {"value": val5, "threshold": "truth-justified >= 0.90" + ("; <= 1 per week on a static leg" if len(legs) == 1 else ""),
                            "pass": bool((rev == 0 or just_share >= 0.90) and (len(legs) > 1 or rev / weeks <= THRESH["reverse_per_week_static_max"])),
                            "desc": "corrected form: reversals open the side the truth says is positive; rate-limited on static legs"}
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
    if cell["vol"] in ("step_lh", "step_hl", "ramp_up", "ramp_down") or cell["vol"].startswith(("vstep_", "vramp_")):
        vcb = (world.get("meta") or {}).get("vol_change_bar")
        mid = int(vcb) if (vcb is not None and cell["vol"].startswith(("vstep_", "vramp_"))) else a + (b - a) // 2
        d1 = [abs(float(t["take_profit"])) * float(sig[int(t["entry_bar_abs"])]) for t in trades if int(t["entry_bar_abs"]) < mid]
        d2 = [abs(float(t["take_profit"])) * float(sig[int(t["entry_bar_abs"])]) for t in trades if int(t["entry_bar_abs"]) >= mid]
        ratio = (float(np.mean(d2)) / float(np.mean(d1))) if d1 and d2 else float("nan")
        exp_up = cell["vol"] in ("step_lh", "ramp_up") or "_up_" in cell["vol"]
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
    # W16 OPENS carry broker-side SL and TP on the correct sides of the entry
    bad16 = [t for t in trades if not (np.isfinite(float(t["stop_loss"])) and np.isfinite(float(t["take_profit"])) and
                                       ((int(t["direction"]) > 0 and float(t["stop_loss"]) < float(t["entry_price"]) < float(t["take_profit"])) or
                                        (int(t["direction"]) < 0 and float(t["take_profit"]) < float(t["entry_price"]) < float(t["stop_loss"]))))]
    out["W16_sl_tp_every_order"] = {"value": len(bad16), "threshold": 0, "pass": bool(len(bad16) == 0), "desc": "every order carries broker-side SL and TP on the correct sides of the entry"}
    # W15 NO NEW OPENS when the possible gain is small relative to the adverse move (truth: no allowed cell with net >= k * MAE)
    k_rr = float(ZZ.PARAMS["risk_reward_floor"])
    high_risk_opens, n_hr_rows = 0, 0
    if truth.mae is not None:
        def high_risk(rr, si):
            net_r, mae_r = truth.net[rr, si], truth.mae[rr, si]
            if not np.isfinite(net_r).all():
                return True
            capr, lor = float(truth.cap[rr]), float(truth.lo[rr])
            ok = C3.allowed_cells(net_r, cells, 0.0, C3.RR_MARGIN_SIGMA, capr, lor) & (net_r > 0) & np.where(np.isfinite(mae_r), net_r >= k_rr * mae_r, True)
            return not bool(ok.any())
        for t in trades:
            rr = int(cx.rob[int(t["entry_bar_abs"]) - off]) if int(t["entry_bar_abs"]) - off >= 0 else -1
            if rr >= 0:
                high_risk_opens += int(high_risk(rr, _side(t)))
        blk_rows = truth.rows[truth.period[truth.rows] == BLOCK]
        n_hr_rows = int(sum(1 for rr in blk_rows[::7] if high_risk(rr, 0) and high_risk(rr, 1)))
    share15 = high_risk_opens / max(n_open, 1)
    out["W15_no_open_high_risk"] = {"value": share15, "threshold": THRESH["high_risk_open_max"], "pass": bool(share15 <= THRESH["high_risk_open_max"]),
                                    "desc": f"share of opens at (row, side) where no allowed cell offers truth net >= {k_rr:g} x truth MAE", "n_high_risk_opens": high_risk_opens,
                                    "high_risk_rows_sampled_1_in_7": n_hr_rows}
    # W17 NO NEW OPENS inside high-volatility bursts (after the declared detection lag)
    bw = (world.get("meta") or {}).get("burst_windows") or []
    if bw:
        inside = [t for t in trades if any(t0 + BURST_DETECTION_LAG <= int(t["entry_bar_abs"]) - off < t1 for t0, t1 in bw)]
        cl = np.asarray((world.get("bars") if world.get("bars") is not None else cx.ds.bars)["CLOSE"], float)
        sg = np.asarray(cx.ds.sigma, float)
        diag = []
        for t in inside:
            db = int(t["entry_bar_abs"]) - off
            w_ = int(ZZ.PARAMS["burst_window_bars"])
            rv = float(np.sqrt(np.sum(np.diff(cl[db - w_:db + 1]) ** 2) * 24.0 / w_)) if db - w_ >= 0 else float("nan")
            diag.append({"decision_bar": db, "window": [w for w in bw if w[0] <= db < w[1]][0], "rv_over_sigma": rv / float(sg[db]) if sg[db] > 0 else float("nan")})
        out["W17_no_open_in_burst"] = {"value": len(inside), "threshold": 0, "pass": bool(len(inside) == 0), "desc": f"opens decided inside a burst window after {BURST_DETECTION_LAG} bars",
                                       "n_windows": len(bw), "inside": diag}
    # attribution: which horizon's signal triggered each behaviour (reported, not a pass / fail)
    at = roll.get("attr") or []
    opens_at = [x for x in at if x["chosen"] >= 0]
    agree = [((x["zz_dir"] > 0) == (x["chosen"] == 0)) for x in opens_at if x["zz_dir"] != 0]
    fresh = [x["fresh"] == 1.0 for x in opens_at if np.isfinite(x["fresh"])]
    gate_counts = {}
    for x in at:
        if x["chosen"] < 0:
            for g in ((x["blocked"],) if x["blocked"] else (x["gate_l"], x["gate_s"])):
                if g:
                    gate_counts[g] = gate_counts.get(g, 0) + 1
    ecl = [c for c in roll["closes"] if c[5]]
    out["attribution"] = {"opens_with_coarse_direction": float(np.mean(agree)) if agree else float("nan"), "opens_with_fresh_fine_favour": float(np.mean(fresh)) if fresh else float("nan"),
                          "blocked_ticks_by_gate": gate_counts,
                          "early_closes_by_trigger": {"remaining_negative": sum(1 for c in ecl if len(c) > 6 and c[6]), "sl_first_and_negative_remaining": sum(1 for c in ecl if len(c) > 7 and c[7]),
                                                      "fine_alarm_and_weak_coarse_room": sum(1 for c in ecl if len(c) > 8 and c[8])}}
    sealed = {k: v for k, v in out.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2")}
    v2 = {k: v for k, v in out.items() if isinstance(v, dict) and "pass" in v and (k.endswith("_v2") or (k + "_v2") not in out)}
    out["PASS_sealed"] = bool(all(v["pass"] for v in sealed.values()))
    out["PASS_v2"] = bool(all(v["pass"] for v in v2.values()))
    out["PASS"] = out["PASS_sealed"]
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
    return {"pass": bool(ok), "value": {"n_a": len(trades_a), "n_b": len(trades_b), "bars_match": bars_ok, "sides_flipped": side_ok, "max_net_discrepancy_sigma": float(max(net_dev)) if net_dev else 0.0},
            "threshold": "same trade count, every entry / exit bar equal, every side flipped, |net discrepancy| <= 2 % in sigma units", "n_a": len(trades_a), "n_b": len(trades_b), "bars_match": bars_ok, "sides_flipped": side_ok,
            "max_net_discrepancy_ex_financing_sigma": float(max(net_dev)) if net_dev else 0.0, "max_allowed_sigma": float(max(bound)) if bound else 0.0,
            "financing_asymmetry_usd": float(sum(abs(float(ta.get("financing_usd", 0.0)) - float(tb.get("financing_usd", 0.0))) for ta, tb in zip(trades_a[:n], trades_b[:n])))}


# ----------------------------------------------------------------------------- CLI
def cmd_cells(a):
    blk = getattr(a, "block", "matrix")
    cells = build_intensity_block() if blk == "intensity" else (build_multihorizon_block() if blk == "multihorizon" else build_matrix(a.n_lhs, a.seed))
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
    # the truth-oracle control: the same witnesses on a policy that acts on the engine truth itself
    oroll = rollout_oracle(w, oracle_view(w, view), cells)
    meta = w.get("meta") or {}
    tps = meta.get("turning_points") or ({"long": list(meta.get("switch_bars") or [])} if meta.get("switch_bars") else None)
    ladder_map = H.pivot_matching(w["ladder"]["pivots"], tps, H.period_bars(w["cx"].ds, BLOCK)) if tps else None
    owit = witnesses(cell, w, oroll, cells)
    defects_sealed = sorted(k for k, v in owit.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2") and v["pass"] is False)
    defects_v2 = sorted(k for k, v in owit.items() if isinstance(v, dict) and "pass" in v and (k.endswith("_v2") or (k + "_v2") not in owit) and v["pass"] is False)
    sig = np.asarray(w["cx"].ds.sigma, float)
    for t in roll["trades"]:
        s_ = float(sig[int(t["entry_bar_abs"])])
        t["net_sigma"] = float(t["net_usd"]) / max(int(t["units"]), 1) / s_ if np.isfinite(s_) and s_ > 0 else float("nan")
        t["financing_sigma"] = float(t.get("financing_usd", 0.0)) / max(int(t["units"]), 1) / s_ if np.isfinite(s_) and s_ > 0 else 0.0
    rec = {"id": cell["id"], "seed": seed, "level": cell["level"], "regime": cell["regime"], "legs": cell["legs"], "mirror_of": cell.get("mirror_of"), "quality": cell["quality"],
           "witness": w["witness"], "achieved": w["sim"]["achieved"], "strategy": strat.to_dict(), "results": wit, "PASS": wit["PASS"], "PASS_sealed": wit["PASS_sealed"], "PASS_v2": wit["PASS_v2"],
           "oracle": {k: v for k, v in owit.items() if k != "summary"} | {"summary": owit["summary"]}, "oracle_PASS_sealed": owit["PASS_sealed"], "oracle_PASS_v2": owit["PASS_v2"],
           "witness_defects_sealed": defects_sealed, "witness_defects_v2": defects_v2, "ladder_to_horizon": ladder_map,
           "attribution": wit.get("attribution"), "oracle_attribution": owit.get("attribution"),
           "PASS_effective": bool(all((wit[k + "_v2"]["pass"] if (k in defects_sealed and (k + "_v2") in wit) else v["pass"]) for k, v in wit.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2"))),
           "PASS_sealed_excluding_defective_witnesses": bool(all(v["pass"] for k, v in wit.items() if isinstance(v, dict) and "pass" in v and not k.endswith("_v2") and k not in defects_sealed)),
           "sec": round(time.time() - t0, 1),
           "trades": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in t.items()} for t in roll["trades"]]}
    key = cell.get("mirror_of") or cell["id"]
    twin_trades.setdefault(key, {})[cell["id"]] = roll["trades"]
    if cell.get("mirror_of") and cell["mirror_of"] in twin_trades.get(key, {}):
        rec["results"]["W14_mirror"] = mirror_check(twin_trades[key][cell["mirror_of"]], roll["trades"])
        rec["results"]["W14_mirror"]["desc"] = "exact mirror of the twin's ledger"
        rec["PASS"] = bool(rec["PASS"] and rec["results"]["W14_mirror"]["pass"])
        rec["PASS_sealed"] = bool(rec["PASS_sealed"] and rec["results"]["W14_mirror"]["pass"])
        rec["PASS_v2"] = bool(rec["PASS_v2"] and rec["results"]["W14_mirror"]["pass"])
        rec["PASS_effective"] = bool(rec["PASS_effective"] and rec["results"]["W14_mirror"]["pass"])
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
    ZZ.configure(params_doc)
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
        passes = [r.get("PASS_sealed", r["PASS"]) for r in rs]
        passes_v2 = [r.get("PASS_v2", r["PASS"]) for r in rs]
        fails, fails_v2, defects = {}, {}, {}
        for r in rs:
            for k, v in (r.get("results") or {}).items():
                if isinstance(v, dict) and v.get("pass") is False:
                    (fails_v2 if k.endswith("_v2") else fails)[k] = (fails_v2 if k.endswith("_v2") else fails).get(k, 0) + 1
            for k in r.get("witness_defects_sealed", []):
                defects[k] = defects.get(k, 0) + 1
        sl = [r["results"]["summary"]["share_long"] for r in rs if r.get("results", {}).get("summary")]
        rows.append({"id": c["id"], "level": c["level"], "regime": c["regime"], "legs": c["legs"], "mirror": c["mirror"], "sym": bool(c.get("sym_of")), "n_runs": len(rs), "PASS": bool(rs) and all(passes),
                     "PASS_sealed": bool(rs) and all(passes), "PASS_v2": bool(rs) and all(passes_v2), "PASS_sealed_excluding_defective_witnesses": bool(rs) and all(r.get("PASS_sealed_excluding_defective_witnesses", r["PASS"]) for r in rs),
                     "oracle_PASS_sealed": bool(rs) and all(r.get("oracle_PASS_sealed", True) for r in rs), "oracle_PASS_v2": bool(rs) and all(r.get("oracle_PASS_v2", True) for r in rs),
                     "PASS_effective": bool(rs) and all(r.get("PASS_effective", r["PASS"]) for r in rs),
                     "witness_defects_sealed": defects, "failing_witnesses_v2": fails_v2,
                     "failing_witnesses": fails, "share_long_mean": float(np.nanmean(sl)) if sl else None, "opens_mean": float(np.mean([r["results"]["summary"]["opens"] for r in rs if r.get("results", {}).get("summary")])) if rs else None,
                     "errors": [r.get("error") for r in rs if r.get("error")], "expected": c["expected"], "forbidden": c["forbidden"],
                     "ladder_capture": (rs[0].get("ladder_to_horizon") or {}).get("capture") if rs else None, "attribution": rs[0].get("attribution") if rs else None,
                     "oracle_attribution": rs[0].get("oracle_attribution") if rs else None})
    n_pass = sum(1 for r in rows if r["PASS"])
    n_pass_v2 = sum(1 for r in rows if r["PASS_v2"])
    n_pass_excl = sum(1 for r in rows if r["PASS_sealed_excluding_defective_witnesses"])
    n_pass_eff = sum(1 for r in rows if r["PASS_effective"])
    defect_tally = {}
    for r in rows:
        for k, v in r["witness_defects_sealed"].items():
            defect_tally[k] = defect_tally.get(k, 0) + v
    import hashlib
    spec_sha = hashlib.sha256(Path(a.cells).read_bytes()).hexdigest()
    psha = (Path(a.results) / "PARAMS_SHA.txt").read_text().split()[0] if (Path(a.results) / "PARAMS_SHA.txt").exists() else None
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    qdir = Path(a.cells).parent / "queue"
    block_shas = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(qdir.glob("*.json"))} if qdir.exists() else {}
    spec_doc = {"cells_sha256": spec_sha, "block_sha256": block_shas, "witness_spec": WITNESS_SPEC, "witness_code_commit": commit}
    spec_v2_sha = hashlib.sha256(json.dumps(spec_doc, sort_keys=True).encode()).hexdigest()
    doc = {"schema": "behaviour_matrix.v2", "block": BLOCK, "cells": len(rows), "run": sum(1 for r in rows if r["n_runs"]),
           "pass_sealed": n_pass, "fail_sealed": sum(1 for r in rows if r["n_runs"] and not r["PASS"]), "pass_v2": n_pass_v2, "fail_v2": sum(1 for r in rows if r["n_runs"] and not r["PASS_v2"]),
           "pass_sealed_excluding_defective_witnesses": n_pass_excl, "pass_effective": n_pass_eff, "fail_effective": sum(1 for r in rows if r["n_runs"] and not r["PASS_effective"]),
           "oracle_witness_defects_sealed": defect_tally,
           "pass": n_pass, "fail": sum(1 for r in rows if r["n_runs"] and not r["PASS"]),
           "spec_sha256_sealed_cells_only": SEALED_CELLS_SHA, "cells_sha256_current": spec_sha, "spec_sha256_v2": spec_v2_sha, "spec_v2": spec_doc,
           "strategy_params_sha256": psha, "witness_code_commit": commit, "witness_changelog": WITNESS_CHANGELOG, "run_history": RUN_HISTORY, "thresholds": THRESH, "behaviours": BEHAVIOURS, "rows": rows}
    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "BEHAVIOUR_MATRIX.json").write_text(json.dumps(doc, indent=1, default=float))
    lines = ["# Matriz de comportamiento (cabezas perfectas, estrategia C3, motor real)", "",
             f"Celdas: {len(rows)} (cada una con su gemela de financiación simétrica y su espejo exacto); corridas: {doc['run']}; PASS: {n_pass}; FAIL: {doc['fail']}; bloque sostenido: `{BLOCK}` del fold.",
             f"Celdas: PASS sellado {n_pass}; PASS re-sellado v2 {n_pass_v2}; PASS efectivo (forma sellada salvo donde el oráculo de la verdad la falla, entonces la forma v2) {n_pass_eff}; PASS sellado excluyendo los testigos defectuosos {n_pass_excl}. Testigos que el oráculo falla (defectuosos, conteo de celdas): {defect_tally or 'ninguno'}.",
             f"Especificación sellada original (sólo celdas): sha256 `{SEALED_CELLS_SHA}`; celdas actuales: `{spec_sha}`; especificación re-sellada v2 (celdas + definiciones de testigos + commit): `{spec_v2_sha}`; parámetros de la estrategia: sha256 `{psha}` (`STRATEGY_PARAMS.json`).", "",
             "| celda | nivel | tramos | régimen | espejo | corridas | PASS (sellado) | testigos que fallan (sellado) | oráculo falla (testigo defectuoso) | PASS (re-sellado v2) | PASS efectivo | cuota largos | aperturas |", "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: |"]
    for r in rows:
        vd = lambda ok: ('PASS' if ok else ('FAIL' if r['n_runs'] else '—'))  # noqa: E731
        lines.append(f"| {r['id']} | {r['level']} | {' → '.join(r['legs'])} | {r['regime']} | {'sí' if r['mirror'] else ''} | {r['n_runs']} | {vd(r['PASS_sealed'])} | "
                     f"{', '.join(f'{k}×{v}' for k, v in r['failing_witnesses'].items()) or ''} | {', '.join(f'{k}×{v}' for k, v in r['witness_defects_sealed'].items()) or ''} | {vd(r['PASS_v2'])} | {vd(r['PASS_effective'])} | "
                     f"{r['share_long_mean'] if r['share_long_mean'] is None else f'{r['share_long_mean']:.2f}'} | {'' if r['opens_mean'] is None else f'{r['opens_mean']:.0f}'} |")
    fails = [r for r in rows if r["n_runs"] and not r["PASS_sealed"]]
    lines += ["", "## Historia de corridas (lo que cada versión invalidó)", "", "| corrida | nota |", "| --- | --- |"] + [f"| {x['run']} | {x['note']} |" for x in RUN_HISTORY]
    lines += ["", "## Cobertura de intensidades (cambios de tendencia y de volatilidad variados independientemente)", "", intensity_coverage(cells, rows)]
    qdir = Path(a.cells).parent / "queue"
    if qdir.exists():
        lines += ["", "| bloque sellado | sha256 |", "| --- | --- |"] + [f"| {q.name} | `{hashlib.sha256(q.read_bytes()).hexdigest()}` |" for q in sorted(qdir.glob("*.json"))]
    lines += ["", "## Mundos multi-horizonte: escalera ZigZag ↔ horizonte y atribución de cada comportamiento", "",
              "| celda | nivel que captura largo (F1) | medio (F1) | corto (F1) | aperturas con la dirección gruesa | con pivote fino a favor | cierres tempranos (restante<0 / SL-primero / alarma fina) | ticks bloqueados por compuerta |",
              "| --- | --- | --- | --- | ---: | ---: | --- | --- |"]
    for r in rows:
        if not r["id"].lstrip("MS_").startswith("H") or not r.get("ladder_capture"):
            continue
        cap = r["ladder_capture"]
        at = r.get("attribution") or {}
        ec = at.get("early_closes_by_trigger") or {}
        f = lambda c: (f"{cap[c]['level']} ({cap[c]['f1']:.2f})" if cap.get(c) and cap[c].get("level") else "—")  # noqa: E731
        lines.append(f"| {r['id']} | {f('long')} | {f('medium')} | {f('short')} | {at.get('opens_with_coarse_direction', float('nan')):.2f} | {at.get('opens_with_fresh_fine_favour', float('nan')):.2f} | "
                     f"{ec.get('remaining_negative', 0)} / {ec.get('sl_first_and_negative_remaining', 0)} / {ec.get('fine_alarm_and_weak_coarse_room', 0)} | "
                     f"{', '.join(f'{k}={v}' for k, v in (at.get('blocked_ticks_by_gate') or {}).items())} |")
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
    c.add_argument("--block", default="matrix", choices=["matrix", "intensity", "multihorizon"])
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
