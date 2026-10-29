#!/usr/bin/env python3
"""Hierarchical scenario catalog, synthetic generators and the TWO-STAGE optimisation (owner 2026-10-26).

L1 coarse regime (trend up / trend down / range) x volatility (low / high): synthetic paths on the REAL calendar (the
   engine's rules apply unchanged) and real sub-periods by realized-vol / trend terciles (strategy_output_simulation.quarter_tags)
L2 fine structure inside the coarse leg: oscillation amplitude and pivot density of the fine ZigZags (low / medium / high)
L3 events: shocks 2/3/4 sigma, regime switch mid-block, spread x5 24 h, missing runs 6-24 h, vol doubling (strategy_intervention)
L4 head quality: per-head AUC / MAE ladder, calibration drift x0.8 / x1.2, 1-bar lag, side asymmetry, openers-only /
   closers-only / NOP-only (strategy_output_simulation.quality_for)
L5 execution: cap 1/2/4, cutoff 12/24 h, risk 0.5/1/2 %, cost x1/2/3 (strategy_output_simulation.scfg_for)
Each leaf = generator config + expected behaviour + witness. Sampling = the full factorial L1 x L4 at the operating
quality + a Latin-hypercube design over the continuous knobs (drift, vol, fine amplitude, pivot density, quality,
cost factor, cap, cutoff, risk).

Stage A (synthetic): every coefficient optimised on the hierarchical dataset with the owner's composite fitness,
aggregated across leaves by MEAN and CVaR-10 % (worst decile) -> Pareto set + ROBUST REGION (points within 5 % of the
best mean fitness overall AND within 5 % of the best in every L1 regime) -> box bounds per gene.
Stage B (real): strategy_output_simulation.run with the gene bounds restricted to the Stage-A box, on the real
selection block, scored once on the decision block; the Stage-A optimum replayed on the same decision block gives the
out-of-block gain / loss of the re-optimisation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import action_contract as ACT  # noqa: E402
from app.short_horizon.expert_guided import c2 as C2  # noqa: E402
from app.short_horizon.expert_guided import c3 as C3  # noqa: E402
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import strategy_full as SF  # noqa: E402
from app.short_horizon import zigzag as Z  # noqa: E402
from tools import strategy_output_simulation as S  # noqa: E402

L1 = {"trend_up": +1.0, "trend_down": -1.0, "range": 0.0}
TREND_DRIFT_SD = 0.03                            # sustained drift of a trend leg in hourly-sd units per hour (~13 %/yr at sd 7e-4)
VOL = {"low": 0.7, "high": 1.6}                 # multiples of the real block's hourly return sd
L2 = {"low": (1.5, 0.5), "medium": (2.5, 1.0), "high": (4.0, 2.0)}   # (oscillation amplitude in sd units, pivot density multiple)
L4_FACTORIAL = ("base", "onesided", "opener_only", "closer_only", "nop_only", "cal12")
OPERATING_QUALITY = 0.70
PIP = 1e-4


# ----------------------------------------------------------------------------- synthetic generator (L1 x L2 on the real calendar)
def synthetic_path(index, base_sd: float, regime: str, vol: str, fine: str, seed: int, start: float = 1.10) -> pd.DataFrame:
    """Hourly OHLC on the given calendar index: drift by L1 (+/- TREND_DRIFT_SD sd per hour as a slow AR(1) drift: 0.03 sd
    per hour ~ 13 % per year for an hourly sd of 7e-4 -- the first version used 0.35 sd per hour, which collapsed a
    two-year trend path by e^-4 and made the engine refuse the short levels; see the mirror / trend witnesses), 0 for range,
    range = mean-reverting around the start), GARCH(1,1)-like volatility clustering scaled by VOL, plus an L2 fine
    oscillation (sinusoidal swings with random period 6-18 h, amplitude x and density x) that creates the fine ZigZag
    pivots. Witness: the realised drift sign, the sd ratio and the fine-pivot density match the leaf (tests)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    sd = base_sd * VOL[vol]
    amp, dens = L2[fine]
    # GARCH-like variance
    h = np.full(n, sd ** 2)
    eps = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.08 * sd ** 2 + 0.10 * eps[t - 1] ** 2 + 0.82 * h[t - 1]
        eps[t] = math.sqrt(h[t]) * rng.standard_normal()
    drift = np.zeros(n)
    k = L1[regime]
    if k != 0.0:
        d = 0.0
        for t in range(n):
            d = 0.98 * d + 0.02 * (k * TREND_DRIFT_SD * sd) + 0.02 * sd * rng.standard_normal()
            drift[t] = d
    logp = np.log(start) + np.cumsum(drift + eps)
    if k == 0.0:                                          # range: pull towards the start level
        for t in range(1, n):
            logp[t] = logp[t] - 0.03 * (logp[t - 1] - math.log(start))
    # L2 fine oscillation: piecewise sinusoids with random periods, amplitude in sd units, density scales the frequency
    osc = np.zeros(n)
    t = 0
    while t < n:
        per = int(max(4, rng.integers(6, 19) / dens))
        a = amp * sd * rng.uniform(0.8, 1.2)
        seg = min(per, n - t)
        osc[t:t + seg] = a * np.sin(2 * np.pi * np.arange(seg) / per)
        t += seg
    logp = logp + osc
    c = np.exp(logp)
    intr = np.abs(rng.standard_normal(n)) * sd * 0.6
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + intr)
    lo = np.minimum(o, c) * (1 - intr)
    return pd.DataFrame({"OPEN": o, "HIGH": hi, "LOW": lo, "CLOSE": c}, index=index)



# ----------------------------------------------------------------------------- L1b: transitions and varying volatility (owner 2026-10-27)
SEQUENCES = {"up_down": ("trend_up", "trend_down"), "down_up": ("trend_down", "trend_up"), "up_range": ("trend_up", "range"),
             "range_up": ("range", "trend_up"), "down_range": ("trend_down", "range"), "range_down": ("range", "trend_down"),
             "up_pause_up": ("trend_up", "range", "trend_up"), "down_pause_down": ("trend_down", "range", "trend_down")}
SWITCH_POS = {"early": 0.25, "middle": 0.5, "late": 0.75}
VOL_PROCESSES = ("step_lh", "step_hl", "garch09", "garch097", "ramp_up", "ramp_down")
CONFIRMATION_LAG_BARS = 120          # declared coarse confirmation lag: five trading days of hourly bars after a coarse switch
MIRROR = {"trend_up": "trend_down", "trend_down": "trend_up", "range": "range"}
EXPECT_TRANSITION = {"old_direction_open_share_after_lag_max": 0.40, "new_leg_side_share_min": 0.60, "early_close_rate_ratio_min": 1.0,
                     "dd_at_switch_bound": "DD in [switch, switch + lag] with early close <= DD without early close (the intervention ratio)",
                     "lag_bars": CONFIRMATION_LAG_BARS}


def vol_multiplier(n: int, vol: str, block: tuple, switch_frac: float = 0.5) -> np.ndarray:
    """Per-bar sd multiplier of the volatility PROCESS: a constant level (VOL), a step low<->high at the switch
    position inside the block, or a ramp across the block; the GARCH kinds carry their clustering in the innovation."""
    a, b = int(block[0]), int(block[1])
    m = np.ones(n)
    if vol.startswith("vstep_"):                                   # intensity block: step of a declared ratio at its OWN change bar
        _, dirn, r = vol.split("_")
        r = float(r)
        sw = a + int(switch_frac * (b - a))
        m[:sw], m[sw:] = (1.0, r) if dirn == "up" else (r, 1.0)
        return m
    if vol.startswith("vramp_"):                                   # intensity block: ramp x2 over 60 % (gentle) or 5 % (steep) of the block
        _, dirn, slope = vol.split("_")
        span = int((0.60 if slope == "gentle" else 0.05) * (b - a))
        sw = a + int(switch_frac * (b - a))
        e = min(sw + max(span, 1), n)
        lo_, hi_ = (1.0, 2.0) if dirn == "up" else (2.0, 1.0)
        m[:sw] = lo_
        m[sw:e] = np.linspace(lo_, hi_, e - sw)
        m[e:] = hi_
        return m
    if vol in VOL:
        m[:] = VOL[vol]
    elif vol in ("step_lh", "step_hl"):
        sw = a + int(switch_frac * (b - a))
        lo_, hi_ = (VOL["low"], VOL["high"]) if vol == "step_lh" else (VOL["high"], VOL["low"])
        m[:sw], m[sw:] = lo_, hi_
    elif vol in ("ramp_up", "ramp_down"):
        lo_, hi_ = (VOL["low"], VOL["high"]) if vol == "ramp_up" else (VOL["high"], VOL["low"])
        m[:a] = lo_
        m[a:b] = np.linspace(lo_, hi_, max(b - a, 1))
        m[b:] = hi_
    return m


TREND_STRENGTH = {"weak": 0.5, "medium": 1.0, "strong": 2.0}     # multiples of TREND_DRIFT_SD (0.015 / 0.03 / 0.06 sd per hour)


def synthetic_path_v2(index, base_sd: float, legs: tuple, vol: str, fine: str, seed: int, block: tuple, switch_fracs=None,
                      persistence: float | None = None, start: float = 1.10, mirror: bool = False, strength: str = "medium", jump: dict | None = None,
                      strengths: list | None = None, vol_frac: float | None = None, mirror_tick: float | None = None):
    """Multi-leg hourly OHLC on the real calendar: coarse legs (regimes) switching at ``switch_fracs`` of the scoring
    block (the first leg runs from the start of the index, the last to its end), a volatility process (constant level,
    step, ramp, GARCH clustering with the given persistence), the L2 fine oscillation, and optionally the exact log
    mirror of the whole path (negated log returns; HIGH/LOW swapped and mirrored). Returns (bars, meta)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    a, b = int(block[0]), int(block[1])
    legs = tuple(legs)
    if switch_fracs is None:
        switch_fracs = [(i + 1) / len(legs) for i in range(len(legs) - 1)]
    switches = [a + int(float(f) * (b - a)) for f in switch_fracs]
    bounds = [0] + switches + [n]
    regime_at = np.empty(n, object)
    strength_at = np.empty(n, object)
    for i, reg in enumerate(legs):
        regime_at[bounds[i]:bounds[i + 1]] = reg
        strength_at[bounds[i]:bounds[i + 1]] = (strengths[i] if strengths else strength)
    pers = persistence if persistence is not None else {"garch09": 0.90, "garch097": 0.97}.get(vol, 0.92)
    alpha = 0.08
    beta = max(0.0, pers - alpha)
    omega = max(1e-6, 1.0 - alpha - beta)
    vfrac = float(vol_frac) if vol_frac is not None else (switch_fracs[0] if switch_fracs else 0.5)
    mult = vol_multiplier(n, vol, block, vfrac)
    amp, dens = L2[fine]
    h = np.ones(n)
    eps = np.zeros(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        prev = eps[t - 1] / (base_sd * mult[t - 1])
        h[t] = omega + alpha * prev * prev + beta * h[t - 1]
        eps[t] = base_sd * mult[t] * math.sqrt(h[t]) * z[t]
    drift = np.zeros(n)
    d = 0.0
    zn = rng.standard_normal(n)
    for t in range(n):
        k = L1[regime_at[t]] * TREND_STRENGTH[strength_at[t]]
        d = 0.98 * d + 0.02 * (k * TREND_DRIFT_SD * base_sd) + 0.02 * base_sd * zn[t]
        drift[t] = d
    logp = np.log(start) + np.cumsum(drift + eps)
    anchor = None
    for t in range(1, n):
        if regime_at[t] == "range":
            if anchor is None or regime_at[t - 1] != "range":
                anchor = logp[t - 1]
            logp[t] = logp[t] - 0.03 * (logp[t - 1] - anchor)
        else:
            anchor = None
    osc = np.zeros(n)
    t = 0
    while t < n:
        per = int(max(4, rng.integers(6, 19) / dens))
        seg = min(per, n - t)
        a_ = amp * base_sd * float(np.mean(mult[t:t + seg])) * rng.uniform(0.8, 1.2)
        osc[t:t + seg] = a_ * np.sin(2 * np.pi * np.arange(seg) / per)
        t += seg
    logp = logp + osc
    jump_bar = None
    if jump and float(jump.get("sigma", 0)) > 0:                    # an overnight gap of k sigma at a bar inside the block, with or against the coarse leg
        jump_bar = a + int(float(jump.get("frac", 0.5)) * (b - a))
        leg_sign = L1[regime_at[jump_bar]] if L1[regime_at[jump_bar]] != 0 else 1.0
        sign = leg_sign if not jump.get("against") else -leg_sign
        logp[jump_bar:] += sign * float(jump["sigma"]) * base_sd * float(mult[jump_bar]) * math.sqrt(24.0)
    c = np.exp(logp)
    intr = np.abs(rng.standard_normal(n)) * base_sd * mult * 0.6
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + intr)
    lo = np.minimum(o, c) * (1 - intr)
    bars = pd.DataFrame({"OPEN": o, "HIGH": hi, "LOW": lo, "CLOSE": c}, index=index)
    if mirror:
        bars = mirror_bars(bars, tick=mirror_tick)
    meta = {"switch_bars": [int(x) for x in switches], "switch_fracs": [float(f) for f in switch_fracs], "legs": list(legs), "vol": vol,
            "persistence": float(pers), "block": [a, b], "mirror": bool(mirror), "strength": strength, "jump_bar": jump_bar, "jump": jump,
            "strengths": list(strengths) if strengths else [strength] * len(legs),
            "vol_change_bar": (a + int(vfrac * (b - a))) if (vol.startswith(("vstep_", "vramp_")) or vol in ("step_lh", "step_hl", "ramp_up", "ramp_down")) else None}
    return bars, meta


MS_MED_AMP = {"low": 0.05, "mid": 0.10, "high": 0.20}      # medium swing slope in hourly-sd units per bar
MS_SHORT_FINE = {"low": (1.5, 0.5), "medium": (2.5, 1.0), "high": (4.0, 2.0)}
MS_VOL = ("none", "slow", "fast", "both")


def synthetic_path_ms(index, base_sd: float, spec: dict, seed: int, block: tuple, mirror: bool = False, start: float = 1.10, mirror_tick: float | None = None):
    """A MULTI-SCALE world as a sum of independent components, each with its own horizon, intensity and timing
    (owner 2026-10-27). ``spec``:
      long   {"legs": [...], "strengths": [...], "switch_fracs": [...]} or None   weeks-months (the AR(1) trend drift)
      medium {"amp": low|mid|high, "align": with|against|neutral, "leg_bars": [lo, hi]} or None   days: alternating
             swings; 'with' = swings in the long leg's direction last 1.5x longer, 'against' = counter-swings do
      short  {"fine": low|medium|high, "bursts": n, "burst_mult": x, "burst_bars": [lo, hi], "burst_at_switch": bool,
             "jumps": [{"sigma", "frac", "against_medium"}]} or None   hours: fine pivots, volatility bursts, gaps
      vol    {"horizon": none|slow|fast|both, "slow_ratio": r, "slow_mean_bars": m, "fast_persistence": p}
    Each component draws from its own random stream (independent timing). Returns (bars, meta) with every planted
    turning point per component (long switches, medium swing ends, short oscillation extrema), burst windows and
    jump bars: the truth the ladder-to-horizon mapping is scored against."""
    root = np.random.SeedSequence(int(seed))
    r_long, r_med, r_short, r_vol, r_eps = (np.random.default_rng(x) for x in root.spawn(5))
    n = len(index)
    a, b = int(block[0]), int(block[1])
    sd = base_sd
    # --- volatility process (own horizon)
    vs = spec.get("vol") or {"horizon": "none"}
    hz = vs.get("horizon", "none")
    mult = np.ones(n)
    slow_switches = []
    if hz in ("slow", "both"):
        state, t = 0, 0
        mean_bars = float(vs.get("slow_mean_bars", 360))
        while t < n:
            dur = int(max(24, r_vol.exponential(mean_bars)))
            mult[t:t + dur] = float(vs.get("slow_ratio", 2.0)) if state else 1.0
            t += dur
            state = 1 - state
            if t < n:
                slow_switches.append(int(t))
    pers = float(vs.get("fast_persistence", 0.97)) if hz in ("fast", "both") else 0.50
    alpha = 0.08 if hz in ("fast", "both") else 0.05
    beta = max(0.0, pers - alpha)
    omega = max(1e-6, 1.0 - alpha - beta)
    # --- short bursts (multiply the volatility inside short windows)
    sh = spec.get("short")
    burst_windows = []
    if sh and int(sh.get("bursts", 0)) > 0:
        lo_b, hi_b = sh.get("burst_bars", [6, 12])
        lw = spec.get("long") or {}
        sw_bars = [a + int(f * (b - a)) for f in (lw.get("switch_fracs") or [])]
        for k in range(int(sh["bursts"])):
            if sh.get("burst_at_switch") and k < len(sw_bars):
                t0 = sw_bars[k]
            else:
                t0 = int(r_short.integers(a, max(a + 1, b - 24)))
            dur = int(r_short.integers(int(lo_b), int(hi_b) + 1))
            mult[t0:t0 + dur] *= float(sh.get("burst_mult", 3.0))
            burst_windows.append([int(t0), int(t0 + dur)])
    h = np.ones(n)
    eps = np.zeros(n)
    z = r_eps.standard_normal(n)
    for t in range(1, n):
        prev = eps[t - 1] / (sd * mult[t - 1])
        h[t] = omega + alpha * prev * prev + beta * h[t - 1]
        eps[t] = sd * mult[t] * math.sqrt(h[t]) * z[t]
    # --- long component
    lw = spec.get("long")
    regime_at = np.full(n, "range", object)
    strength_at = np.full(n, "medium", object)
    long_switches = []
    drift = np.zeros(n)
    if lw:
        legs = list(lw["legs"])
        fr = list(lw.get("switch_fracs") or [(i + 1) / len(legs) for i in range(len(legs) - 1)])
        long_switches = [a + int(f * (b - a)) for f in fr]
        bounds = [0] + long_switches + [n]
        stl = list(lw.get("strengths") or ["medium"] * len(legs))
        for i, reg in enumerate(legs):
            regime_at[bounds[i]:bounds[i + 1]] = reg
            strength_at[bounds[i]:bounds[i + 1]] = stl[i]
        d = 0.0
        zn = r_long.standard_normal(n)
        for t in range(n):
            k = L1[regime_at[t]] * TREND_STRENGTH[strength_at[t]]
            d = 0.98 * d + 0.02 * (k * TREND_DRIFT_SD * sd) + 0.02 * sd * zn[t]
            drift[t] = d
    logp = np.log(start) + np.cumsum(drift + eps)
    if lw:
        anchor = None
        for t in range(1, n):
            if regime_at[t] == "range":
                if anchor is None or regime_at[t - 1] != "range":
                    anchor = logp[t - 1]
                logp[t] = logp[t] - 0.03 * (logp[t - 1] - anchor)
            else:
                anchor = None
    # --- medium swings (days)
    md = spec.get("medium")
    med = np.zeros(n)
    med_ends, med_dir = [], np.zeros(n)
    if md:
        slope = MS_MED_AMP[md.get("amp", "mid")] * sd
        lo_m, hi_m = md.get("leg_bars", [48, 144])
        t, sgn, level = 0, (1.0 if r_med.random() < 0.5 else -1.0), 0.0
        while t < n:
            L = int(r_med.integers(int(lo_m), int(hi_m) + 1))
            longdir = L1[regime_at[min(t, n - 1)]]
            if md.get("align", "neutral") == "with" and longdir != 0 and sgn == longdir:
                L = int(L * 1.5)
            if md.get("align", "neutral") == "against" and longdir != 0 and sgn == -longdir:
                L = int(L * 1.5)
            e = min(t + L, n)
            med[t:e] = level + sgn * slope * np.arange(1, e - t + 1)
            med_dir[t:e] = sgn
            level = med[e - 1]
            if e < n:
                med_ends.append(int(e))
            t, sgn = e, -sgn
        logp = logp + med
    # --- short oscillation and jumps (hours)
    short_ext = []
    jump_bars = []
    if sh:
        amp, dens = MS_SHORT_FINE[sh.get("fine", "medium")]
        t = 0
        osc = np.zeros(n)
        while t < n:
            per = int(max(4, r_short.integers(6, 19) / dens))
            seg = min(per, n - t)
            a_ = amp * sd * float(np.mean(mult[t:t + seg])) * r_short.uniform(0.8, 1.2)
            osc[t:t + seg] = a_ * np.sin(2 * np.pi * np.arange(seg) / per)
            for q in (per // 4, (3 * per) // 4):
                if q < seg:
                    short_ext.append(int(t + q))
            t += seg
        logp = logp + osc
        for jp in sh.get("jumps") or []:
            jb = a + int(float(jp.get("frac", 0.5)) * (b - a))
            ref = med_dir[jb] if md else (L1[regime_at[jb]] or 1.0)
            sign = -ref if jp.get("against_medium") else ref
            logp[jb:] += sign * float(jp["sigma"]) * sd * float(mult[jb]) * math.sqrt(24.0)
            jump_bars.append(int(jb))
    c = np.exp(logp)
    intr = np.abs(r_eps.standard_normal(n)) * sd * mult * 0.6
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + intr)
    lo = np.minimum(o, c) * (1 - intr)
    bars = pd.DataFrame({"OPEN": o, "HIGH": hi, "LOW": lo, "CLOSE": c}, index=index)
    if mirror:
        bars = mirror_bars(bars, tick=mirror_tick)
    meta = {"kind": "multi_scale", "block": [a, b], "mirror": bool(mirror), "spec": spec, "switch_bars": long_switches,
            "legs": list(lw["legs"]) if lw else ["range"], "strengths": list(lw.get("strengths") or []) if lw else [],
            "turning_points": {"long": long_switches, "medium": med_ends, "short": short_ext}, "burst_windows": burst_windows,
            "jump_bars": jump_bars, "vol_slow_switches": slow_switches, "jump_bar": (jump_bars[0] if jump_bars else None),
            "jump": ((sh.get("jumps") or [None])[0] if sh else None), "vol_change_bar": None}
    return bars, meta


def pivot_matching(pivots_by_level: dict, turning_points: dict, block: tuple, tol=None) -> dict:
    """Per ladder level and planted component: recall (share of the component's true turning points inside the block
    with a level pivot within the tolerance), precision (share of the level's pivots near a point of that component)
    and F1; the level that captures each component = argmax F1."""
    tol = tol or {"long": 72, "medium": 12, "short": 2}
    a, b = int(block[0]), int(block[1])
    out, capture = {}, {}
    for lvl, piv in pivots_by_level.items():
        pb = np.array([int(p[0]) for p in piv if a <= int(p[0]) < b], np.int64)
        out[lvl] = {}
        for comp, pts in turning_points.items():
            tp = np.array([x for x in pts if a <= x < b], np.int64)
            if not len(tp) or not len(pb):
                out[lvl][comp] = {"recall": float("nan") if not len(tp) else 0.0, "precision": 0.0 if len(tp) else float("nan"), "f1": float("nan") if not len(tp) else 0.0, "n_true": int(len(tp)), "n_pivots": int(len(pb))}
                continue
            t_ = int(tol.get(comp, 6))
            rec = float(np.mean([np.any(np.abs(pb - x) <= t_) for x in tp]))
            prec = float(np.mean([np.any(np.abs(tp - y) <= t_) for y in pb]))
            f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
            out[lvl][comp] = {"recall": rec, "precision": prec, "f1": f1, "n_true": int(len(tp)), "n_pivots": int(len(pb))}
    for comp in turning_points:
        best = max(((lvl, out[lvl][comp]["f1"]) for lvl in out if np.isfinite(out[lvl][comp]["f1"])), key=lambda x: x[1], default=(None, float("nan")))
        capture[comp] = {"level": best[0], "f1": best[1]}
    return {"per_level": out, "capture": capture, "tolerance_bars": tol}


def mirror_bars(bars: pd.DataFrame, tick: float | None = None) -> pd.DataFrame:
    """The exact PRICE mirror of a path: p' = K - p with K = max(HIGH) + min(LOW) (every price change negated), HIGH' =
    mirror(LOW), LOW' = mirror(HIGH). The engine works in price units (sigma from squared close differences, TP / SL
    distances, spreads), so the additive mirror keeps every distance, every sigma and every range identical and flips
    every move: the strategy's trades must mirror exactly (a log mirror would change the price-unit sigma)."""
    K = float(np.nanmax(bars["HIGH"])) + float(np.nanmin(bars["LOW"]))   # reflect around the middle of the path's range: the mirror
    if tick is not None:                                                    # full_f4 (F10 2026-10-29): levels live on a tick grid (SL away, TP toward the entry);
        K = round(K / float(tick)) * float(tick)                            # the floor / ceil rounding mirrors exactly only when K is ON the grid
    f = lambda x: K - np.asarray(x, float)  # noqa: E731                     # stays inside [min, max] (around the first close a strong
                                                                            # 3-year uptrend mirrored to prices near or below zero)
    return pd.DataFrame({"OPEN": f(bars["OPEN"]), "HIGH": f(bars["LOW"]), "LOW": f(bars["HIGH"]), "CLOSE": f(bars["CLOSE"])}, index=bars.index)


def transition_witness(bars: pd.DataFrame, base_sd: float, meta: dict) -> dict:
    """Per coarse leg (inside the block): realised drift sign against the leg's regime (mirrored legs expect the mirrored
    sign); per vol segment: sd ratio; the fine-pivot density; the switch bars."""
    r = np.r_[0.0, np.diff(np.log(bars["CLOSE"].to_numpy(float)))]
    a, b = meta["block"]
    bounds = [a] + list(meta["switch_bars"]) + [b]
    legs = []
    for i, reg in enumerate(meta["legs"]):
        seg = r[bounds[i]:bounds[i + 1]]
        exp = L1[reg] * (-1.0 if meta.get("mirror") else 1.0)
        legs.append({"regime": reg, "bars": [int(bounds[i]), int(bounds[i + 1])], "drift_sign": float(np.sign(seg.sum())) if len(seg) else 0.0,
                     "expected_drift_sign": exp, "sd_ratio": float(seg.std() / base_sd) if len(seg) > 2 else float("nan")})
    mid = a + (b - a) // 2
    vol_ratio = float(r[mid:b].std() / max(r[a:mid].std(), 1e-12)) if b - a > 4 else float("nan")
    vol = meta["vol"]
    exp_vol = {"step_lh": ">1", "ramp_up": ">1", "step_hl": "<1", "ramp_down": "<1"}.get(vol, "~1 (clustering, no level change)")
    return {"legs": legs, "switch_bars": list(meta["switch_bars"]), "vol_second_over_first_half": vol_ratio, "expected_vol_ratio": exp_vol,
            "fine_pivots_per_100_bars": float(100.0 * fine_pivot_count(bars, base_sd) / max(len(r), 1)), "mirror": bool(meta.get("mirror"))}


def expected_transition_text(legs: tuple, switch_fracs, vol: str, mirror: bool = False) -> str:
    legs_s = " -> ".join(MIRROR[x] if mirror else x for x in legs)
    return (f"coarse legs {legs_s} switching at {', '.join(f'{f:.2f}' for f in switch_fracs)} of the block; after each switch the strategy stops "
            f"opening in the old direction within {CONFIRMATION_LAG_BARS} bars (old-direction share of opens <= 0.40 after the lag at quality >= 0.7), "
            f"early closes rise at the switch (rate in [switch, switch + lag] >= rate before), side shares follow the new leg (>= 0.60), "
            f"and the drawdown around the switch with early close is bounded by the no-early-close counterfactual; volatility process {vol}")


TRANSITION_KNOBS = {"n_legs": (2.0, 4.999), "seq_seed": (0.0, 1.0), "switch_spread": (0.2, 0.8), "persistence": (0.85, 0.98), "ramp": (-1.0, 1.0),
                    "fine_amp": (0.4, 2.0), "fine_dens": (0.5, 2.0), "quality": (0.5, 1.0), "cost_mult": (1.0, 3.0), "cap": (1, 4), "cutoff_h": (12.0, 24.0),
                    "risk_pct": (0.005, 0.02)}


def catalog_transitions(n_lhs: int = 24, seed: int = 1) -> list:
    """L1b leaves: T1 every ordered regime pair (and the paused same-direction pairs) x switch position x fine structure;
    T2 the volatility processes on each static regime; T3 the quality ladder on the middle-switch pairs; T4 Latin-hypercube
    multi-leg blocks over the transition knobs; and the exact mirror of every T1 and T3 leaf."""
    leaves = []
    for name, legs in SEQUENCES.items():
        for pos, f in SWITCH_POS.items():
            fr = [f] if len(legs) == 2 else [max(0.05, f - 0.12), min(0.95, f + 0.12)]
            for fine in L2:
                leaves.append({"id": f"T1_{name}_{pos}_{fine}", "level": "L1b_T1", "legs": list(legs), "switch_fracs": fr, "vol": "garch09", "fine": fine,
                               "regime": legs[-1], "variant": "base", "quality": OPERATING_QUALITY, "exec": {},
                               "expected": expected_transition_text(legs, fr, "garch09"), "witness": "transition_witness: per-leg drift sign, vol ratio, switch bars",
                               "expect": EXPECT_TRANSITION})
    for reg in L1:
        for vp in VOL_PROCESSES:
            leaves.append({"id": f"T2_{reg}_{vp}", "level": "L1b_T2", "legs": [reg], "switch_fracs": [], "vol": vp, "fine": "medium", "regime": reg,
                           "variant": "base", "quality": OPERATING_QUALITY, "exec": {},
                           "expected": f"static {reg} under volatility process {vp}: openers keep paying; sizing (risk) and the per-bar cost gate adapt to the new sigma; activity >= 2/wk",
                           "witness": "transition_witness: vol second/first half ratio", "expect": {}})
    for name, legs in SEQUENCES.items():
        fr = [0.5] if len(legs) == 2 else [0.38, 0.62]
        for q in (0.5, 0.6, 0.8, 1.0):
            leaves.append({"id": f"T3_{name}_q{q:.1f}", "level": "L1b_T3", "legs": list(legs), "switch_fracs": fr, "vol": "garch09", "fine": "medium",
                           "regime": legs[-1], "variant": "base", "quality": q, "exec": {},
                           "expected": expected_transition_text(legs, fr, "garch09") + f"; at quality {q:.1f} (chance must not pay, oracle must)",
                           "witness": "transition_witness + achieved head quality", "expect": EXPECT_TRANSITION})
    u = lhs(n_lhs, len(TRANSITION_KNOBS), seed)
    regs = list(L1)
    for i in range(n_lhs):
        x = {k: lo + (hi - lo) * u[i, j] for j, (k, (lo, hi)) in enumerate(TRANSITION_KNOBS.items())}
        n_legs = int(x["n_legs"])
        rng = np.random.default_rng(int(x["seq_seed"] * 1e6))
        legs = [regs[int(rng.integers(0, 3))]]
        while len(legs) < n_legs:
            cand = regs[int(rng.integers(0, 3))]
            if cand != legs[-1]:
                legs.append(cand)
        centre = 0.5
        spread = x["switch_spread"]
        fr = sorted(float(np.clip(centre + spread * (k_ / (n_legs - 1) - 0.5), 0.08, 0.92)) for k_ in range(n_legs - 1)) if n_legs > 1 else []
        vol = "ramp_up" if x["ramp"] > 0.5 else "ramp_down" if x["ramp"] < -0.5 else "garch09"
        fine = "high" if x["fine_amp"] > 1.4 else "low" if x["fine_amp"] < 0.7 else "medium"
        leaves.append({"id": f"T4_{i:02d}", "level": "L1b_T4", "legs": legs, "switch_fracs": fr, "vol": vol, "persistence": round(x["persistence"], 3), "fine": fine,
                       "regime": legs[-1], "variant": "base", "quality": round(x["quality"], 3),
                       "exec": {"cost_mult": round(x["cost_mult"], 2), "cap": int(round(x["cap"])), "cutoff_h": round(x["cutoff_h"], 1), "risk_pct": round(x["risk_pct"], 4)},
                       "knobs": {k: round(v, 4) for k, v in x.items()}, "expected": expected_transition_text(legs, fr, vol),
                       "witness": "transition_witness + the leaf's own fitness archive", "expect": EXPECT_TRANSITION})
    mirrors = []
    for x in leaves:
        if x["level"] in ("L1b_T1", "L1b_T3"):
            m = dict(x)
            m["id"], m["mirror_of"], m["level"], m["mirror"] = "M_" + x["id"], x["id"], x["level"] + "_mirror", True
            m["regime"] = MIRROR[x["regime"]]
            m["expected"] = expected_transition_text(tuple(x["legs"]), x["switch_fracs"], x["vol"], mirror=True) + "; EXACT mirror of " + x["id"] + ": trades must mirror (side-symmetry witness)"
            mirrors.append(m)
    return leaves + mirrors


def fine_pivot_count(bars: pd.DataFrame, base_sd: float, mult: float = 2.0) -> int:
    """Fine ZigZag pivots: reversals of at least ``mult`` x base_sd x price (the fine-structure density witness)."""
    from app.short_horizon.expert_guided import oracle as OR
    h, l = bars["HIGH"].to_numpy(float), bars["LOW"].to_numpy(float)
    thr = float(mult * base_sd * np.nanmedian(bars["CLOSE"].to_numpy(float)))
    piv = OR.reversal_pivots(h, l, thr)[0]
    return int(len(piv))


def leaf_witness(bars: pd.DataFrame, base_sd: float, regime: str, vol: str, fine: str) -> dict:
    r = np.diff(np.log(bars["CLOSE"].to_numpy(float)))
    n_piv = fine_pivot_count(bars, base_sd)
    return {"drift_sign": float(np.sign(r.sum())), "expected_drift_sign": L1[regime], "sd_ratio": float(r.std() / base_sd), "expected_sd_mult": VOL[vol],
            "fine_pivots_per_100_bars": float(100.0 * n_piv / max(len(r), 1)), "fine": fine, "expected_fine_order": "low < medium < high"}


# ----------------------------------------------------------------------------- the hierarchy and the sampling design
def lhs(n: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.zeros((n, dims))
    for j in range(dims):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


KNOBS = {"drift": (-1.0, 1.0), "vol_mult": (0.6, 1.8), "fine_amp": (0.4, 2.0), "fine_dens": (0.5, 2.0), "quality": (0.55, 0.90),
         "cost_mult": (1.0, 3.0), "cap": (1, 4), "cutoff_h": (12.0, 24.0), "risk_pct": (0.005, 0.02)}


def catalog(n_lhs: int = 24, seed: int = 0) -> list:
    """Every leaf: id, level, generator config, expected behaviour, witness. L1 x L2 synthetic leaves at the operating
    quality, the full factorial L1 x L4 at the operating quality, and n_lhs Latin-hypercube leaves over the knobs."""
    leaves = []
    for reg in L1:
        for vol in VOL:
            for fine in L2:
                leaves.append({"id": f"L1L2_{reg}_{vol}_{fine}", "level": "L1xL2", "regime": reg, "vol": vol, "fine": fine, "variant": "base", "quality": OPERATING_QUALITY,
                               "exec": {}, "expected": f"openers follow the {reg} drift; closers pay in the {fine}-density fine swings; activity >= 2/wk",
                               "witness": "leaf_witness: drift sign, sd ratio, fine-pivot density"})
    for reg in L1:
        for vol in VOL:
            for v4 in L4_FACTORIAL:
                leaves.append({"id": f"L1L4_{reg}_{vol}_{v4}", "level": "L1xL4", "regime": reg, "vol": vol, "fine": "medium", "variant": v4, "quality": OPERATING_QUALITY,
                               "exec": {}, "expected": f"head-quality variant {v4} under {reg}/{vol}: the informative heads must still pay; uninformative ones must not open",
                               "witness": "simulated-head achieved quality per block (simulate_table)"})
    u = lhs(n_lhs, len(KNOBS), seed)
    for i in range(n_lhs):
        x = {k: lo + (hi - lo) * u[i, j] for j, (k, (lo, hi)) in enumerate(KNOBS.items())}
        reg = "trend_up" if x["drift"] > 0.33 else "trend_down" if x["drift"] < -0.33 else "range"
        vol = "high" if x["vol_mult"] > 1.15 else "low"
        fine = "high" if x["fine_amp"] > 1.4 else "low" if x["fine_amp"] < 0.7 else "medium"
        leaves.append({"id": f"LHS_{i:02d}", "level": "LHS", "regime": reg, "vol": vol, "fine": fine, "variant": "base", "quality": round(x["quality"], 3),
                       "exec": {"cost_mult": round(x["cost_mult"], 2), "cap": int(round(x["cap"])), "cutoff_h": round(x["cutoff_h"], 1), "risk_pct": round(x["risk_pct"], 4)},
                       "knobs": {k: round(v, 4) for k, v in x.items()}, "expected": "the robust region must keep this leaf's fitness within 5 % of its best",
                       "witness": "leaf_witness + the leaf's own fitness archive"})
    return leaves + catalog_transitions(n_lhs, seed + 1)


# ----------------------------------------------------------------------------- Stage A
def period_bars(ds, period) -> tuple:
    """(first bar, last bar) of a named period of the dataset (dates in the dataset's own contract) or a bar pair."""
    if not isinstance(period, str):
        return int(period[0]), int(period[1])
    a, b = ds.meta["periods"][period]
    idx = ds.bars.index
    return int(idx.searchsorted(pd.Timestamp(a, tz="UTC"))), int(idx.searchsorted(pd.Timestamp(b, tz="UTC")))


def leaf_world(cx_real, SCFG, AC, leaf: dict, calendar_period, seed: int, periods: tuple | None = None) -> dict:
    """A synthetic world for one leaf: synthetic bars on the real calendar of ``calendar_period`` (and the bars around
    it), its own engine truth and simulated heads at the leaf's quality; the strategy is scored on ``calendar_period``."""
    ds = cx_real.ds
    r = np.diff(np.log(ds.bars["CLOSE"].to_numpy(float)))
    base_sd = float(np.nanstd(r))
    meta = None
    absolute = getattr(SCFG.variant, "level_mode", "sigma") == "absolute_pips"
    from app.short_horizon.expert_guided import strategy_zz as _ZZ
    mtick = _ZZ.LEVEL_TICK if absolute else None                      # full_f4: the mirror's reflection constant on the level grid (sigma profiles: unchanged)
    if leaf.get("ms"):
        import re as _re
        import zlib
        base_id = _re.sub(r"^(M_|S_)+", "", leaf["id"])
        bars, meta = synthetic_path_ms(ds.bars.index, base_sd, leaf["ms"], (zlib.crc32(base_id.encode()) & 0xFFFF) + 7919 * int(seed), period_bars(ds, calendar_period), mirror=bool(leaf.get("mirror")), mirror_tick=mtick)
    elif leaf.get("legs"):
        import zlib
        import re as _re
        base_id = _re.sub(r"^(M_|S_)+", "", leaf["id"])                       # the twins share the base cell's generator seed: an EXACT mirror
        seed_v2 = (zlib.crc32(base_id.encode()) & 0xFFFF) + 7919 * int(seed)   # the run seed varies the path (twins run seed 0 = the base cell's seed 0)
        block = period_bars(ds, calendar_period)
        bars, meta = synthetic_path_v2(ds.bars.index, base_sd, tuple(leaf["legs"]), leaf["vol"], leaf["fine"], seed_v2, block, leaf.get("switch_fracs") or None,
                                       leaf.get("persistence"), mirror=bool(leaf.get("mirror")), strength=leaf.get("strength", "medium"), jump=leaf.get("jump"),
                                       strengths=leaf.get("strengths"), vol_frac=leaf.get("vol_frac"), mirror_tick=mtick)
    else:
        bars = synthetic_path(ds.bars.index, base_sd, leaf["regime"], leaf["vol"], leaf["fine"], seed)
    from tools.expert_guided_e1t import Ctx
    sigma, sigma_rule = Z.causal_sigma(bars), Z.causal_range_sigma(bars)
    ds2 = dataclasses.replace(ds, bars=bars, sigma=sigma, sigma_rule=sigma_rule)
    cx2 = Ctx(ds2)
    cx2.obs_ok = cx_real.obs_ok.copy()
    S2 = S.scfg_for("base", SCFG)
    ex = leaf.get("exec") or {}
    xv = S2.variant if S2.variant is not None else C3.FULL_F3
    if ex.get("cost_mult"):
        S2 = dataclasses.replace(S2, profile=S2.profile.scaled(float(ex["cost_mult"])))
    if ex.get("cap"):
        S2 = dataclasses.replace(S2, max_entries_per_day=int(ex["cap"]))
    if ex.get("spread") == "spike":
        S2 = S.scfg_for("spike", S2)
    elif ex.get("spread") == "x5":
        from tools.strategy_intervention import SpreadWindowProfile
        a_, b_ = period_bars(ds, calendar_period)
        mid = int(cx2.v[a_ + (b_ - a_) // 2])
        S2 = dataclasses.replace(S2, profile=SpreadWindowProfile(**dataclasses.asdict(S2.profile), window_t0_ns=mid, window_hours=24.0, factor=5.0))
    if ex.get("missing_runs"):
        from tools.strategy_intervention import missing_runs_mask
        cx2.obs_ok = cx2.obs_ok & missing_runs_mask(cx2, calendar_period if isinstance(calendar_period, str) else S.SEL_BLOCK, 8, 6, 24, seed)
    if ex.get("cutoff_h") or ex.get("risk_pct"):
        S2 = dataclasses.replace(S2, variant=dataclasses.replace(xv, entry_cutoff_hours_before_flat=float(ex.get("cutoff_h", xv.entry_cutoff_hours_before_flat or 24.0)),
                                                                 risk_pct=float(ex.get("risk_pct", xv.risk_pct))))
    # the engine TRUTH (outcome heads per cell of the sigma grid), the exploration visits and the truth-based controls
    # run on the SIGMA-LEVEL engine whatever the execution profile: the cell grid is a property of the world; the
    # profile's level mode (absolute per-horizon levels) governs the strategy's own orders only (S6, 2026-10-29)
    # full_f4 (coordinator ruling 2026-10-29, supersedes the line above for the absolute-level profile): the truth is the
    # exact replay of the DECISION MENU (menu.py) on the PROFILE engine; the truth-based controls run on the profile too
    if getattr(S2.variant, "level_mode", "sigma") == "absolute_pips":
        from app.short_horizon.expert_guided import strategy_zz as ZZ
        S2_truth = S2
        truth = S.build_truth_menu(cx2, S2, "synthetic", AC, bars, ZZ.PARAMS, only_periods=periods if periods is not None else (S.CAL_BLOCK, S.SEL_BLOCK),
                                   mirror_exploration=bool(leaf.get("mirror")), swap_refs=bool(leaf.get("swap_refs")))
    else:
        S2_truth = S.truth_config(S2)
        truth = S.build_truth(cx2, S2_truth, "synthetic", AC["outcomes"]["cells"], AC, only_periods=periods if periods is not None else (S.CAL_BLOCK, S.SEL_BLOCK),
                              mirror_exploration=bool(leaf.get("mirror")))
    q = S.quality_for(leaf["variant"], float(leaf["quality"]))
    table, sim = S.simulate_table(cx2, S2_truth, truth, q, seed, side_swap=bool(leaf.get("mirror")))
    if meta is not None and meta.get("kind") == "multi_scale":
        wit = {"turning_points": {k: len(v) for k, v in meta["turning_points"].items()}, "burst_windows": len(meta["burst_windows"]), "jumps": len(meta["jump_bars"]),
               "legs": transition_witness(bars, base_sd, {**meta, "vol": "ms"})["legs"] if meta["switch_bars"] or meta["legs"] else []}
    else:
        wit = transition_witness(bars, base_sd, meta) if meta is not None else leaf_witness(bars, base_sd, leaf["regime"], leaf["vol"], leaf["fine"])
    return {"cx": cx2, "scfg": S2, "scfg_truth": S2_truth, "truth": truth, "table": table, "sim": sim, "witness": wit, "period": calendar_period, "meta": meta, "bars": bars}


def composite(c: dict, scales: dict) -> float:
    """The owner's composite: Sharpe + annualised return + orders/day, each divided by its scale (c_k = 1 / median |term|)."""
    if not (math.isfinite(c.get("sharpe", float("nan"))) and c.get("trades", 0) > 0):
        return float("nan")
    return float(c["sharpe"] / scales["sharpe"] + c["r_ann"] / scales["r_ann"] + c["opd"] / scales["opd"])


def cvar(x: np.ndarray, alpha: float = 0.10) -> float:
    x = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    if not len(x):
        return float("nan")
    k = max(1, int(math.ceil(alpha * len(x))))
    return float(x[:k].mean())


def _leaf_scorer(w, fam, bounds, cells, AC):
    """The per-leaf scoring closure (gene map frozen on the leaf's calibration block, per-bar cost, selection-block score)."""
    cx, SCFG, truth, table = w["cx"], w["scfg"], w["truth"], w["table"]

    cal_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == S.CAL_BLOCK))
    sel_rows = np.flatnonzero(cx.obs_ok & (cx.ds.period == S.SEL_BLOCK))
    view_rows = np.union1d(cal_rows, sel_rows)
    table["cost_row"] = S.row_costs(cx, SCFG, truth.cost - C3.cost_sigma_of(cx, SCFG, 0.0, cal_rows))   # per-bar cost BEFORE the view (the perbar gate)
    view = C2.restrict(table, view_rows)
    ref = C3.closer_reference_sides(table, S.closer_src_of(truth))
    sig = np.asarray(cx.ds.sigma, float)
    min_opw = float(AC["strategy"]["calibration_min_trades_per_week"])
    if fam is None:
        gm = C3.gene_map(view, cal_rows, ref, cells, truth.e_time, truth.cost)
        make = lambda g, _gm=gm: C3.StrategyC3.from_genes(_gm.to_absolute(g), "nop_veto" in g)  # noqa: E731
        Pol, roll = S.PolicyC3PerBar, ACT.rollout_with_trades
    else:
        gm = SF.gene_map_full(view, cal_rows, ref, cells, truth.e_time, truth.cost, bounds)
        make = lambda g, _gm=gm: SF.StrategyFull.from_genes(_gm.to_absolute(g), "nop_veto" in g, fam[0], fam[1])  # noqa: E731
        Pol, roll = S.policy_factory(S.PolicyFullPerBar, SCFG), SF.rollout_with_volume
    score = S.score_fn(cx, SCFG, cells, truth.e_time, truth.cost, AC, S.SEL_BLOCK, sig, min_opw, Pol, roll)
    return (score, view, make)

def stage_a(worlds: list, bounds: dict, family: str, AC, seed: int, population: int, generations: int, out: Path) -> dict:
    """NSGA-II over every gene with two objectives: the MEAN composite over the leaves and its CVaR-10 % (worst decile),
    under Deb's constraint domination on the leaf-average activity (CI lower bound >= 2/wk) and side share (<= 0.90);
    every evaluated point is archived with its per-leaf components."""
    from deap import base, creator, tools
    import random
    fam = S.FAMILIES[family]
    names = list(bounds)
    lo = [float(bounds[g][0]) for g in names]
    hi = [float(bounds[g][1]) for g in names]
    cells = [tuple(c) for c in AC["outcomes"]["cells"]]
    # per-leaf scoring closures (gene maps frozen on each leaf's calibration block)
    scorers = []
    kept, excluded = [], []
    for w in worlds:
        try:
            scorers.append(_leaf_scorer(w, fam, bounds, cells, AC))
            kept.append(w)
        except (SystemExit, ValueError, KeyError) as e:          # a leaf whose calibration block cannot freeze the gene map (e.g. no short-side
            w["excluded"] = str(e)                                # reference visits in a one-sided synthetic world) is excluded and recorded
            excluded.append({"id": w.get("leaf", {}).get("id"), "reason": str(e)})
        w.pop("truth", None); w.pop("table", None); w.pop("sim", None)      # memory: the scorer holds the restricted view only (78 leaves OOM-killed a 3G cap)
    worlds = kept
    if not scorers:
        raise SystemExit("STAGE_A_NO_FEASIBLE_LEAF")
    import gc
    gc.collect()
    archive = []
    scales = {"sharpe": 1.0, "r_ann": 0.2, "opd": 0.6}     # provisional; recalibrated from the first generation (declared rule c_k = 1/median|term|)

    global _SCORERS
    _SCORERS = scorers
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(min(int(a_workers()), len(scorers))) if len(scorers) > 1 else None

    def evaluate_point(g: dict) -> dict:
        if pool is None:
            comps = [score(view, make(g)) for score, view, make in scorers]
        else:
            comps = pool.map(_score_leaf, [(i, g) for i in range(len(scorers))])
        fit = np.array([composite(c, scales) for c in comps])
        opw_lb = np.array([c.get("opw_ci_lb", c["opw"]) if math.isfinite(c.get("opw_ci_lb", float("nan"))) else 0.0 for c in comps])
        side = np.array([c.get("side_max_share", 0.0) if math.isfinite(c.get("side_max_share", float("nan"))) else 0.0 for c in comps])
        agent = np.array([c.get("agent_closes", 0) for c in comps])
        mean_fit = float(np.nanmean(fit)) if np.isfinite(fit).any() else float("nan")
        cv = cvar(fit)
        viol = max(0.0, 2.0 - float(opw_lb.mean())) + max(0.0, float(side.mean()) - 0.90) + (0.0 if agent.sum() > 0 else 1.0) + (1.0 if not math.isfinite(mean_fit) else 0.0)
        return {"genes": g, "mean": mean_fit, "cvar10": cv, "valid": viol == 0.0, "violation": viol,
                "per_leaf": [{k: c.get(k) for k in ("sharpe", "r_ann", "opd", "opw", "opw_ci_lb", "side_max_share", "agent_closes", "trades", "dd", "lcb")} for c in comps]}

    if not hasattr(creator, "StageAFit"):
        creator.create("StageAFit", base.Fitness, weights=(1.0, 1.0))
        creator.create("StageAInd", list, fitness=creator.StageAFit)
    state = random.getstate()
    random.seed(int(seed))
    tb = base.Toolbox()
    tb.register("ind", lambda: creator.StageAInd([random.uniform(a, b) for a, b in zip(lo, hi)]))
    tb.register("pop", tools.initRepeat, list, tb.ind)
    tb.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=15.0)
    tb.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=20.0, indpb=1.0 / len(names))
    tb.register("select", tools.selNSGA2)
    BIG = 1e6

    def objectives(r):
        if r["valid"]:
            return (r["mean"], r["cvar10"])
        return (-BIG - r["violation"], -BIG - r["violation"])

    cache = {}

    def ev(ind):
        key = tuple(round(float(x), 6) for x in ind)
        if key not in cache:
            r = evaluate_point(dict(zip(names, key)))
            r["generation"] = gen[0]
            cache[key] = r
            archive.append(r)
        return objectives(cache[key])
    gen = [0]
    pop = tb.pop(n=population)
    for ind in pop:
        ind.fitness.values = ev(ind)
    # recalibrate the composite scales on the first generation's valid points (c_k = 1 / median |term|)
    terms = {"sharpe": [], "r_ann": [], "opd": []}
    for r in archive:
        for c in r["per_leaf"]:
            if c["sharpe"] is not None and math.isfinite(c["sharpe"]):
                for k in terms:
                    terms[k].append(abs(float(c[k])))
    for k in terms:
        if terms[k] and float(np.median(terms[k])) > 1e-6:
            scales[k] = float(np.median(terms[k]))
    cache.clear()
    archive.clear()
    for ind in pop:
        ind.fitness.values = ev(ind)
    pop = tb.select(pop, len(pop))
    for g_ in range(1, generations + 1):
        gen[0] = g_
        off = tools.selTournamentDCD(pop, len(pop))
        off = [creator.StageAInd(list(o)) for o in off]
        for i in range(0, len(off) - 1, 2):
            if random.random() < 0.9:
                tb.mate(off[i], off[i + 1])
            tb.mutate(off[i])
            tb.mutate(off[i + 1])
        for ind in off:
            ind.fitness.values = ev(ind)
        pop = tb.select(pop + off, population)
        print(json.dumps({"stage_a": family, "gen": g_, "evals": len(archive), "best_mean": max((r["mean"] for r in archive if r["valid"] and math.isfinite(r["mean"])), default=None)}), flush=True)
    random.setstate(state)
    if pool is not None:
        pool.close()
    valid = [r for r in archive if r["valid"] and math.isfinite(r["mean"]) and math.isfinite(r["cvar10"])]
    pareto = _pareto(valid)
    region = robust_region_escalated(valid, worlds, names)
    doc = {"family": family, "genes": names, "bounds": bounds, "population": population, "generations": generations, "seed": seed, "scales": scales,
           "leaves": len(worlds), "leaves_excluded": excluded, "evaluations": len(archive), "valid_points": len(valid), "pareto": pareto, "robust_region": region,
           "best_mean": max(valid, key=lambda r: r["mean"]) if valid else None, "best_cvar": max(valid, key=lambda r: r["cvar10"]) if valid else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"STAGE_A_{family}.json").write_text(json.dumps(doc, indent=1, default=float))
    import gzip
    with gzip.open(out / f"STAGE_A_{family}_archive.json.gz", "wt") as fh:
        json.dump(archive, fh, default=float)
    return doc


_SCORERS = []


def a_workers() -> int:
    import os
    return int(os.environ.get("STAGE_A_WORKERS", "8"))


def _score_leaf(args):
    i, g = args
    score, view, make = _SCORERS[i]
    return score(view, make(g))


def _pareto(points: list) -> list:
    front = []
    for p in points:
        dom = any((q["mean"] >= p["mean"] and q["cvar10"] >= p["cvar10"] and (q["mean"] > p["mean"] or q["cvar10"] > p["cvar10"])) for q in points)
        if not dom:
            front.append({"genes": p["genes"], "mean": p["mean"], "cvar10": p["cvar10"]})
    return sorted(front, key=lambda r: -r["mean"])


TOLERANCES = (0.05, 0.10, 0.20, 0.30, 0.50)
MIN_ROBUST_POINTS = 4


def robust_region_escalated(valid: list, worlds: list, names: list) -> dict:
    """The owner's 5 % rule first; when it admits fewer than MIN_ROBUST_POINTS points (regimes trade off against each
    other: no point is near-best in trend-up AND trend-down AND range at once) the tolerance is widened along
    TOLERANCES until the region holds at least that many points. The tolerance used and the 5 % result are recorded."""
    first = robust_region(valid, worlds, names, TOLERANCES[0])
    for tol in TOLERANCES:
        r = robust_region(valid, worlds, names, tol)
        if r["points"] >= MIN_ROBUST_POINTS:
            r["tolerance_used"] = tol
            r["at_5pct"] = {"points": first["points"], "box": first["box"]}
            return r
    r = robust_region(valid, worlds, names, TOLERANCES[-1])
    r["tolerance_used"] = TOLERANCES[-1]
    r["at_5pct"] = {"points": first["points"], "box": first["box"]}
    return r


def robust_region(valid: list, worlds: list, names: list, tol: float = 0.05) -> dict:
    """Points within tol of the best mean fitness overall AND within tol of the best per-L1-regime mean fitness in every
    regime -> the box (min / max per gene) they span, with the count."""
    if not valid:
        return {"points": 0, "box": {}, "tolerance": tol, "rule": "within tol of the best mean overall and in every L1 regime"}
    regimes = {}
    for i, w in enumerate(worlds):
        regimes.setdefault(w["leaf"]["regime"], []).append(i)
    best = max(r["mean"] for r in valid)
    per_reg_best = {}
    for reg, idx in regimes.items():
        per_reg_best[reg] = max(float(np.nanmean([composite_from(r["per_leaf"][i]) for i in idx])) for r in valid)
    keep = []
    for r in valid:
        if r["mean"] < best - tol * abs(best):
            continue
        ok = True
        for reg, idx in regimes.items():
            m = float(np.nanmean([composite_from(r["per_leaf"][i]) for i in idx]))
            if m < per_reg_best[reg] - tol * abs(per_reg_best[reg]):
                ok = False
                break
        if ok:
            keep.append(r)
    box = {g: [min(r["genes"][g] for r in keep), max(r["genes"][g] for r in keep)] for g in names} if keep else {}
    return {"points": len(keep), "box": box, "per_regime_best": per_reg_best, "best_mean": best, "tolerance": tol,
            "rule": f"within {tol:.0%} of the best mean overall and within {tol:.0%} of the best mean in every L1 regime", "members": [r["genes"] for r in keep[:50]]}


_SCALES = {"sharpe": 1.0, "r_ann": 0.2, "opd": 0.6}


def composite_from(c: dict, scales: dict | None = None) -> float:
    s_ = scales or _SCALES
    if c.get("sharpe") is None or not math.isfinite(c["sharpe"]) or not c.get("trades"):
        return float("nan")
    return float(c["sharpe"] / s_["sharpe"] + c["r_ann"] / s_["r_ann"] + c["opd"] / s_["opd"])


# ----------------------------------------------------------------------------- CLI
def cmd_catalog(a):
    leaves = catalog(a.n_lhs, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(leaves, indent=1))
    print(json.dumps({"leaves": len(leaves), "levels": {k: sum(1 for x in leaves if x["level"] == k) for k in ("L1xL2", "L1xL4", "LHS")}}))


def cmd_stage_a(a):
    AC = S.contract_for_sim(a.contract)
    cx, SCFG = S.load_fold(a.fold, a.admitted, a.csv_dir)
    SCFG = S.with_variant(SCFG, C3.FULL_F3)
    leaves = json.loads(Path(a.catalog).read_text())
    if a.levels:
        leaves = [x for x in leaves if x["level"] in a.levels.split(",")]
    if a.max_leaves:
        leaves = leaves[:a.max_leaves]
    worlds = []
    t0 = time.time()
    for i, leaf in enumerate(leaves):
        w = leaf_world(cx, SCFG, AC, leaf, S.SEL_BLOCK, a.seed * 1000 + i)
        w["leaf"] = leaf
        w["achieved"] = w["sim"]["achieved"]
        worlds.append(w)
        print(json.dumps({"world": leaf["id"], "witness": w["witness"], "sec": round(time.time() - t0)}), flush=True)
    fam = S.FAMILIES[a.family]
    bounds = SF.gene_bounds(*fam) if fam else {k: list(v) for k, v in C3.Q_BOUNDS.items()}
    doc = stage_a(worlds, bounds, a.family, AC, a.seed, a.population, a.generations, Path(a.out))
    doc_leaves = [{"id": w["leaf"]["id"], "witness": w["witness"], "achieved": w["achieved"], "excluded": w.get("excluded")} for w in worlds]
    (Path(a.out) / f"STAGE_A_{a.family}_leaves.json").write_text(json.dumps(doc_leaves, indent=1, default=float))
    print(json.dumps({"family": a.family, "leaves": len(worlds), "evaluations": doc["evaluations"], "valid": doc["valid_points"], "pareto": len(doc["pareto"]),
                      "robust_points": doc["robust_region"]["points"], "best_mean": (doc["best_mean"] or {}).get("mean")}))


def cmd_robust_region(a):
    """Recompute the robust region of a finished Stage A from its archive (per-leaf components) and the catalog."""
    import gzip
    out = Path(a.out)
    doc = json.loads((out / f"STAGE_A_{a.family}.json").read_text())
    pts = json.loads(gzip.open(out / f"STAGE_A_{a.family}_archive.json.gz", "rt").read())
    pts = pts if isinstance(pts, list) else pts.get("archive", pts)
    leaves = json.loads((out / f"STAGE_A_{a.family}_leaves.json").read_text())
    cat = {x["id"]: x for x in json.loads(Path(a.catalog).read_text())}
    worlds = [{"leaf": cat[x["id"]]} for x in leaves if not x.get("excluded")]
    valid = [p for p in pts if p.get("valid") and p["mean"] is not None and np.isfinite(p["mean"])]
    region = robust_region_escalated(valid, worlds, doc["genes"])
    doc["robust_region_previous"] = doc.get("robust_region")
    doc["robust_region"] = region
    (out / f"STAGE_A_{a.family}.json").write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"family": a.family, "points": region["points"], "tolerance_used": region.get("tolerance_used"), "at_5pct": region.get("at_5pct", {}).get("points"), "box": region["box"]}))


def stop_requested(out_path) -> bool:
    """Owner STOP flag (2026-10-27): a file named STOP in the work root (any ancestor of the output path, three levels up)
    makes every NEW job of the lane exit before doing anything; running jobs finish their current file."""
    from pathlib import Path as _P
    p = _P(str(out_path)).resolve()
    for q in [p] + list(p.parents)[:3]:
        if (q / "STOP").exists():
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog")
    c.add_argument("--n-lhs", type=int, default=24)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    s = sub.add_parser("stage-a")
    s.add_argument("--fold", default="wf2022")
    s.add_argument("--admitted", required=True)
    s.add_argument("--csv-dir", required=True)
    s.add_argument("--contract", default=str(S.CONTRACT))
    s.add_argument("--catalog", required=True)
    s.add_argument("--levels", default="")
    s.add_argument("--max-leaves", type=int, default=0)
    s.add_argument("--family", default="c3", choices=list(S.FAMILIES))
    s.add_argument("--population", type=int, default=32)
    s.add_argument("--generations", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("robust-region", help="recompute the robust region of a finished Stage A from its archive")
    r.add_argument("--family", default="c3", choices=list(S.FAMILIES))
    r.add_argument("--catalog", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if stop_requested(a.out):
        print("STOP flag present in the work root: nothing started (owner order 2026-10-27)")
        raise SystemExit(75)
    {"catalog": cmd_catalog, "stage-a": cmd_stage_a, "robust-region": cmd_robust_region}[a.cmd](a)


if __name__ == "__main__":
    main()
