"""Every strategy coefficient and parameter lives in ONE JSON file (owner rule 2026-10-27): the strategy reads them,
nothing is hard-coded, every key has declared bounds and a default, unknown keys and out-of-bounds values are refused,
and DEAP tunes the same keys later -- always re-validating on the synthetic behaviour-matrix cells.

Keys (quantile genes are mapped to absolute thresholds on the calibration block by the frozen gene map):
  open_rate_l, open_rate_s        open-margin quantiles per side (C3 b27781e ties them: ONE pooled margin, so they must be equal)
  nop_veto                        ABSOLUTE threshold on calibrated P(NOP) (null = no veto)
  close_rem_rate_l/_s             per-side early-close quantiles on the remaining value
  close_p_l/_s                    per-side ABSOLUTE thresholds on calibrated P(SL first) (a quantile over a saturated
                                  reference mapped to 1e-6 under perfect closers: behaviour matrix 2026-10-27)
  allow_reverse                   reversal permission (>= 0.5 = allowed)
  observe_bars                    best-of-day observation window (0 = first qualifying signal)
  tp_a_l, tp_b_l, sl_c_l, sl_e_l, tp_a_s, ... parametric TP / SL coefficients per side (family 'param')
  v_alpha_l, v_beta_l, v_gamma_l, v_delta_l, ..._s, v_min   parametric volume coefficients per side (family 'param')
  risk_pct                        risk per trade of the risk-sizing family
  entry_cutoff_hours              Friday cutoff window (hours before the forced flat)
  max_entries_per_day
  rr_margin_sigma                 TP < SL cells allowed only when net > cost + this
  confirmation_lag_bars           declared coarse-switch confirmation lag
  open_margin_eps                 an open needs EV > margin + eps (sigma): removes float knife-edges on the quantile sample
  closer_mode                     C3Z: 1 = close when the advantage head says P(t) - V_hat(t) >= -closer_adv_margin; 0 = the legacy closer rules
  closer_adv_margin               C3Z: the advantage close margin (sigma units of the trade)
  tp_mode, k_tp_protect           C3Z: 0 = the cell's TP; 1 = protective TP = max(cell TP, k x predicted MFE) (broker TP / SL always on the order)
  forced_every_bars               the forced-entry exit audit: one forced open per this many bars, sides alternating (not a trading coefficient)
  zz_post_flip_bars               C3Z: for this many bars after the coarse direction flips, an open against it needs a fresh fine pivot (no strong-net exception)
  zz_counter_net_frac             C3Z: an open against the coarse leg passes the room gate when its predicted net >= frac x TP
  ev_from_net                     C3Z: rank the sides by the chosen cell's predicted engine net (1) or by the p3 x cell EV (0)
  close_margin_floor              C3Z: the remaining-value close margin is never below this (a negative mapped margin closed positions worth holding)
  risk_reward_floor               C3Z: a cell is allowed only when its predicted net >= floor x its predicted MAE
  burst_window_bars, burst_vol_ratio   C3Z: no open while realized vol over the window (24 h scale) > ratio x causal sigma
  zz_room_tp_mult                 C3Z: the coarse ZigZag room must pay mult x the cell's TP
  zz_fine_h, zz_timing_gate       C3Z: no open with a fine counter-pivot against the side within h bars (gate on/off)
  zz_nop_gate, zz_theta_f         C3Z: no open on a ladder NOP-hard tick (no level with f_open >= theta_f)
  zz_close_room                   C3Z: close early on a fine alarm when the coarse room is below this (sigma)
  close_p_requires_rem_below      C3Z: the P(SL first) close fires only when the predicted remaining value is below this
  zz_reverse_flip_bars            C3Z: a reversal only within this many bars of a coarse direction flip (else a plain close)
  z_close, k_persist              C3Z: close when A_hat - z_close x s_hat >= -closer_adv_margin on k_persist consecutive executable ticks
  tp_grid, sl_grid                the TP / SL cell grid (sigma multiples)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BOUNDS = {
    "open_rate_l": [0.002, 0.50], "open_rate_s": [0.002, 0.50], "nop_veto": [0.30, 0.95],
    "close_rem_rate_l": [0.0, 0.60], "close_rem_rate_s": [0.0, 0.60], "close_p_l": [0.30, 0.95], "close_p_s": [0.30, 0.95],
    "allow_reverse": [0.0, 1.0], "observe_bars": [0, 3],
    "tp_a_l": [0.25, 2.0], "tp_b_l": [-1.5, 1.5], "sl_c_l": [0.5, 2.5], "sl_e_l": [-1.5, 1.5],
    "tp_a_s": [0.25, 2.0], "tp_b_s": [-1.5, 1.5], "sl_c_s": [0.5, 2.5], "sl_e_s": [-1.5, 1.5],
    "v_alpha_l": [0.0, 1.0], "v_beta_l": [-1.0, 1.0], "v_gamma_l": [0.0, 1.0], "v_delta_l": [0.0, 1.0],
    "v_alpha_s": [0.0, 1.0], "v_beta_s": [-1.0, 1.0], "v_gamma_s": [0.0, 1.0], "v_delta_s": [0.0, 1.0], "v_min": [0.0, 0.5],
    "risk_pct": [0.0025, 0.03], "entry_cutoff_hours": [4.0, 48.0], "max_entries_per_day": [1, 6], "rr_margin_sigma": [0.0, 0.5],
    "confirmation_lag_bars": [24, 480], "open_margin_eps": [0.0, 1e-3],
    "closer_mode": [0.0, 1.0], "closer_adv_margin": [0.0, 1.0], "tp_mode": [0.0, 1.0], "k_tp_protect": [1.0, 4.0], "forced_every_bars": [6, 96], "zz_post_flip_bars": [0, 960], "zz_counter_net_frac": [0.0, 2.0], "ev_from_net": [0.0, 1.0], "close_margin_floor": [0.0, 2.0], "risk_reward_floor": [0.0, 3.0], "burst_window_bars": [2, 48], "burst_vol_ratio": [1.2, 6.0], "zz_room_tp_mult": [0.0, 3.0], "zz_fine_h": [1, 48],
    "zz_timing_gate": [0.0, 1.0], "zz_nop_gate": [0.0, 1.0], "zz_close_room": [-2.0, 5.0], "close_p_requires_rem_below": [-1.0, 1.0], "zz_theta_f": [0.0, 1.0], "zz_reverse_flip_bars": [0, 960], "z_close": [0.0, 3.0], "k_persist": [1, 6], "closer_filter": [0.0, 1.0],
    # v2 (business contract, 2026-10-29): the absolute-level genes with Musashi's proposed search bounds, the risk budget rho,
    # and the two declared categorical options (reference quantile, horizon rule)
    "a_sl": [0.5, 2.0], "b_sl": [0.0, 1.0], "buffer_pips": [0.0, 5.0], "a_tp": [0.5, 2.0], "b_tp": [0.0, 1.0], "rho": [0.001, 0.01],
    "ref_p75": [0.0, 1.0], "horizon_rule": [0.0, 1.0],
    # v2.1 (coordinator ruling 2026-10-29): the hierarchy vetoes act only on low-confidence slots, EV / sd_pred < z_veto (full_f4)
    "z_veto": [0.0, 4.0],
}
# v2.2 (coordinator ruling A on tier 2 of 443b7a1, 2026-10-29): PER-PROFILE bounds. BOUNDS above is the full_f3 table (the
# audited lane-F3 profile, byte-identical, pinned by tests/test_strategy_params_v22.py); full_f4 selects BOUNDS_F4, which widens
# the pooled open rate to [0.002, 1.0]: the open-margin gate was the first diverging decision against the truth oracle in 69 of
# the 76 W1_side / W8_old_direction_after_lag / W0_activity failures (evidence/tier2_full_f4_menu_443b7a1/FIRST_DIVERGENCE_SUMMARY.txt);
# under the menu the margin is the (1 - open_rate) quantile of the best allowed predicted net, floored at 0 (strategy_zz.net_open_margin),
# so open_rate = 1.0 puts the margin at the reference's minimum (the 0 quantile), floored at 0
# (ruling B, same review): burst_vol_ratio's full_f4 search range is [1.2, 2.5] and its full_f4 canonical value 1.5 (DEFAULTS_F4)
BOUNDS_F4 = {**BOUNDS, "open_rate_l": [0.002, 1.0], "open_rate_s": [0.002, 1.0], "burst_vol_ratio": [1.2, 2.5]}
BOUNDS_BY_PROFILE = {"full_f3": BOUNDS, "full_f4": BOUNDS_F4}
LIST_KEYS = {"tp_grid": [0.25, 4.0], "sl_grid": [0.25, 4.0]}
META_KEYS = {"schema", "note", "defaults_are", "sealed", "exec_profile"}
EXEC_PROFILES = ("full_f3", "full_f4")       # exec_profile selects the execution profile (c3.EXEC_VARIANTS); absent = full_f3 (the audited lane-F3 default, byte-identical)
DEFAULTS = {
    "open_rate_l": 0.45, "open_rate_s": 0.45, "nop_veto": 0.50,
    "close_rem_rate_l": 0.30, "close_rem_rate_s": 0.30, "close_p_l": 0.50, "close_p_s": 0.50,
    "allow_reverse": 0.0, "observe_bars": 0,
    "tp_a_l": 1.0, "tp_b_l": 0.0, "sl_c_l": 1.0, "sl_e_l": 0.0, "tp_a_s": 1.0, "tp_b_s": 0.0, "sl_c_s": 1.0, "sl_e_s": 0.0,
    "v_alpha_l": 1.0, "v_beta_l": 0.0, "v_gamma_l": 0.0, "v_delta_l": 0.0, "v_alpha_s": 1.0, "v_beta_s": 0.0, "v_gamma_s": 0.0, "v_delta_s": 0.0, "v_min": 0.0,
    "risk_pct": 0.01, "entry_cutoff_hours": 24.0, "max_entries_per_day": 4, "rr_margin_sigma": 0.10, "confirmation_lag_bars": 120,
    "tp_grid": [0.5, 0.75, 1.0, 1.5, 2.0], "sl_grid": [0.5, 0.75, 1.0, 1.5, 2.0], "open_margin_eps": 1e-6,
    "closer_mode": 1.0, "closer_adv_margin": 0.02, "tp_mode": 1.0, "k_tp_protect": 1.5, "forced_every_bars": 24, "zz_post_flip_bars": 240, "zz_counter_net_frac": 1.2, "ev_from_net": 1.0, "close_margin_floor": 0.0, "risk_reward_floor": 0.5, "burst_window_bars": 6, "burst_vol_ratio": 2.0, "zz_room_tp_mult": 1.75, "zz_fine_h": 6,
    "zz_timing_gate": 1.0, "zz_nop_gate": 1.0, "zz_close_room": 0.5, "close_p_requires_rem_below": 0.0, "zz_theta_f": 0.5, "zz_reverse_flip_bars": 960, "z_close": 1.0, "k_persist": 1, "closer_filter": 1.0,
    "a_sl": 1.0, "b_sl": 0.0, "buffer_pips": 1.0, "a_tp": 1.0, "b_tp": 0.0, "rho": 0.005, "ref_p75": 0.0, "horizon_rule": 0.0, "z_veto": 2.0,
}
DEFAULT_EXEC_PROFILE = "full_f4"             # the v2 canonical point runs the shared business contract
# v2.2 (coordinator ruling B on tier 2 of 443b7a1, 2026-10-29): PER-PROFILE defaults. DEFAULTS above is the full_f3 point
# (byte-identical); full_f4 lowers the causal burst gate's ratio to 1.5 x sigma: the 26 opens inside planted burst windows
# (W17, 23 records) came 3-10 bars into the window at realized vol / sigma 0.49-1.98, every one truth-positive
# (evidence/tier2_full_f4_menu_443b7a1 + the instrument check in the v2.2 evidence folder)
DEFAULTS_F4 = {**DEFAULTS, "burst_vol_ratio": 1.5}
DEFAULTS_BY_PROFILE = {"full_f3": DEFAULTS, "full_f4": DEFAULTS_F4}


class StrategyParamsError(ValueError):
    pass


def validate(doc: dict) -> dict:
    """Refuses an unknown key, a missing key, an out-of-bounds value or a tied pair that differs."""
    if "exec_profile" in doc and doc["exec_profile"] not in EXEC_PROFILES:
        raise StrategyParamsError(f"STRATEGY_PARAMS_EXEC_PROFILE_UNKNOWN:{doc['exec_profile']}")
    bounds = bounds_of(exec_profile_of(doc))
    unknown = sorted(set(doc) - set(BOUNDS) - set(LIST_KEYS) - META_KEYS)
    if unknown:
        raise StrategyParamsError(f"STRATEGY_PARAMS_UNKNOWN_KEY:{unknown}")
    missing = sorted((set(BOUNDS) | set(LIST_KEYS)) - set(doc))
    if missing:
        raise StrategyParamsError(f"STRATEGY_PARAMS_MISSING_KEY:{missing}")
    for k, (lo, hi) in bounds.items():
        v = doc[k]
        if v is None and k == "nop_veto":
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= float(v) <= hi):
            raise StrategyParamsError(f"STRATEGY_PARAMS_OUT_OF_BOUNDS:{k}={v} not in [{lo}, {hi}]")
    for k, (lo, hi) in LIST_KEYS.items():
        v = doc[k]
        if not isinstance(v, list) or not v or any((not isinstance(x, (int, float))) or not (lo <= float(x) <= hi) for x in v):
            raise StrategyParamsError(f"STRATEGY_PARAMS_OUT_OF_BOUNDS:{k}={v}")
    if abs(float(doc["open_rate_l"]) - float(doc["open_rate_s"])) > 1e-12:
        raise StrategyParamsError("STRATEGY_PARAMS_TIED_KEYS_DIFFER:open_rate_l != open_rate_s (C3 b27781e pools the open margin)")
    return doc


def exec_profile_of(doc: dict) -> str:
    return str(doc.get("exec_profile", "full_f3"))


def bounds_of(profile: str) -> dict:
    """The bounds table of an execution profile (v2.2): full_f3 = BOUNDS (audited, byte-identical), full_f4 = BOUNDS_F4."""
    if profile not in BOUNDS_BY_PROFILE:
        raise StrategyParamsError(f"STRATEGY_PARAMS_EXEC_PROFILE_UNKNOWN:{profile}")
    return BOUNDS_BY_PROFILE[profile]


def load(path) -> dict:
    return validate(json.loads(Path(path).read_text()))


def default_doc(profile: str = DEFAULT_EXEC_PROFILE) -> dict:
    """The canonical parameter document of a profile (v2.2: per-profile defaults and bounds; the committed file is the full_f4 one)."""
    if profile not in DEFAULTS_BY_PROFILE:
        raise StrategyParamsError(f"STRATEGY_PARAMS_EXEC_PROFILE_UNKNOWN:{profile}")
    return {"schema": "strategy_params.v2", "note": "every strategy coefficient; defaults = the canonical point; bounds in app/short_horizon/expert_guided/strategy_params.py (BOUNDS); v2 (2026-10-29) adds the business-contract genes a_sl / b_sl / buffer_pips / a_tp / b_tp / rho and the declared options ref_p75 / horizon_rule; v2.1 (2026-10-29) adds z_veto (the hierarchy vetoes bind only on low-confidence slots, full_f4); v2.2 (2026-10-29, tier-2 rulings A / B) makes the bounds and defaults PER PROFILE: full_f4 widens open_rate to [0.002, 1.0], bounds burst_vol_ratio to [1.2, 2.5] with the canonical value 1.5 (BOUNDS_F4 / DEFAULTS_F4); full_f3 keeps BOUNDS / DEFAULTS byte-identical; exec_profile selects the execution profile (full_f3 = the audited lane-F3 profile, full_f4 = the shared business contract)",
            "defaults_are": "canonical", "exec_profile": profile, **DEFAULTS_BY_PROFILE[profile]}


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def genes_c3(doc: dict) -> dict:
    """The C3 quantile gene vector from the parameter file (pooled open_rate as C3 declares it)."""
    g = {"open_rate": float(doc["open_rate_l"]), "close_rem_rate_l": float(doc["close_rem_rate_l"]), "close_rem_rate_s": float(doc["close_rem_rate_s"]),
         "close_p_l": float(doc["close_p_l"]), "close_p_s": float(doc["close_p_s"]), "allow_reverse": float(doc["allow_reverse"])}
    if doc.get("nop_veto") is not None:
        g["nop_veto"] = float(doc["nop_veto"])
    return g


def genes_full(doc: dict, tp_family: str, vol_family: str) -> dict:
    g = dict(genes_c3(doc))
    g["observe_bars"] = float(doc["observe_bars"])
    if tp_family == "param":
        for s in ("l", "s"):
            for k in ("tp_a", "tp_b", "sl_c", "sl_e"):
                g[f"{k}_{s}"] = float(doc[f"{k}_{s}"])
    if vol_family == "param":
        for s in ("l", "s"):
            for k in ("v_alpha", "v_beta", "v_gamma", "v_delta"):
                g[f"{k}_{s}"] = float(doc[f"{k}_{s}"])
        g["v_min"] = float(doc["v_min"])
    return g
