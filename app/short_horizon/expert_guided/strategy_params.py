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
}
LIST_KEYS = {"tp_grid": [0.25, 4.0], "sl_grid": [0.25, 4.0]}
META_KEYS = {"schema", "note", "defaults_are", "sealed"}
DEFAULTS = {
    "open_rate_l": 0.251, "open_rate_s": 0.251, "nop_veto": 0.50,
    "close_rem_rate_l": 0.30, "close_rem_rate_s": 0.30, "close_p_l": 0.50, "close_p_s": 0.50,
    "allow_reverse": 1.0, "observe_bars": 0,
    "tp_a_l": 1.0, "tp_b_l": 0.0, "sl_c_l": 1.0, "sl_e_l": 0.0, "tp_a_s": 1.0, "tp_b_s": 0.0, "sl_c_s": 1.0, "sl_e_s": 0.0,
    "v_alpha_l": 1.0, "v_beta_l": 0.0, "v_gamma_l": 0.0, "v_delta_l": 0.0, "v_alpha_s": 1.0, "v_beta_s": 0.0, "v_gamma_s": 0.0, "v_delta_s": 0.0, "v_min": 0.0,
    "risk_pct": 0.01, "entry_cutoff_hours": 24.0, "max_entries_per_day": 4, "rr_margin_sigma": 0.10, "confirmation_lag_bars": 120,
    "tp_grid": [0.5, 0.75, 1.0, 1.5, 2.0], "sl_grid": [0.5, 0.75, 1.0, 1.5, 2.0], "open_margin_eps": 1e-6,
}


class StrategyParamsError(ValueError):
    pass


def validate(doc: dict) -> dict:
    """Refuses an unknown key, a missing key, an out-of-bounds value or a tied pair that differs."""
    unknown = sorted(set(doc) - set(BOUNDS) - set(LIST_KEYS) - META_KEYS)
    if unknown:
        raise StrategyParamsError(f"STRATEGY_PARAMS_UNKNOWN_KEY:{unknown}")
    missing = sorted((set(BOUNDS) | set(LIST_KEYS)) - set(doc))
    if missing:
        raise StrategyParamsError(f"STRATEGY_PARAMS_MISSING_KEY:{missing}")
    for k, (lo, hi) in BOUNDS.items():
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


def load(path) -> dict:
    return validate(json.loads(Path(path).read_text()))


def default_doc() -> dict:
    return {"schema": "strategy_params.v1", "note": "every strategy coefficient; defaults = the canonical point; bounds in app/short_horizon/expert_guided/strategy_params.py (BOUNDS)",
            "defaults_are": "canonical", **DEFAULTS}


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
