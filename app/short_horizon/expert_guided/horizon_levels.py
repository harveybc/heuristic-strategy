"""Absolute-level order construction per horizon (S2 of the shared business contract, owner decision 2026-10-29).

A re-implementation, not a copy, of the semantics Musashi published in agent-multi 82862801 app/horizon_risk_rules.py
(the cross-lane parity test replays his CASES through these functions with HIS synthetic constants injected):

  dSL = a_sl x MAE_ref + b_sl x sigma_h + buffer_pips x pip        dTP = a_tp x TP_ref - b_tp x sigma_h
  SL rounded AWAY from the entry and TP TOWARD it on the price grid (Decimal arithmetic, one tick); an order is a NOP
  when a level is not strictly beyond both the entry quote and the liquidation-side quote by the broker minimum
  distance, when a reference or a gene makes a distance <= 0, or when the risk budget / broker minimum admit no units;
  a contracted TP is never inflated. Sizing (S3 wires it into the engine): q = quality x min(1, rho x balance / L1),
  L1 = units_max(q = 1) x (rounded SL distance + cost per unit + funding reserve); units = floor(q x units_max).

What is OURS and declared here, not his: the references come from the vendored TRAIN profiles (PROFILES.json, pinned);
sigma_h = causal_sigma(24 h) x sqrt(h / 24); the engine's cost convention (hour-dependent spread profile, slippage,
commission, swaps) is used in our runs, his constants only in the parity test. His primitives raise ValueError on bad
inputs; our consumer (safe_propose) turns every such case into a NOP with a counted reason, never an abort.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILES_PATH = ROOT / "docs/short_horizon/clean_round/contracts/PROFILES.json"
PROFILES_SHA256 = "aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96"      # agent-multi 82862801 docs/audits/evidence/zigzag_risk_profiles_20261028
CASES_PATH = ROOT / "docs/short_horizon/clean_round/contracts/evidence/CASES_shared_risk_rules_20261028.json.gz"
CASES_SHA256 = "8cdc263c645c9185c80c698e4cf3cce75ba03b895c93f63bfe1c27c5382ca1f1"         # agent-multi 82862801 docs/audits/evidence/shared_risk_rules_20261028
PIP = 1e-4
HORIZONS = (3, 6, 12, 24)
GENE_BOUNDS = {"a_sl": [0.5, 2.0], "b_sl": [0.0, 1.0], "buffer_pips": [0.0, 5.0], "a_tp": [0.5, 2.0], "b_tp": [0.0, 1.0], "rho": [0.001, 0.01]}   # his proposed search bounds
GENE_BASELINE = {"a_sl": 1.0, "b_sl": 0.0, "a_tp": 1.0, "b_tp": 0.0, "buffer_pips": 1.0, "rho": 0.005}
SIGMA_H_RULE = "sigma_h = causal_sigma(24 h, zigzag.causal_sigma at the decision) x sqrt(h / 24)"


class LevelInputError(ValueError):
    pass


def _pinned(path: Path, sha: str) -> bytes:
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise LevelInputError(f"PINNED_FILE_DIGEST:{path.name}:{got}!={sha}")
    return raw


def load_profiles() -> dict:
    return json.loads(_pinned(PROFILES_PATH, PROFILES_SHA256))


def load_cases() -> list:
    return json.loads(gzip.decompress(_pinned(CASES_PATH, CASES_SHA256)))


def references(h: int, side: int, quantile: str = "median", profiles: dict | None = None) -> dict:
    """(tp_reference, mae_reference) in PRICE units for horizon h and side (+1 buy / -1 sell), from the TRAIN profiles
    (median by default, p75 as the declared option). The profiles are hindsight oracle populations: priors, not
    executable levels (their own scope line)."""
    d = profiles if profiles is not None else load_profiles()
    row = next((r for r in d["rows"] if int(r["horizon"]) == int(h)), None)
    if row is None or quantile not in ("median", "p75"):
        raise LevelInputError(f"NO_REFERENCE:h{h}:{quantile}")
    p = row["directions"]["buy" if side > 0 else "sell"]["positive_calendar"]
    return {"tp_reference": float(p["tp_pips"][quantile]) * PIP, "mae_reference": float(p["mae_pips"][quantile]) * PIP, "n": int(p["n"]),
            "zigzag": dict(row["params"]), "quantile": quantile, "duration_hours": float(p["duration_hours"][quantile])}


def sigma_h(sigma24: float, h: int) -> float:
    return float(sigma24) * math.sqrt(float(h) / 24.0)


def coefficients(**overrides) -> dict:
    g = dict(GENE_BASELINE)
    unknown = set(overrides) - set(g)
    if unknown:
        raise LevelInputError(f"UNKNOWN_RISK_GENE:{sorted(unknown)}")
    g.update({k: float(v) for k, v in overrides.items()})
    if not all(math.isfinite(v) for v in g.values()) or min(g.values()) < 0 or g["a_sl"] <= 0 or g["a_tp"] <= 0 or g["rho"] > 1:
        raise LevelInputError("INVALID_RISK_GENE")
    return g


def levels(direction: int, entry_quote: float, tp_reference: float, mae_reference: float, sigma: float, genes: dict, tick: float) -> dict:
    """SL / TP prices on the tick grid: SL rounded away from the entry, TP toward it (a buy rounds both down, a sell both up)."""
    vals = (entry_quote, tp_reference, mae_reference, sigma, tick)
    if direction not in (-1, 1) or not all(math.isfinite(float(v)) for v in vals) or min(entry_quote, tp_reference, tick) <= 0 or min(mae_reference, sigma) < 0:
        raise LevelInputError("INVALID_LEVEL_INPUT")
    g = coefficients(**genes)
    D = lambda x: Decimal(str(x))  # noqa: E731
    e, s, step = D(entry_quote), D(direction), D(tick)
    d_sl = D(g["a_sl"]) * D(mae_reference) + D(g["b_sl"]) * D(sigma) + D(g["buffer_pips"]) * D(PIP)
    d_tp = D(g["a_tp"]) * D(tp_reference) - D(g["b_tp"]) * D(sigma)
    rounding = ROUND_FLOOR if direction == 1 else ROUND_CEILING
    grid = lambda x: float((x / step).to_integral_value(rounding=rounding) * step)  # noqa: E731
    return {"sl": grid(e - s * d_sl), "tp": grid(e + s * d_tp), "raw_sl_distance": float(d_sl), "raw_tp_distance": float(d_tp)}


def size_from_risk(balance: float, rel_volume: float, risk_budget: float, loss_at_q1: float, quality: float, minimum_q: float = 0.0) -> dict:
    vals = (balance, rel_volume, risk_budget, loss_at_q1, quality, minimum_q)
    if not all(math.isfinite(float(v)) for v in vals) or balance <= 0 or not 0 < rel_volume <= 1 or risk_budget < 0 or loss_at_q1 <= 0 \
            or not 0 <= quality <= 1 or not 0 <= minimum_q <= 1:
        raise LevelInputError("INVALID_SIZING_INPUT")
    q = quality * min(1.0, risk_budget / loss_at_q1)
    reason = "RISK_BOUNDED" if q > 0 else "ZERO_BUDGET_OR_QUALITY"
    if 0 < q < minimum_q:
        q, reason = 0.0, "BELOW_BROKER_MINIMUM"
    return {"q": q, "margin": q * rel_volume * balance, "estimated_loss": q * loss_at_q1, "reason": reason}


def propose(*, direction, entry_quote, liquidation_quote, balance, tp_reference, mae_reference, sigma, genes, sizing, tick, min_distance,
            quality, loss_cost_per_unit, funding_reserve_per_unit) -> dict:
    """The order template at fill time (quotes known; never the next open). Returns status ORDER with sl / tp / q /
    units / margin / estimated_loss, or NOP with a reason. Raises LevelInputError on an input outside the contract."""
    g = coefficients(**genes)
    vals = (liquidation_quote, balance, min_distance, quality, loss_cost_per_unit, funding_reserve_per_unit)
    if not all(math.isfinite(float(v)) for v in vals) or min(liquidation_quote, balance) <= 0 or min(min_distance, loss_cost_per_unit, funding_reserve_per_unit) < 0 \
            or not 0 <= quality <= 1 or direction not in (-1, 1) or direction * (entry_quote - liquidation_quote) < -1e-12:
        raise LevelInputError("INVALID_ORDER_INPUT")
    if sizing.rel_volume != 0.1 or sizing.leverage != 50 or getattr(sizing, "basis", "margin") != "margin":
        raise LevelInputError("BUSINESS_SIZING_CONTRACT")
    lv = levels(direction, entry_quote, tp_reference, mae_reference, sigma, g, tick)
    bad = lv["sl"] <= 0 or lv["tp"] <= 0 or lv["raw_tp_distance"] <= 0 or lv["raw_sl_distance"] <= 0 or any(
        direction * (lv["tp"] - p) <= 0 or direction * (p - lv["sl"]) <= 0 or min(abs(lv["tp"] - p), abs(p - lv["sl"])) < min_distance - 1e-12
        for p in (entry_quote, liquidation_quote))
    if bad:
        return {"status": "NOP", "reason": "LEVELS_NOT_EXECUTABLE", "q": 0.0, "units": 0, **lv}
    maximum = sizing.units_max(balance, entry_quote)
    if maximum < sizing.min_units:
        return {"status": "NOP", "reason": "BELOW_MINIMUM_UNITS", "q": 0.0, "units": 0, **lv}
    loss_per_unit = direction * (entry_quote - lv["sl"]) + loss_cost_per_unit + funding_reserve_per_unit
    risk_budget = g["rho"] * balance
    sized = size_from_risk(balance, sizing.rel_volume, risk_budget, maximum * loss_per_unit, quality)
    if sized["q"] == 0:
        return {"status": "NOP", "reason": sized["reason"], "q": 0.0, "units": 0, **lv}
    units = sizing.order_units(balance, entry_quote, percentual_order_size=sized["q"], risk_accepts_minimum=False)
    if units is None:
        return {"status": "NOP", "reason": "BELOW_MINIMUM_UNITS", "q": 0.0, "units": 0, **lv}
    margin, loss = sizing.required_margin(units, entry_quote), units * loss_per_unit
    if margin > 0.1 * balance + 1e-9 or loss > quality * risk_budget + 1e-9:
        raise LevelInputError("NATIVE_SIZING_CAP_BREACH")
    return {"status": "ORDER", "direction": direction, **lv, "q": sized["q"], "units": int(units), "margin": margin, "risk_budget": risk_budget,
            "estimated_loss": loss, "loss_at_q1": maximum * loss_per_unit, "loss_per_unit": loss_per_unit, "quality": quality, "genes": g}


def safe_propose(**kw) -> dict:
    """The consumer's form (coordinator ruling 2026-10-29): an input outside the contract (NaN / negative sigma, a bad
    gene, a bad quote) is a NOP with a counted reason, never an abort of the decision step."""
    try:
        return propose(**kw)
    except LevelInputError as e:
        return {"status": "NOP", "reason": f"INVALID_INPUT:{e.args[0] if e.args else 'LEVEL'}", "q": 0.0, "units": 0}
    except (TypeError, AttributeError) as e:
        return {"status": "NOP", "reason": f"INVALID_INPUT:{type(e).__name__}", "q": 0.0, "units": 0}
