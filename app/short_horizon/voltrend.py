"""VolTrendStrategy: trend forecast + volatility forecast -> direction, TP, SL, size (owner, 2026-10-19).

A single simulable component. Every weighting is a declared, bounded parameter in ``SCHEMA``; there is
no hidden constant. Geometry, per block with reference price r, trend forecast d (signed block drift)
and volatility forecast s:
  direction = sign(d) if |d| >= theta * s else no order (the control model t0 has d = 0 and uses the coin)
  TP        = r + dir * (w_trend * |d| + c_tp * s)                     (tp_form = "additive")
            = r + dir * min(max(w_trend * |d|, min_distance), c_tp * s) (tp_form = "clipped")
  SL        = r - dir * m * s
  size      = clip(pos_a + pos_b * |d| / s - pos_c * s / s_bar, pos_min, 1)   (percentual_order_size)
units follow the owner sizing rule (rel_volume x percentual_order_size x balance x leverage / price).
The objective adapter ``VolTrendStrategy.evaluate`` enforces the SL/TP/units rules, the daily cap, the
frequency rule and the ruin stop-out inside the simulation; an invalid configuration is penalized, never
dropped.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .blocks import Blocks, OK
from . import execution as X

PENALTY_SHARPE, PENALTY_DD = -10.0, 100.0

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "VolTrendStrategy parameters",
    "type": "object", "additionalProperties": False,
    "properties": {
        "c_tp": {"type": "number", "minimum": 0.0, "maximum": 3.0, "default": 1.0, "description": "vol multiple in the TP"},
        "m": {"type": "number", "minimum": 0.5, "maximum": 4.0, "default": 2.0, "description": "SL = entry - m * sigma_hat"},
        "w_trend": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 1.0, "description": "weight of |trend_hat| in the TP"},
        "theta": {"type": "number", "minimum": 0.0, "maximum": 50.0, "default": 0.0, "description": "abstain when |trend_hat| < theta * sigma_hat"},
        "tp_form": {"type": "string", "enum": ["additive", "clipped"], "default": "additive"},
        "pos_a": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
        "pos_b": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 0.0},
        "pos_c": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 0.0},
        "pos_min": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.25},
        "rel_volume": {"type": "number", "minimum": 0.01, "maximum": 0.20, "default": 0.10},
        "leverage": {"type": "integer", "enum": [10, 50, 100], "default": 100},
        "block_hours": {"type": "integer", "enum": [6, 24], "default": 24},
        "time_stop_hours": {"type": "integer", "minimum": 1, "maximum": 24, "default": 24, "description": "<= block_hours; the block end is the forced flat"},
        "trend_model": {"type": "string", "enum": ["t0", "t1", "t2", "t3", "t4", "tx"], "default": "t0"},
        "vol_model": {"type": "string", "enum": ["MEAN20", "HAR", "ORACLE"], "default": "HAR"},
        "max_orders_per_day": {"type": "integer", "minimum": 1, "maximum": 4, "default": 4},
        "min_trades_per_week": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 3.0},
    },
}
SCHEMA["required"] = list(SCHEMA["properties"])


class ParamError(ValueError):
    pass


def defaults() -> dict:
    return {k: v["default"] for k, v in SCHEMA["properties"].items()}


def validate(p: dict) -> dict:
    props = SCHEMA["properties"]
    extra = set(p) - set(props)
    if extra:
        raise ParamError(f"unknown parameters {sorted(extra)}")
    missing = set(props) - set(p)
    if missing:
        raise ParamError(f"missing parameters {sorted(missing)}")
    for k, spec in props.items():
        v = p[k]
        if spec["type"] == "string":
            if v not in spec["enum"]:
                raise ParamError(f"{k}={v!r} not in {spec['enum']}")
        elif spec["type"] == "integer":
            if isinstance(v, bool) or int(v) != v:
                raise ParamError(f"{k} must be an integer")
            if "enum" in spec and int(v) not in spec["enum"]:
                raise ParamError(f"{k}={v} not in {spec['enum']}")
            if "minimum" in spec and not (spec["minimum"] <= v <= spec["maximum"]):
                raise ParamError(f"{k}={v} outside [{spec['minimum']}, {spec['maximum']}]")
        else:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ParamError(f"{k} must be a finite number")
            if not (spec["minimum"] <= v <= spec["maximum"]):
                raise ParamError(f"{k}={v} outside [{spec['minimum']}, {spec['maximum']}]")
    if p["time_stop_hours"] > p["block_hours"]:
        raise ParamError("time_stop_hours must not exceed block_hours")
    return p


def _canonical(p: dict) -> str:
    return json.dumps({k: p[k] for k in sorted(p)}, sort_keys=True, separators=(",", ":"))


def manifest_hash(p: dict) -> str:
    return hashlib.sha256(_canonical(p).encode()).hexdigest()


def manifest_json(p: dict) -> str:
    return json.dumps({"parameters": {k: p[k] for k in sorted(p)}, "manifest_sha256": manifest_hash(p),
                       "schema_sha256": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()},
                      indent=1)


# ------------------------------------------------------------------ geometry
def build_orders(blocks: Blocks, trend_hat, sigma_hat, sigma_bar, p: dict, coin=None,
                 rules: X.OrderRules = X.OrderRules()) -> list:
    trend_hat, sigma_hat, sigma_bar = (np.asarray(x, float) for x in (trend_hat, sigma_hat, sigma_bar))
    out = []
    for k in range(len(blocks)):
        d, s, sb = trend_hat[k], sigma_hat[k], sigma_bar[k]
        if blocks.status[k] != OK or not (np.isfinite(d) and np.isfinite(s) and s > 0):
            out.append(None)
            continue
        if d == 0.0:
            direction = int(coin[k]) if coin is not None else 0
        else:
            direction = int(np.sign(d))
        if direction == 0 or abs(d) < p["theta"] * s:
            out.append(None)
            continue
        if p["tp_form"] == "additive":
            tp_dist = p["w_trend"] * abs(d) + p["c_tp"] * s
        else:
            tp_dist = min(max(p["w_trend"] * abs(d), rules.min_distance), p["c_tp"] * s)
        sl_dist = p["m"] * s
        ratio = (s / sb) if np.isfinite(sb) and sb > 0 else 1.0
        size = float(np.clip(p["pos_a"] + p["pos_b"] * abs(d) / s - p["pos_c"] * ratio, p["pos_min"], 1.0))
        r = float(blocks.ref[k])
        out.append(X.BlockOrder(direction, "MARKET", None, r + direction * tp_dist, r - direction * sl_dist, None, size))
    return out


# ------------------------------------------------------------------ simulable component
class VolTrendStrategy:
    """``inputs``: {"trend": {model_id: array}, "sigma": {model_id: array}, "sigma_bar": array, "coin": array,
    optional "sigma_rule": array (causal block sigma for the order rules; default = MEAN20 range)}."""

    def __init__(self, bars, blocks: Blocks, calendar, inputs: dict, capital: float = 10_000.0,
                 profile: X.RealisticProfile = X.RealisticProfile(), cost_mult: float = 1.0):
        self.bars, self.blocks, self.cal, self.inputs, self.capital = bars, blocks, calendar, inputs, capital
        self.profile = profile.scaled(cost_mult)
        self.rules = X.OrderRules()
        self.sigma_rule = inputs.get("sigma_rule")
        if self.sigma_rule is None:
            from .strategies import causal_block_sigma
            self.sigma_rule = causal_block_sigma(blocks)

    def orders(self, p: dict):
        validate(p)
        trend = self.inputs["trend"][p["trend_model"]]
        sigma = self.inputs["sigma"][p["vol_model"]]
        raw = build_orders(self.blocks, trend, sigma, self.inputs["sigma_bar"], p, coin=self.inputs.get("coin"),
                           rules=self.rules)
        checked, skipped = [], 0
        for k, o in enumerate(raw):
            if o is None or not np.isfinite(self.sigma_rule[k]):
                checked.append(None)
                continue
            try:
                X.check_order(o, float(self.blocks.ref[k]), float(self.sigma_rule[k]), self.rules)
                checked.append(o)
            except X.OrderRuleError:
                checked.append(None)
                skipped += 1
        return checked, skipped

    def evaluate(self, p: dict, period, seed: int = 0) -> dict:
        """Deterministic given (p, period, seed); seed only labels the run (the simulator is deterministic)."""
        orders, skipped = self.orders(p)
        sizing = X.Sizing(rel_volume=p["rel_volume"], leverage=float(p["leverage"]))
        res = X.run_backtest(self.bars, self.blocks, orders, self.profile, self.cal, self.capital, period,
                             profile=self.profile, max_orders_per_day=p["max_orders_per_day"], ruin_floor=0.0,
                             order_rules=self.rules, sigma_block=self.sigma_rule, sizing=sizing)
        m = res.metrics
        valid = bool(m["trades_per_week"] >= p["min_trades_per_week"] and m["max_orders_per_day"] <= p["max_orders_per_day"]
                     and not m["ruined"])
        sharpe = m["sharpe_daily_sqrt365"] if (valid and np.isfinite(m["sharpe_daily_sqrt365"])) else PENALTY_SHARPE
        dd = m["max_drawdown_pct"] if valid else PENALTY_DD
        return {"sharpe": m["sharpe_daily_sqrt365"], "max_dd": m["max_drawdown_pct"], "max_dd_usd": m["max_drawdown_usd"],
                "net": m["net_profit_usd"], "orders_per_week": m["trades_per_week"], "trades": m["trades"],
                "valid": valid, "ruined": bool(m["ruined"]), "objectives": (float(sharpe), float(dd)),
                "rule_skipped_orders": skipped, "share_exit_tp": m["share_exit_tp"], "share_exit_sl": m["share_exit_sl"],
                "share_exit_time": m["share_exit_time"], "max_orders_per_day": m["max_orders_per_day"],
                "seed": seed, "manifest_sha256": manifest_hash(p)}
