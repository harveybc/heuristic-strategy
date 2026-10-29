"""Declared execution-layer variants (fix lane F3, audit register 2026-10-26: A4-1, A4-2, A4-4, A4-7, A4-9, A4-10, A4-12).

Every variant is an explicit, identity-covered choice carried by ``EnvConfig.variant``; ``None`` (the default) keeps
the engine byte-identical to the audited behaviour. Nothing here reads the future: every rule acts at the decision
after bar b with information known then.

  A4-1  reverse_counts_as_one_entry   a close-and-reverse re-uses the cap slot of the position it closes when that
                                      position was opened in the same business day (one entry DECISION); a new entry
                                      still obeys the daily cap
  A4-2  best_of_day                   with a daily cap the first qualifying bar is not the best one; ``BestOfDayGate``
                                      is a causal secretary rule: observe ``observe_bars`` scored bars of the business
                                      day without entering, then enter at the first bar whose score is >= the best
                                      score seen so far in the day and >= ``threshold``; no look-ahead inside the day
  A4-4  sizing_mode="risk"            units = risk_pct x balance / stop distance (USD per unit = price distance for
                                      EURUSD), capped by the owner's rel_volume rule (<= rel_volume x balance x
                                      leverage / price) and the margin, integer, >= 1 or refused (UNITS)
  A4-9  entry_cutoff_hours_before_flat an entry is refused when its fill is less than this many hours before the forced
                                      flat (the calendar's own cutoff stays in force); the calendar clock (hours since
                                      the operational start / to the forced flat) is exposed to the policy only through
                                      ``calendar_clock_mask`` when a contract declares it: it is calendar, not feedback,
                                      and the engine's agent-state modes and CLI surface stay unchanged
  A4-7  early_close_ok                selection helper: a strategy point qualifies only with a minimum SHARE of agent
                                      closes AND an early-close value whose lower confidence bound is >= 0 on held-out
                                      calibration rows (the strategy layer calls it; this module owns the rule)
  A4-10 effective_cap / clip_multiples the engine clips every level into [min_distance, effectiveness_c x sigma_rule];
                                      heads and grids must be bounded at that cap in sigma multiples, per bar
  A4-12 short_trigger_on_ask          bars are bid; a short's buy-stop triggers when ASK >= SL, i.e. bid high >= SL -
                                      spread, and its buy-limit TP fills when ASK <= TP, i.e. bid low <= TP - spread;
                                      the fill prices stay the order's levels (declared approximation)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

SIZING_MODES = ("notional_share", "risk", "risk_budget")   # risk_budget = the business contract's q = min(1, rho x balance / L1) (S3)
LEVEL_MODES = ("sigma", "absolute_pips")


@dataclass(frozen=True)
class ExecVariant:
    reverse_counts_as_one_entry: bool = False          # A4-1
    sizing_mode: str = "notional_share"                 # A4-4: "risk" = risk_pct of balance per stop distance
    risk_pct: float = 0.01
    entry_cutoff_hours_before_flat: float | None = None  # A4-9
    short_trigger_on_ask: bool = False                  # A4-12
    leverage: float | None = None                       # business contract 2026-10-29: the account leverage scenario (None = EnvConfig.sizing)
    no_daily_cap: bool = False                          # business contract 2026-10-29: max_entries_per_day = None (no strategy-imposed cap)
    level_mode: str = "sigma"                           # S2: "absolute_pips" = the policy passes SL / TP PRICES built by horizon_levels (no sigma bounds, no band)
    rho: float = 0.005                                  # S3 risk_budget sizing: the risk budget share of the CURRENT balance (a gene from S6 on; baseline 0.005)
    funding_reserve_per_unit: float = 0.0               # S3: funding reserve per unit added to the modeled loss (declared constant)
    quality_c: float = 1.0                              # S3: c of q = c x min(1, R / L1); fixed 1 until head confidence is calibrated (coordinator ruling)
    note: str = ""

    def __post_init__(self):
        if self.sizing_mode not in SIZING_MODES:
            raise ValueError(f"SIZING_MODE_UNKNOWN:{self.sizing_mode}")
        if not 0 < self.risk_pct <= 0.05:
            raise ValueError("RISK_PCT_OUT_OF_RANGE:(0, 0.05]")
        if self.entry_cutoff_hours_before_flat is not None and self.entry_cutoff_hours_before_flat < 0:
            raise ValueError("ENTRY_CUTOFF_NEGATIVE")
        if self.leverage is not None and not 0 < self.leverage <= 100:
            raise ValueError("LEVERAGE_OUT_OF_RANGE:(0, 100]")
        if self.level_mode not in LEVEL_MODES:
            raise ValueError(f"LEVEL_MODE_UNKNOWN:{self.level_mode}")
        if not (0 <= self.rho <= 1) or self.funding_reserve_per_unit < 0 or not (0 <= self.quality_c <= 1):
            raise ValueError("RISK_BUDGET_PARAMS_OUT_OF_RANGE")

    def is_legacy(self) -> bool:
        """True when every option equals the audited default (the engine path is then byte-identical)."""
        return self == ExecVariant(note=self.note)

    def document(self) -> dict:
        d = asdict(self)
        d.pop("note")
        for k, default in (("leverage", None), ("no_daily_cap", False), ("level_mode", "sigma"), ("rho", 0.005), ("funding_reserve_per_unit", 0.0), ("quality_c", 1.0)):   # fields added 2026-10-29: absent at their defaults, so every earlier identity is unchanged
            if d[k] == default:
                d.pop(k)
        return d

    @property
    def identity(self) -> str:
        return "exec_" + hashlib.sha256(json.dumps(self.document(), sort_keys=True).encode()).hexdigest()[:16]


LEGACY = ExecVariant(note="audited behaviour (A4 register)")


# ------------------------------------------------------------------------------------------------ A4-9 calendar clock
# The session clock (hours since the operational start / to the forced flat, in days) is CALENDAR information, not
# agent feedback. It stays off unless a contract declares ``calendar_clock_inputs``; then the policy input keeps exactly
# these two slots of env.STATE_NAMES (the engine's AGENT_STATE_MODES and the CLI surface are unchanged).
CALENDAR_CLOCK_SLOTS = ("hours_since_operational_start_days", "hours_to_forced_flat_days")


def calendar_clock_mask(base_mode: str = "none"):
    """Float mask over env.STATE_NAMES: the base agent-state mode's slots plus the two calendar-clock slots."""
    from .env import AGENT_STATE_MODES, STATE_NAMES
    if base_mode not in AGENT_STATE_MODES:
        raise ValueError(f"AGENT_STATE_UNKNOWN:{base_mode}")
    keep = set(AGENT_STATE_MODES[base_mode]) | set(CALENDAR_CLOCK_SLOTS)
    return np.array([1.0 if n in keep else 0.0 for n in STATE_NAMES], np.float32)


# ------------------------------------------------------------------------------------------------ A4-4 sizing
def risk_budget_units(balance: float, price: float, stop_distance: float, cost_per_unit: float, funding_reserve: float, rho: float, quality: float, sizing) -> dict:
    """S3 (business contract): q = quality x min(1, rho x balance / L1), L1 = units_max x loss_per_unit, loss_per_unit = the
    rounded SL distance + the cost per unit + the funding reserve; units = order_units(q) (floor of q x units_max,
    refused below the broker minimum or above the free margin). Returns {"units", "q", "loss_per_unit", "reason"}."""
    from . import horizon_levels as HL
    if not (np.isfinite(stop_distance) and stop_distance > 0 and balance > 0):
        return {"units": None, "q": 0.0, "loss_per_unit": float("nan"), "reason": "STOP_DISTANCE"}
    maximum = sizing.units_max(balance, price)
    if maximum < sizing.min_units:
        return {"units": None, "q": 0.0, "loss_per_unit": float("nan"), "reason": "BELOW_MINIMUM_UNITS"}
    loss_per_unit = float(stop_distance) + float(cost_per_unit) + float(funding_reserve)
    sized = HL.size_from_risk(balance, sizing.rel_volume, float(rho) * balance, maximum * loss_per_unit, float(quality))
    if sized["q"] <= 0:
        return {"units": None, "q": 0.0, "loss_per_unit": loss_per_unit, "reason": sized["reason"]}
    u = sizing.order_units(balance, price, percentual_order_size=sized["q"], risk_accepts_minimum=False)
    if u is None:
        return {"units": None, "q": sized["q"], "loss_per_unit": loss_per_unit, "reason": "BELOW_MINIMUM_UNITS_OR_MARGIN"}
    return {"units": int(u), "q": sized["q"], "loss_per_unit": loss_per_unit, "reason": "RISK_BOUNDED"}


def risk_units(balance: float, price: float, stop_distance: float, risk_pct: float, sizing) -> int | None:
    """Integer units risking ``risk_pct`` of the balance at the stop, capped by the rel_volume/leverage rule and the
    margin; None when below the broker minimum or the stop distance is not usable."""
    if not (np.isfinite(stop_distance) and stop_distance > 0 and balance > 0):
        return None
    u = int(math.floor(risk_pct * balance / stop_distance))
    u = min(u, sizing.units_max(balance, price))
    if u < sizing.min_units:
        return None
    if sizing.required_margin(u, price) > balance:
        return None
    return u


# ------------------------------------------------------------------------------------------------ A4-10 effective cap
def effective_cap(env, b: int) -> float:
    """The largest level distance the engine keeps, in sigma multiples at the decision after bar b."""
    s, sr = env.decision_sigma(b), float(env.sigma_rule[min(b + 1, len(env.sigma_rule) - 1)])
    if not (np.isfinite(s) and s > 0 and np.isfinite(sr) and sr > 0):
        return float("nan")
    return env.cfg.rules.effectiveness_c * sr / s


def min_multiple(env, b: int) -> float:
    s = env.decision_sigma(b)
    return env.cfg.rules.min_distance / s if np.isfinite(s) and s > 0 else float("nan")


def clip_multiples(env, b: int, tp_mult: float, sl_mult: float):
    """(tp, sl, clipped) as the engine will execute them: the declared bounds, then the per-bar effective band."""
    cfg = env.cfg
    cap, lo = effective_cap(env, b), min_multiple(env, b)
    tp = float(np.clip(tp_mult, *cfg.tp_bounds))
    sl = float(np.clip(sl_mult, *cfg.sl_bounds))
    if not np.isfinite(cap):
        return tp, sl, False
    ntp, nsl = min(max(tp, lo), cap), min(max(sl, lo), cap)
    return ntp, nsl, (ntp != tp) or (nsl != sl)


def cap_profile(env, bars: np.ndarray) -> dict:
    """Quantiles of the effective cap over the given decision bars (for bounding grids and heads per fold)."""
    caps = np.array([effective_cap(env, int(b)) for b in bars], float)
    caps = caps[np.isfinite(caps)]
    if not len(caps):
        return {"n": 0}
    return {"n": int(len(caps)), **{f"p{q}": float(np.percentile(caps, q)) for q in (1, 5, 25, 50, 75, 95, 99)}}


# ------------------------------------------------------------------------------------------------ A4-7 early close
def early_close_ok(agent_close_share: float, value_lcb: float, min_share: float = 0.05) -> tuple[bool, str]:
    """A4-7: a strategy qualifies only when agent closes are at least ``min_share`` of its exits AND the lower
    confidence bound of its early-close value (actual minus hold-to-TP/SL counterfactual) is >= 0."""
    if not np.isfinite(agent_close_share) or agent_close_share < min_share:
        return False, f"EARLY_CLOSE_SHARE_BELOW_MIN:{agent_close_share:.4f}<{min_share}"
    if not np.isfinite(value_lcb) or value_lcb < 0:
        return False, f"EARLY_CLOSE_VALUE_LCB_NEGATIVE:{value_lcb:.4f}"
    return True, "OK"


# ------------------------------------------------------------------------------------------------ A4-2 best of day
class BestOfDayGate:
    """Wrap a SCORED policy ``act(bar, r, st, env) -> (action, tp, sl, score)`` into a plain policy for
    ``Ctx.rollout``. Causal secretary rule per business day: the first ``observe_bars`` eligible flat decisions of the
    day are observed only (HOLD); afterwards the first open signal whose score is >= the day's running best score AND
    >= ``threshold`` is placed. ``observe_bars=0`` degenerates to the first-qualifying rule the audit measured.
    Positioned bars (closes/holds) pass through untouched. Counts first-vs-best diagnostics."""

    def __init__(self, act, observe_bars: int = 3, threshold: float = -math.inf):
        if observe_bars < 0:
            raise ValueError("OBSERVE_BARS_NEGATIVE")
        self.act, self.observe_bars, self.threshold = act, int(observe_bars), float(threshold)
        self.day, self.seen, self.best = None, 0, -math.inf
        self.stats = {"days": 0, "placed": 0, "deferred_observing": 0, "deferred_below_best": 0, "below_threshold": 0}

    def _reset_day(self, day):
        if day != self.day:
            self.day, self.seen, self.best = day, 0, -math.inf
            self.stats["days"] += 1

    def __call__(self, bar, r, st, env):
        out = self.act(bar, r, st, env)
        action, tp, sl, score = out
        if int(st[0]) != 0:
            return action, tp, sl
        k = min(bar + 1, len(env.v) - 1)
        self._reset_day(env.business_day(k))
        ok, _ = env.entry_gate(bar)
        if not ok:
            if action in (1, 2):
                self.stats["entry_gate_closed"] = self.stats.get("entry_gate_closed", 0) + 1   # counted: the open never reaches the engine
            return 0, tp, sl
        if action not in (1, 2):
            return action, tp, sl
        self.seen += 1
        prev_best = self.best
        self.best = max(self.best, float(score))
        if score < self.threshold:
            self.stats["below_threshold"] += 1
            return 0, tp, sl
        if self.seen <= self.observe_bars:
            self.stats["deferred_observing"] += 1
            return 0, tp, sl
        if score < prev_best - 1e-9:                                  # a tie within float rounding is not 'below' (a world and its exact
            self.stats["deferred_below_best"] += 1                    # mirror scored 0.7499999999999943 vs 0.75 and only one of them entered)
            return 0, tp, sl
        self.stats["placed"] += 1
        return action, tp, sl


def first_of_day(act):
    """The audited rule: the first qualifying bar of the day enters (a scored policy made plain)."""
    return BestOfDayGate(act, observe_bars=0)
