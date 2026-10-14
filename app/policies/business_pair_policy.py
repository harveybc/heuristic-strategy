"""Business short/long pair policy with masked named heads (I6-DMO).

Pure standard library. It imports nothing from ``app`` (so it can be loaded by
explicit file path from a pinned commit without executing ``app/__init__`` or
``app/policies/__init__``, which import ``trading_contracts``), and nothing from
backtrader, pandas or numpy.

Owner contract (I6-N audit request section 3.1 and the market-week contract
section 3):

* Short family: 2, 3, 4, 5, 6, 12 physical hours. Long family: 18, 24, 36, 48,
  60, 72 physical hours. A head is identified by ``(family, horizon_hours)``,
  never by a column position.
* Each head carries ``available``. An unavailable head has no value and a
  reason; it is never replaced by the current price, zero or any default.
* Without an open position, entry uses ONLY the long family (minimum and
  maximum of the available long heads, same arithmetic as the legacy
  ``calculate_entry_geometry``). With fewer than ``min_valid_long_heads``
  distinct available long heads the policy abstains.
* With an open position, both families are ordered by ``horizon_hours`` and the
  FIRST SAMPLED head that touches a barrier of the CURRENT position decides:
  stop loss first -> early close; take profit first -> hold.

Declared rules where the owner contract leaves the detail open (owner rulings
pending; recorded in ``RULES``):

* ``BARRIER_EQUALITY_RULE = "INCLUSIVE_TOUCH"``: a forecast exactly equal to a
  barrier touches it (long: SL if value <= SL, TP if value >= TP; short
  mirrored). This matches the realised-price TP/SL checks of the legacy plugin.
* ``TIE_RULE``: a single sampled value cannot touch both barriers because the
  barriers are validated to lie on opposite sides (long SL < TP, short TP < SL).
  Two heads with the same ``(family, horizon_hours)`` are refused
  (``DUPLICATE_HEAD``); the two families have disjoint horizons, so two heads at
  the same physical hour are also refused (``DUPLICATE_HEAD``).
* ``NO_CROSSING_RULE``: no sampled touch -> HOLD (``NO_SAMPLED_CROSSING``).
  No available head at all -> HOLD (``NO_AVAILABLE_HEADS``). In both cases the
  real protective SL/TP orders and the calendar force-flat stay in force; they
  belong to the executor, not to this policy.
* ``SAMPLING_RULE``: only the sampled endpoints are inspected. Nothing is
  interpolated between them, and the result is not a claim of continuous-path
  first passage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

RULES_VERSION = "heuristic_strategy.business_pair_policy.v1"
SHORT_HORIZONS_HOURS: Tuple[int, ...] = (2, 3, 4, 5, 6, 12)
LONG_HORIZONS_HOURS: Tuple[int, ...] = (18, 24, 36, 48, 60, 72)
FAMILY_HORIZONS = {"short": SHORT_HORIZONS_HOURS, "long": LONG_HORIZONS_HOURS}

BARRIER_EQUALITY_RULE = "INCLUSIVE_TOUCH"
TIE_RULE = "BARRIERS_ON_OPPOSITE_SIDES_DUPLICATE_HOUR_REFUSED"
NO_CROSSING_RULE = "HOLD_REAL_PROTECTIONS_AND_FORCE_FLAT_REMAIN"
SAMPLING_RULE = "FIRST_SAMPLED_ENDPOINT_IN_HORIZON_ORDER_NO_INTERPOLATION"
MISSING_HEAD_RULE = "SKIP_WITH_REASON_NEVER_FILL"
ENTRY_FAMILY_RULE = "LONG_FAMILY_ONLY"

RULES = {
    "rules_version": RULES_VERSION,
    "barrier_equality": BARRIER_EQUALITY_RULE,
    "tie": TIE_RULE,
    "no_crossing": NO_CROSSING_RULE,
    "sampling": SAMPLING_RULE,
    "missing_head": MISSING_HEAD_RULE,
    "entry_family": ENTRY_FAMILY_RULE,
}


class PolicyError(ValueError):
    """Typed refusal; ``code`` is stable."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _finite_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError("NOT_A_NUMBER", f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise PolicyError("NON_FINITE", f"{name} must be finite")
    return value


@dataclass(frozen=True)
class HeadForecast:
    """One named head. ``available`` decides; ``value`` is None when unavailable."""

    family: str
    horizon_hours: int
    available: bool
    value: Optional[float]
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in FAMILY_HORIZONS:
            raise PolicyError("FAMILY_INVALID", f"family {self.family!r}")
        if type(self.horizon_hours) is not int or \
                self.horizon_hours not in FAMILY_HORIZONS[self.family]:
            raise PolicyError("HORIZON_NOT_IN_FAMILY",
                              f"{self.horizon_hours!r} not in {self.family} family")
        if not isinstance(self.available, bool):
            raise PolicyError("AVAILABLE_NOT_BOOL", "available must be a bool")
        if self.available:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) \
                    or not math.isfinite(float(self.value)):
                raise PolicyError("NON_FINITE_AVAILABLE_VALUE",
                                  f"{self.family} h{self.horizon_hours}: available needs a finite value")
            object.__setattr__(self, "value", float(self.value))
        else:
            if self.value is not None:
                raise PolicyError("UNAVAILABLE_HEAD_CARRIES_VALUE",
                                  f"{self.family} h{self.horizon_hours}: unavailable head has a value")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise PolicyError("UNAVAILABLE_HEAD_WITHOUT_REASON",
                                  f"{self.family} h{self.horizon_hours}: reason required")


@dataclass(frozen=True)
class EntryParameters:
    """Legacy ``PredictionEntryExitParameters`` fields plus the minimum long heads."""

    pip_cost: float = 0.00001
    rel_volume: float = 0.02
    min_order_volume: float = 10_000.0
    max_order_volume: float = 1_000_000.0
    leverage: float = 100.0
    profit_threshold: float = 5.0
    min_drawdown_pips: float = 10.0
    tp_multiplier: float = 0.9
    sl_multiplier: float = 2.0
    lower_rr_threshold: float = 0.5
    upper_rr_threshold: float = 2.0
    max_trades_per_5days: int = 3
    min_valid_long_heads: int = 2

    def __post_init__(self) -> None:
        def bad(detail: str) -> PolicyError:
            return PolicyError("PARAMETER_INVALID", detail)
        if self.pip_cost <= 0:
            raise bad("pip_cost must be positive")
        if self.rel_volume < 0:
            raise bad("rel_volume cannot be negative")
        if self.min_order_volume < 0 or self.max_order_volume < self.min_order_volume:
            raise bad("order volume bounds are invalid")
        if self.leverage <= 0:
            raise bad("leverage must be positive")
        if self.min_drawdown_pips <= 0:
            raise bad("min_drawdown_pips must be positive")
        if self.tp_multiplier <= 0 or self.sl_multiplier <= 0:
            raise bad("TP and SL multipliers must be positive")
        if self.upper_rr_threshold <= self.lower_rr_threshold:
            raise bad("upper_rr_threshold must exceed lower_rr_threshold")
        if type(self.max_trades_per_5days) is not int or self.max_trades_per_5days < 0:
            raise bad("max_trades_per_5days must be a non-negative int")
        if type(self.min_valid_long_heads) is not int or \
                not 1 <= self.min_valid_long_heads <= len(LONG_HORIZONS_HOURS):
            raise bad("min_valid_long_heads must be an int in 1..6")


@dataclass(frozen=True)
class EntryGeometry:
    direction: str
    take_profit_price: float
    stop_loss_price: float
    reward_risk_ratio: float
    ideal_profit_pips: float
    ideal_drawdown_pips: float


@dataclass(frozen=True)
class EntryDecision:
    action: str  # OPEN_LONG | OPEN_SHORT | ABSTAIN
    reason: str
    geometry: Optional[EntryGeometry]
    consulted: Tuple[Tuple[str, int, float], ...]
    skipped: Tuple[Tuple[str, int, str], ...]
    rules_version: str = RULES_VERSION


@dataclass(frozen=True)
class ExitDecision:
    action: str  # EARLY_CLOSE | HOLD
    reason: str  # SL_FIRST | TP_FIRST | NO_SAMPLED_CROSSING | NO_AVAILABLE_HEADS
    first_crossing_horizon_hours: Optional[int]
    first_crossing_family: Optional[str]
    consulted: Tuple[Tuple[str, int, float], ...]
    skipped: Tuple[Tuple[str, int, str], ...]
    rules_version: str = RULES_VERSION


def _ordered(heads: Sequence[HeadForecast]):
    seen = set()
    hours = set()
    for item in heads:
        if not isinstance(item, HeadForecast):
            raise PolicyError("HEAD_TYPE_INVALID", "heads must be HeadForecast records")
        key = (item.family, item.horizon_hours)
        if key in seen or item.horizon_hours in hours:
            raise PolicyError("DUPLICATE_HEAD", f"{key} given twice")
        seen.add(key)
        hours.add(item.horizon_hours)
    ordered = sorted(heads, key=lambda x: x.horizon_hours)
    consulted = tuple((x.family, x.horizon_hours, x.value) for x in ordered if x.available)
    skipped = tuple((x.family, x.horizon_hours, x.reason) for x in ordered if not x.available)
    return ordered, consulted, skipped


def _geometry(current_price: float, predictions: Sequence[float],
              params: EntryParameters) -> Optional[EntryGeometry]:
    # Arithmetic copied operation by operation from the legacy
    # app/policies/prediction_entry_exit.py::calculate_entry_geometry
    # (blob 094058959bd50b2947a0bbefd014d3bf6da5fb11); a parity test pins it.
    ideal_profit_buy = (max(predictions) - current_price) / params.pip_cost
    ideal_drawdown_buy = max(
        (current_price - min(predictions)) / params.pip_cost,
        params.min_drawdown_pips,
    )
    rr_buy = ideal_profit_buy / ideal_drawdown_buy if ideal_drawdown_buy > 0 else 0.0
    tp_buy = current_price + params.tp_multiplier * ideal_profit_buy * params.pip_cost
    sl_buy = current_price - params.sl_multiplier * ideal_drawdown_buy * params.pip_cost

    ideal_profit_sell = (current_price - min(predictions)) / params.pip_cost
    ideal_drawdown_sell = max(
        (max(predictions) - current_price) / params.pip_cost,
        params.min_drawdown_pips,
    )
    rr_sell = ideal_profit_sell / ideal_drawdown_sell if ideal_drawdown_sell > 0 else 0.0
    tp_sell = current_price - params.tp_multiplier * ideal_profit_sell * params.pip_cost
    sl_sell = current_price + params.sl_multiplier * ideal_drawdown_sell * params.pip_cost

    long_signal = ideal_profit_buy >= params.profit_threshold
    short_signal = ideal_profit_sell >= params.profit_threshold
    if long_signal and rr_buy >= rr_sell:
        return EntryGeometry("long", tp_buy, sl_buy, rr_buy,
                             ideal_profit_buy, ideal_drawdown_buy)
    if short_signal and rr_sell > rr_buy:
        return EntryGeometry("short", tp_sell, sl_sell, rr_sell,
                             ideal_profit_sell, ideal_drawdown_sell)
    return None


def entry_decision(*, current_price: float, long_heads: Sequence[HeadForecast],
                   params: EntryParameters) -> EntryDecision:
    """Entry candidate from the available long heads only."""
    if not isinstance(params, EntryParameters):
        raise PolicyError("PARAMETER_INVALID", "params must be EntryParameters")
    price = _finite_number("current_price", current_price)
    for item in long_heads:
        if isinstance(item, HeadForecast) and item.family != "long":
            raise PolicyError("SHORT_HEAD_IN_ENTRY",
                              f"entry consumes the long family only, got {item.family}")
    _, consulted, skipped = _ordered(long_heads)
    if len(consulted) < params.min_valid_long_heads:
        return EntryDecision("ABSTAIN", "INSUFFICIENT_LONG_HEADS", None, consulted, skipped)
    geometry = _geometry(price, [c[2] for c in consulted], params)
    if geometry is None:
        return EntryDecision("ABSTAIN", "NO_ENTRY_EDGE", None, consulted, skipped)
    action = "OPEN_LONG" if geometry.direction == "long" else "OPEN_SHORT"
    return EntryDecision(action, f"FORECAST_{geometry.direction.upper()}", geometry,
                         consulted, skipped)


def early_close_decision(*, direction: str, take_profit: float, stop_loss: float,
                         heads: Sequence[HeadForecast]) -> ExitDecision:
    """Chronological first-sampled-touch rule for the CURRENT position barriers."""
    if direction not in ("long", "short"):
        raise PolicyError("INVALID_DIRECTION", f"direction {direction!r}")
    tp = _finite_number("take_profit", take_profit)
    sl = _finite_number("stop_loss", stop_loss)
    if (direction == "long" and not sl < tp) or (direction == "short" and not tp < sl):
        raise PolicyError("INVALID_BARRIERS",
                          f"{direction}: SL {sl} and TP {tp} are not on opposite sides")
    ordered, consulted, skipped = _ordered(heads)
    if not consulted:
        return ExitDecision("HOLD", "NO_AVAILABLE_HEADS", None, None, consulted, skipped)
    for item in ordered:
        if not item.available:
            continue
        v = item.value
        if direction == "long":
            sl_touch, tp_touch = v <= sl, v >= tp
        else:
            sl_touch, tp_touch = v >= sl, v <= tp
        if sl_touch:
            return ExitDecision("EARLY_CLOSE", "SL_FIRST", item.horizon_hours, item.family,
                                consulted, skipped)
        if tp_touch:
            return ExitDecision("HOLD", "TP_FIRST", item.horizon_hours, item.family,
                                consulted, skipped)
    return ExitDecision("HOLD", "NO_SAMPLED_CROSSING", None, None, consulted, skipped)


def compute_order_units(*, reward_risk_ratio: float, available_cash: float,
                        params: EntryParameters) -> float:
    """Legacy ``compute_legacy_order_size`` arithmetic (units of base currency)."""
    if reward_risk_ratio >= params.upper_rr_threshold:
        unconstrained = params.max_order_volume
    elif reward_risk_ratio <= params.lower_rr_threshold:
        unconstrained = params.min_order_volume
    else:
        fraction = (
            (reward_risk_ratio - params.lower_rr_threshold)
            / (params.upper_rr_threshold - params.lower_rr_threshold)
        )
        unconstrained = params.min_order_volume + fraction * (
            params.max_order_volume - params.min_order_volume
        )
    max_from_cash = available_cash * params.rel_volume * params.leverage
    return min(unconstrained, max_from_cash)
