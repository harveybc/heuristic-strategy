"""Account, margin and cost convention for the long/short plugin.

Account currency is the quote currency. Notional is units times price.
Margin is notional divided by leverage. Backtrader's ``margin`` argument
is not that fraction: it is cash per unit on a futures-like book, and a
truthy value with ``commtype`` left unset also flips the commission from a
percent of price to a fixed amount per unit. This module does not use it
that way.
"""

from __future__ import annotations

import math

import backtrader as bt

MARGIN_FRACTION_CAP = 0.05
LOT_UNITS = 100_000.0
SUCCESSOR = "successor"
LEGACY = "legacy"
SWAP_CLOCK = (
    "UTC timestamps while the position is open, from the fill bar to the "
    "close-fill bar. Elapsed time is the timestamp delta, not a count of "
    "bars. Each increment is debited to cash before the decision that reads "
    "the balance. A gap charges the missing hours."
)

# commission_per_lot / LOT_UNITS is a per-side rate on notional, not a
# round-trip fee of 7 currency units. percabs=True, so the engine does not
# divide the rate by 100. Charge = abs(size) * rate * price on each fill.
COMMISSION_AUDIT = {
    "parameter": "commission_per_lot",
    "parameter_value": 7.0,
    "broker_rate": "commission_per_lot / 100000",
    "broker_rate_value": 0.00007,
    "legacy_call": "setcommission(commission=rate, margin=None, mult=1.0)",
    "engine_when_margin_is_none_and_commtype_is_none": {
        "stocklike": True,
        "commtype": "COMM_PERC",
        "percabs": True,
    },
    "charge": "abs(size) * rate * fill_price",
    "per_side": True,
    "round_trip_fee": False,
    "flat_fee_of_7_currency_units": False,
    "successor_keeps_the_per_side_notional_rate": True,
    "margin_parameter_not_used_as_fraction": (
        "Successor leaves CommInfo margin at None and sets commtype to "
        "COMM_PERC explicitly, so the rate stays a fraction of price. "
        "Collateral is stock-like notional divided by CommInfo leverage. "
        "shortcash is false: a short posts that collateral and is not "
        "credited the notional. automargin is not used; it would revalue "
        "the futures margin with price and add that residual to equity."
    ),
    "spread_slippage": (
        "Configured spread and slippage are price offsets, "
        "(spread_pips + slippage_pips) * pip_cost / 2 per side, applied by "
        "fixed slippage on the fill. They are inside the fill price. The "
        "ledger records them and does not debit them a second time."
    ),
    "swap": (
        "swap_per_lot_per_day currency units per 100000 units per 24 elapsed "
        "hours, both directions, debited once to cash on the timestamp clock."
    ),
}


class MarginFractionError(ValueError):
    """Raised when the margin fraction is outside (0, 0.05]."""


def validate_margin_fraction(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarginFractionError("margin_fraction must be a real number")
    fraction = float(value)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > MARGIN_FRACTION_CAP:
        raise MarginFractionError("margin_fraction must be in (0, 0.05]")
    return fraction


def notional(units: float, price: float) -> float:
    return abs(units) * price


def margin_required(units: float, price: float, leverage: float) -> float:
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    return notional(units, price) / leverage


def units_within_margin(
    price: float,
    margin_budget: float,
    leverage: float,
    cash: float,
    commission_rate: float,
) -> int:
    """Largest whole number of units whose margin stays inside the budget.

    Commission is not added to the margin. It can only reduce the size, so
    the cash still covers margin plus the opening commission. Rounding is
    downward.
    """
    if (
        price <= 0.0
        or margin_budget <= 0.0
        or leverage <= 0.0
        or cash <= 0.0
        or not math.isfinite(price)
        or not math.isfinite(margin_budget)
    ):
        return 0
    rate = commission_rate if commission_rate > 0.0 else 0.0
    raw = margin_budget * leverage / price
    units = int(math.floor(raw + 1e-9))
    while units > 0:
        margin = units * price / leverage
        commission = units * rate * price
        if margin <= margin_budget + 1e-9 and margin + commission <= cash + 1e-9:
            return units
        units -= 1
    return 0


def unconstrained_order_volume(reward_risk_ratio: float, params) -> float:
    if reward_risk_ratio >= params.upper_rr_threshold:
        return float(params.max_order_volume)
    if reward_risk_ratio <= params.lower_rr_threshold:
        return float(params.min_order_volume)
    span = params.upper_rr_threshold - params.lower_rr_threshold
    fraction = (reward_risk_ratio - params.lower_rr_threshold) / span
    return float(
        params.min_order_volume
        + fraction * (params.max_order_volume - params.min_order_volume)
    )


def compute_successor_order_size(
    *,
    reward_risk_ratio: float,
    price: float,
    equity: float,
    cash: float,
    margin_fraction: float,
    leverage: float,
    commission_rate: float,
    params,
) -> float:
    """RR size, then the shared margin cap. The minimum cannot raise margin."""
    fraction = validate_margin_fraction(margin_fraction)
    if equity <= 0.0 or price <= 0.0:
        return 0.0
    allowed = units_within_margin(
        price,
        equity * fraction,
        leverage,
        cash,
        commission_rate,
    )
    sized = min(unconstrained_order_volume(reward_risk_ratio, params), float(allowed))
    if sized < float(params.min_order_volume):
        return 0.0
    return float(int(math.floor(sized + 1e-9)))


def swap_cash(
    units: float,
    elapsed_hours: float,
    swap_per_lot_per_day: float,
    *,
    lot_units: float = LOT_UNITS,
) -> float:
    if elapsed_hours < 0.0 or not math.isfinite(elapsed_hours):
        raise ValueError("swap elapsed hours must be a non-negative finite number")
    if lot_units <= 0.0:
        raise ValueError("lot size must be positive")
    lots = abs(units) / lot_units
    return elapsed_hours / 24.0 * lots * swap_per_lot_per_day


def debit_cash(broker, amount: float) -> None:
    """Remove a cost from broker cash and the cached equity immediately."""
    if amount < 0.0 or not math.isfinite(amount):
        raise ValueError("cost debit must be a non-negative finite amount")
    if amount == 0.0:
        return
    broker.cash -= amount
    if hasattr(broker, "_value"):
        broker._value -= amount


def order_state_name(order) -> str:
    name = order.Status[order.status]
    if name == "Canceled":
        return "Cancelled"
    return name


def commission_cash(units: float, price: float, rate: float) -> float:
    return abs(units) * rate * price


class AccountingBroker(bt.brokers.BackBroker):
    """Stock-like collateral with a margin-fraction cap at the fill price.

    The cap runs at the execution price, after slippage. A close is not
    resized. An open that cannot meet the minimum without crossing the
    fraction is Margin, and the position is left unchanged.
    """

    def __init__(
        self,
        *,
        margin_fraction: float,
        trading_leverage: float,
        commission_rate: float,
        min_order_volume: float,
    ):
        self.margin_fraction = validate_margin_fraction(margin_fraction)
        self.trading_leverage = float(trading_leverage)
        if self.trading_leverage <= 0.0:
            raise ValueError("leverage must be positive")
        self.commission_rate = float(commission_rate)
        self.min_order_volume = float(min_order_volume)
        self.cap_events: list[dict] = []
        super().__init__()

    def _budget(self) -> float:
        equity = float(self.getvalue())
        if equity <= 0.0:
            return 0.0
        return equity * self.margin_fraction

    def _reject_margin(self, order) -> None:
        order.margin()
        self.notify(order)
        self._ococheck(order)
        self._bracketize(order, cancel=True)

    def _execute(self, order, ago=None, price=None, cash=None, position=None, dtcoc=None):
        if ago is not None and price is not None and price > 0.0 and order.executed.remsize:
            if self._cap_opening(order, float(price)):
                return None
        return super()._execute(
            order, ago=ago, price=price, cash=cash, position=position, dtcoc=dtcoc
        )

    def _cap_opening(self, order, price: float) -> bool:
        """Return True when the order was rejected and must not execute."""
        current = float(self.positions[order.data].size)
        requested = float(order.executed.remsize)
        if requested == 0.0:
            return False
        opens = _opening_units(current, requested)
        if opens == 0.0:
            return False
        budget = self._budget()
        allowed = units_within_margin(
            price,
            budget,
            self.trading_leverage,
            float(self.cash),
            self.commission_rate,
        )
        if abs(opens) <= allowed + 1e-9:
            self.cap_events.append(
                {
                    "action": "accepted",
                    "requested_units": requested,
                    "allowed_units": float(allowed),
                    "price": price,
                    "margin_budget": budget,
                    "margin_used": margin_required(opens, price, self.trading_leverage),
                }
            )
            return False
        if allowed < self.min_order_volume:
            self.cap_events.append(
                {
                    "action": "margin_rejected",
                    "requested_units": requested,
                    "allowed_units": allowed,
                    "price": price,
                    "margin_budget": budget,
                }
            )
            self._reject_margin(order)
            return True
        sign = 1.0 if requested > 0.0 else -1.0
        # Pure opens shrink. A reversal is not resized into a different trade.
        if current != 0.0 and (current > 0.0) != (requested > 0.0):
            self.cap_events.append(
                {
                    "action": "margin_rejected",
                    "requested_units": requested,
                    "allowed_units": allowed,
                    "price": price,
                    "margin_budget": budget,
                }
            )
            self._reject_margin(order)
            return True
        order.executed.remsize = sign * allowed
        self.cap_events.append(
            {
                "action": "capped",
                "requested_units": requested,
                "allowed_units": allowed,
                "price": price,
                "margin_budget": budget,
                "margin_used": margin_required(allowed, price, self.trading_leverage),
            }
        )
        return False


def _opening_units(current: float, requested: float) -> float:
    if requested == 0.0:
        return 0.0
    if current == 0.0:
        return requested
    if (current > 0.0 and requested > 0.0) or (current < 0.0 and requested < 0.0):
        return requested
    if abs(requested) <= abs(current) + 1e-9:
        return 0.0
    return requested + current


def configure_successor_broker(
    cerebro,
    *,
    cash: float,
    leverage: float,
    margin_fraction: float,
    commission_rate: float,
    min_order_volume: float,
    slippage_per_side: float,
):
    """Install the shared collateral convention. CommInfo margin stays None."""
    broker = AccountingBroker(
        margin_fraction=margin_fraction,
        trading_leverage=leverage,
        commission_rate=commission_rate,
        min_order_volume=min_order_volume,
    )
    cerebro.broker = broker
    broker.setcash(float(cash))
    broker.set_shortcash(False)
    broker.set_coc(False)
    broker.setcommission(
        commission=float(commission_rate),
        margin=None,
        mult=1.0,
        commtype=bt.CommInfoBase.COMM_PERC,
        percabs=True,
        stocklike=True,
        leverage=float(leverage),
        automargin=False,
    )
    broker.set_slippage_fixed(
        float(slippage_per_side), slip_open=True, slip_limit=True
    )
    return broker


def reconcile_book(
    *,
    initial_cash: float,
    cash: float,
    equity: float,
    position_units: float,
    entry_price: float,
    mark_price: float,
    leverage: float,
    realized_pnl: float,
) -> dict:
    """One identity: equity = cash + collateral + unrealized price PnL.

    Collateral is the cash posted at the fill, abs(units) * entry / leverage.
    Unrealized is signed units times (mark - entry). Costs already taken from
    cash are not added again.
    """
    if position_units:
        collateral = margin_required(position_units, entry_price, leverage)
        unrealized = position_units * (mark_price - entry_price)
    else:
        collateral = 0.0
        unrealized = 0.0
    identity = cash + collateral + unrealized
    flat = position_units == 0.0
    return {
        "collateral": collateral,
        "unrealized_pnl": unrealized,
        "equity_from_parts": identity,
        "equity_gap": equity - identity,
        "flat_cash_gap": (cash - initial_cash - realized_pnl) if flat else None,
        "flat": flat,
    }
