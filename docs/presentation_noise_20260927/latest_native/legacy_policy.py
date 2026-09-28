from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from trading_contracts import (
    AssetAction,
    AssetIntent,
    DecisionContext,
    ProducerIdentity,
    RiskGeometry,
)


@dataclass(frozen=True)
class PredictionEntryExitParameters:
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
    exit_variant: str = "E"

    def __post_init__(self) -> None:
        if self.pip_cost <= 0:
            raise ValueError("pip_cost must be positive")
        if self.rel_volume < 0:
            raise ValueError("rel_volume cannot be negative")
        if self.min_order_volume < 0 or self.max_order_volume < self.min_order_volume:
            raise ValueError("order volume bounds are invalid")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.min_drawdown_pips <= 0:
            raise ValueError("min_drawdown_pips must be positive")
        if self.tp_multiplier <= 0 or self.sl_multiplier <= 0:
            raise ValueError("TP and SL multipliers must be positive")
        if self.upper_rr_threshold <= self.lower_rr_threshold:
            raise ValueError("upper_rr_threshold must exceed lower_rr_threshold")
        if self.max_trades_per_5days < 0:
            raise ValueError("max_trades_per_5days cannot be negative")
        if self.exit_variant.upper() not in set("ABCDEFG"):
            raise ValueError("exit_variant must be one of A through G")


@dataclass(frozen=True)
class EntryGeometry:
    direction: str
    take_profit_price: float
    stop_loss_price: float
    reward_risk_ratio: float
    ideal_profit_pips: float
    ideal_drawdown_pips: float


@runtime_checkable
class TradeLifecyclePolicy(Protocol):
    def decide(self, context: DecisionContext) -> AssetIntent:
        ...


def _finite_path(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def calculate_entry_geometry(
    *,
    current_price: float,
    long_horizon_predictions: Sequence[float],
    params: PredictionEntryExitParameters,
) -> EntryGeometry | None:
    predictions = _finite_path(long_horizon_predictions)
    if not predictions or not math.isfinite(current_price):
        return None

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
        return EntryGeometry(
            direction="long",
            take_profit_price=tp_buy,
            stop_loss_price=sl_buy,
            reward_risk_ratio=rr_buy,
            ideal_profit_pips=ideal_profit_buy,
            ideal_drawdown_pips=ideal_drawdown_buy,
        )
    if short_signal and rr_sell > rr_buy:
        return EntryGeometry(
            direction="short",
            take_profit_price=tp_sell,
            stop_loss_price=sl_sell,
            reward_risk_ratio=rr_sell,
            ideal_profit_pips=ideal_profit_sell,
            ideal_drawdown_pips=ideal_drawdown_sell,
        )
    return None


def should_early_close(
    *,
    direction: str,
    variant: str,
    short_horizon_predictions: Sequence[float],
    long_horizon_predictions: Sequence[float],
    stop_loss_price: float,
    entry_price: float | None,
) -> bool:
    short_path = _finite_path(short_horizon_predictions)
    long_path = _finite_path(long_horizon_predictions)
    mode = variant.upper()
    if mode == "G":
        return False
    if direction not in {"long", "short"}:
        raise ValueError("direction must be long or short")

    if direction == "long":
        short_trigger = bool(short_path) and min(short_path) < stop_loss_price
        long_trigger = bool(long_path) and min(long_path) < stop_loss_price
        if mode == "A":
            combined = short_path + long_path
            return bool(combined) and min(combined) < stop_loss_price
        if mode == "B":
            return long_trigger
        if mode == "C":
            return short_trigger
        if mode == "D":
            return short_trigger and long_trigger
        if mode == "E":
            if short_path and long_path:
                return 0.6 * min(short_path) + 0.4 * min(long_path) < stop_loss_price
            return short_trigger or long_trigger
        if mode == "F":
            buffer = 0.5 * abs(stop_loss_price - entry_price) if entry_price else 0.0
            buffered_short = bool(short_path) and min(short_path) < stop_loss_price - buffer
            return buffered_short or long_trigger
    else:
        short_trigger = bool(short_path) and max(short_path) > stop_loss_price
        long_trigger = bool(long_path) and max(long_path) > stop_loss_price
        if mode == "A":
            combined = short_path + long_path
            return bool(combined) and max(combined) > stop_loss_price
        if mode == "B":
            return long_trigger
        if mode == "C":
            return short_trigger
        if mode == "D":
            return short_trigger and long_trigger
        if mode == "E":
            if short_path and long_path:
                return 0.6 * max(short_path) + 0.4 * max(long_path) > stop_loss_price
            return short_trigger or long_trigger
        if mode == "F":
            buffer = 0.5 * abs(stop_loss_price - entry_price) if entry_price else 0.0
            buffered_short = bool(short_path) and max(short_path) > stop_loss_price + buffer
            return buffered_short or long_trigger
    return False


def compute_legacy_order_size(
    *,
    reward_risk_ratio: float,
    available_cash: float,
    params: PredictionEntryExitParameters,
) -> float:
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


def _path(context: DecisionContext, aliases: Sequence[str]) -> list[float]:
    for bundle in context.predictions:
        for alias in aliases:
            horizon = bundle.horizons.get(alias)
            if horizon is None:
                continue
            values = [
                value
                for key, value in sorted(horizon.outputs.items())
                if key.startswith("path_") and isinstance(value, (int, float))
            ]
            if values:
                return _finite_path(values)
    return []


class PredictionEntryExitPolicy:
    def __init__(
        self,
        params: PredictionEntryExitParameters,
        *,
        artifact_hash: str,
        producer: ProducerIdentity,
    ) -> None:
        self.params = params
        self.artifact_hash = artifact_hash
        self.producer = producer

    def _intent(
        self,
        context: DecisionContext,
        *,
        action: AssetAction,
        reason: str,
        target_exposure: float | None = None,
        risk_geometry: RiskGeometry | None = None,
    ) -> AssetIntent:
        return AssetIntent(
            object_id=f"{context.object_id}:{action.value}:{reason}",
            as_of=context.as_of,
            valid_until=context.valid_until,
            producer=self.producer,
            trace_id=context.trace_id,
            config_hash=context.config_hash,
            cell_id=context.cell_id,
            asset_id=context.market.asset_id,
            action=action,
            target_exposure=target_exposure,
            strategy_rel_volume=self.params.rel_volume,
            risk_geometry=risk_geometry,
            reason_codes=[reason],
            artifact_hash=self.artifact_hash,
        )

    def decide(self, context: DecisionContext) -> AssetIntent:
        current_price = context.market.local_features.get("close")
        if not isinstance(current_price, (int, float)) or not math.isfinite(float(current_price)):
            return self._intent(context, action=AssetAction.NO_TRADE, reason="missing_current_price")
        price = float(current_price)
        short_path = _path(context, ("short", "hourly", "1h"))
        long_path = _path(context, ("long", "daily", "1d"))

        if context.position.side in {"long", "short"}:
            side = context.position.side
            tp = context.position.take_profit
            sl = context.position.stop_loss
            if tp is not None and (
                (side == "long" and price >= tp) or (side == "short" and price <= tp)
            ):
                return self._intent(context, action=AssetAction.CLOSE, reason="take_profit_hit")
            if sl is not None and (
                (side == "long" and price <= sl) or (side == "short" and price >= sl)
            ):
                return self._intent(context, action=AssetAction.CLOSE, reason="stop_loss_hit")
            if sl is not None and should_early_close(
                direction=side,
                variant=self.params.exit_variant,
                short_horizon_predictions=short_path,
                long_horizon_predictions=long_path,
                stop_loss_price=sl,
                entry_price=context.position.average_price,
            ):
                return self._intent(context, action=AssetAction.CLOSE, reason="prediction_early_close")
            return self._intent(context, action=AssetAction.HOLD, reason="position_remains_valid")

        recent_trades = context.market.local_features.get("trades_last_5d", 0)
        if isinstance(recent_trades, (int, float)) and recent_trades >= self.params.max_trades_per_5days:
            return self._intent(context, action=AssetAction.NO_TRADE, reason="frequency_limit")
        geometry = calculate_entry_geometry(
            current_price=price,
            long_horizon_predictions=long_path,
            params=self.params,
        )
        if geometry is None:
            return self._intent(context, action=AssetAction.NO_TRADE, reason="no_entry_edge")
        exposure = 1.0 if geometry.direction == "long" else -1.0
        return self._intent(
            context,
            action=AssetAction.TARGET,
            reason=f"forecast_{geometry.direction}",
            target_exposure=exposure,
            risk_geometry=RiskGeometry(
                mode="forecast_path",
                stop_price=geometry.stop_loss_price,
                take_profit_price=geometry.take_profit_price,
            ),
        )

    @staticmethod
    def optimizable_parameter_schema() -> list[dict[str, float | str]]:
        return [
            {"name": "profit_threshold", "type": "float", "minimum": 0.5, "maximum": 20.0},
            {"name": "tp_multiplier", "type": "float", "minimum": 0.5, "maximum": 1.5},
            {"name": "sl_multiplier", "type": "float", "minimum": 1.5, "maximum": 6.0},
            {"name": "lower_rr_threshold", "type": "float", "minimum": 0.2, "maximum": 1.0},
            {"name": "upper_rr_threshold", "type": "float", "minimum": 1.3, "maximum": 6.0},
        ]
