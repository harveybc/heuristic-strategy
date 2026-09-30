"""Elapsed-hour harness for the long/short plugin.

The legacy row-offset generators stay on process_data. This path runs only
when config['prediction_generator'] is 'elapsed_hours'. Decisions stay
close-only. Fills stay next-open market orders. No protective order is added.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

from app.plugins.plugin_long_short_predictions import Plugin
from app.strategy_support import (
    ELAPSED_OFFSET_UNIT,
    LEGACY_OFFSET_UNIT,
    RESERVED_START_UTC,
    _finite_price,
    _normalize_one,
    create_elapsed_hour_predictions,
    parse_horizons,
)

HARNESS_CONFIG_KEY = "prediction_generator"
HARNESS_CONFIG_VALUE = "elapsed_hours"
MAE_AGGREGATION = "macro_average_of_per_horizon_mae"
TERMINAL_CONVENTION = (
    "pending positions stay open; no forced liquidation; "
    "close() inside stop() is observed and is not treated as a fill"
)
EQUAL_MAE_NOTE = (
    "Equal MAE does not imply equal decisions. Persistence and "
    "ideal-plus-noise at the same MAE are different inputs. No sweep is run."
)


def _require_elapsed_config(config: dict) -> None:
    if config.get(HARNESS_CONFIG_KEY) != HARNESS_CONFIG_VALUE:
        raise ValueError(
            "elapsed-hour harness requires prediction_generator='elapsed_hours'"
        )
    if config.get("exit_variant") != "E":
        raise ValueError("elapsed-hour harness requires explicit exit_variant E")
    offset = config.get("offset_unit", ELAPSED_OFFSET_UNIT)
    if offset != "hours":
        raise ValueError("elapsed-hour harness offset unit must be hours")


def _label_frame(bars: pd.DataFrame, *, source_timezone: str) -> pd.DataFrame:
    from app.strategy_support import normalize_timestamp_index

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars index must be a DatetimeIndex")
    index = normalize_timestamp_index(bars.index, source_timezone=source_timezone)
    keep = [bool(stamp < RESERVED_START_UTC) for stamp in index]
    # iloc drops reserved rows before their prices are copied forward.
    label = bars.iloc[keep].copy()
    label.index = pd.DatetimeIndex(index[keep], name="DATE_TIME")
    return label


def _family_frame(
    ideal: pd.DataFrame,
    label: pd.DataFrame,
    columns: Iterable[str],
    mode: str,
    *,
    price_column: str,
) -> pd.DataFrame:
    selected = list(columns)
    if mode == "ideal":
        return ideal.loc[:, selected].copy()
    if mode != "persistence":
        raise ValueError("family mode must be ideal or persistence")
    origin = label[price_column].reindex(ideal.index)
    values = [
        _finite_price(raw, where="persistence origin") for raw in origin.tolist()
    ]
    return pd.DataFrame(
        {column: values for column in selected},
        index=ideal.index,
    )


def _rename_family(frame: pd.DataFrame, horizons: tuple[int, ...], prefix: str) -> pd.DataFrame:
    renamed = {
        f"elapsed_{hours}h": f"{prefix}{position}"
        for position, hours in enumerate(horizons, start=1)
    }
    return frame.rename(columns=renamed)


def _horizon_errors(
    label: pd.DataFrame,
    prediction: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    price_column: str,
) -> tuple[dict[str, float], dict[str, float]]:
    model: dict[str, float] = {}
    naive: dict[str, float] = {}
    origin_price = label[price_column].reindex(prediction.index)
    for hours in horizons:
        column = f"elapsed_{hours}h"
        targets = prediction.index + pd.Timedelta(hours=hours)
        actual = label[price_column].reindex(targets)
        actual.index = prediction.index
        gap = (prediction[column].astype(float) - actual.astype(float)).abs()
        naive_gap = (origin_price.astype(float) - actual.astype(float)).abs()
        model[str(hours)] = float(gap.mean())
        naive[str(hours)] = float(naive_gap.mean())
    return model, naive


def _macro(values: dict[str, float]) -> float:
    if not values:
        raise ValueError("MAE aggregation has no horizons")
    return float(sum(values.values()) / len(values))


def _iso(value) -> str | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def _broker_frame(label: pd.DataFrame) -> pd.DataFrame:
    required = ("OPEN", "HIGH", "LOW", "CLOSE")
    missing = [name for name in required if name not in label.columns]
    if missing:
        raise ValueError(f"broker feed missing {missing}")
    naive_index = label.index.tz_convert("UTC").tz_localize(None)
    broker = pd.DataFrame(
        {
            "open": label["OPEN"].to_numpy(dtype=float),
            "high": label["HIGH"].to_numpy(dtype=float),
            "low": label["LOW"].to_numpy(dtype=float),
            "close": label["CLOSE"].to_numpy(dtype=float),
            "volume": 0.0,
            "openinterest": 0.0,
        },
        index=naive_index,
    )
    broker.index.name = "DATE_TIME"
    return broker


class _ObservedStrategy(Plugin.HeuristicStrategy):
    """Records the real strategy's requests. It does not change them."""

    def __init__(self, *args, **kwargs):
        self.events = []
        self.cost_rows = []
        self.fill_rows = []
        self._requested_action = None
        super().__init__(*args, **kwargs)

    def buy(self, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        self._requested_action = ("buy", None if size is None else float(size))
        return super().buy(*args, **kwargs)

    def sell(self, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        self._requested_action = ("sell", None if size is None else float(size))
        return super().sell(*args, **kwargs)

    def close(self, *args, **kwargs):
        size = abs(float(self.position.size))
        result = super().close(*args, **kwargs)
        # close() submits buy or sell. Record the decision, not that child order.
        self._requested_action = ("close", size)
        return result

    def notify_order(self, order):
        super().notify_order(order)
        if order.status == order.Completed:
            self.fill_rows.append(
                {
                    "datetime": _iso(self.data0.datetime.datetime(0)),
                    "price": float(order.executed.price),
                    "size": float(order.executed.size),
                    "is_buy": bool(order.isbuy()),
                }
            )

    def notify_trade(self, trade):
        raw = None
        if trade.isclosed:
            raw = {"pnl_gross": float(trade.pnl), "pnl_after_commission": float(trade.pnlcomm)}
        super().notify_trade(trade)
        if raw is not None and self.trades:
            stored = float(self.trades[-1]["pnl"])
            raw["pnl_after_swap"] = stored
            raw["commission"] = raw["pnl_gross"] - raw["pnl_after_commission"]
            raw["swap"] = raw["pnl_after_commission"] - stored
            self.cost_rows.append(raw)

    def next(self):
        dt = self.data0.datetime.datetime(0)
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        snapshot = None
        if dt_hour in self.pred_df.index:
            row = self.pred_df.loc[dt_hour]
            snapshot = {
                "short": [
                    float(row[f"Prediction_h_{i}"])
                    for i in range(1, self.num_hourly_preds + 1)
                ],
                "long": [
                    float(row[f"Prediction_d_{i}"])
                    for i in range(1, self.num_daily_preds + 1)
                ],
            }
        self._requested_action = None
        position_before = float(self.position.size)
        super().next()
        action, size = self._requested_action or ("none", None)
        self.events.append(
            {
                "timestamp": _iso(dt_hour),
                "open": float(self.data0.open[0]),
                "close": float(self.data0.close[0]),
                "position_before": position_before,
                "request": action,
                "size": size,
                "direction_after": self.current_direction,
                "take_profit_after": (
                    None if self.current_tp is None else float(self.current_tp)
                ),
                "stop_loss_after": (
                    None if self.current_sl is None else float(self.current_sl)
                ),
                "predictions": snapshot,
            }
        )

    def stop(self):
        self.position_before_stop = float(self.position.size)
        self.closed_trades_before_stop = len(self.trades)
        self.marked_equity_before_stop = float(self.broker.getvalue())
        self.cash_before_stop = float(self.broker.getcash())
        super().stop()
        self.position_after_stop = float(self.position.size)
        self.closed_trades_after_stop = len(self.trades)
        self.stop_close_obtained_fill = bool(
            self.position_before_stop != 0.0 and self.position_after_stop == 0.0
        )


def _public_trades(trades: list[dict]) -> list[dict]:
    public = []
    for trade in trades:
        public.append(
            {
                "open_dt": _iso(trade.get("open_dt")),
                "close_dt": _iso(trade.get("close_dt")),
                "volume": float(trade.get("volume", 0.0)),
                "pnl": float(trade.get("pnl", 0.0)),
                "pips": float(trade.get("pips", 0.0)),
                "duration_bars": float(trade.get("duration", 0.0)),
                "max_dd_pips": float(trade.get("max_dd", 0.0)),
            }
        )
    return public


def run_elapsed_hour_harness(
    bars: pd.DataFrame,
    config: dict,
    *,
    work_directory: str,
) -> dict:
    """Run variant E on an elapsed-hour DEV feed through the real plugin."""
    _require_elapsed_config(config)
    source_timezone = config.get("source_timezone")
    if not isinstance(source_timezone, str):
        raise ValueError("source timezone must be declared; refusing to guess")
    price_column = config.get("price_column", "CLOSE")
    short_horizons = parse_horizons(
        config.get("short_horizons_hours"), what="short_horizons_hours"
    )
    long_horizons = parse_horizons(
        config.get("long_horizons_hours"), what="long_horizons_hours"
    )
    short_mode = config.get("short_family")
    long_mode = config.get("long_family")
    if short_mode not in {"ideal", "persistence"} or long_mode not in {
        "ideal",
        "persistence",
    }:
        raise ValueError("short_family and long_family must be ideal or persistence")

    label = _label_frame(bars, source_timezone=source_timezone)
    if label.empty:
        raise ValueError("DEV feed has no bars before the reserved cut")
    if label.index.max() >= RESERVED_START_UTC:
        raise RuntimeError("DEV feed retained a reserved bar")

    session_end = config.get("broker_session_end")
    if session_end is None:
        broker_label = label
    else:
        end = _normalize_one(session_end, source_timezone=source_timezone)
        if end >= RESERVED_START_UTC:
            raise ValueError("broker session end is on or after the reserved cut")
        broker_label = label.loc[label.index <= end]
        if broker_label.empty:
            raise ValueError("broker session is empty")

    ideal = create_elapsed_hour_predictions(
        label,
        short_horizons + long_horizons,
        price_column=price_column,
        source_timezone="UTC",
    )
    if ideal.empty:
        raise ValueError("elapsed-hour support is empty")
    consumed = tuple(ideal.attrs["consumed_timestamps_utc"])
    if any(stamp >= RESERVED_START_UTC for stamp in consumed):
        raise RuntimeError("elapsed predictions consumed a reserved timestamp")

    short_columns = [f"elapsed_{hours}h" for hours in short_horizons]
    long_columns = [f"elapsed_{hours}h" for hours in long_horizons]
    short_frame = _family_frame(
        ideal, label, short_columns, short_mode, price_column=price_column
    )
    long_frame = _family_frame(
        ideal, label, long_columns, long_mode, price_column=price_column
    )
    arm_prediction = pd.concat([short_frame, long_frame], axis=1)
    arm_mae, naive_mae = _horizon_errors(
        label,
        arm_prediction,
        short_horizons + long_horizons,
        price_column=price_column,
    )

    hourly = _rename_family(short_frame, short_horizons, "Prediction_h_")
    daily = _rename_family(long_frame, long_horizons, "Prediction_d_")
    predictions = hourly.join(daily, how="inner")
    if not predictions.index.equals(ideal.index):
        raise RuntimeError("short and long families lost the shared support")
    naive_utc = predictions.index.tz_convert("UTC").tz_localize(None)
    table = predictions.copy()
    table.index = naive_utc
    table.index.name = "DATE_TIME"

    os.makedirs(work_directory, exist_ok=True)
    prediction_path = os.path.join(work_directory, "predictions.csv")
    table.reset_index().to_csv(prediction_path, index=False)

    plugin = Plugin()
    params = plugin.params
    if params["exit_variant"] != "E":
        raise RuntimeError("plugin baseline is not variant E")
    if params["tp_multiplier"] != 0.9 or params["sl_multiplier"] != 2.0:
        raise RuntimeError("refusing to change TP or SL multipliers")

    import backtrader as bt

    import app.plugins.plugin_long_short_predictions as plugin_module

    plugin_module._QUIET = True
    import matplotlib

    matplotlib.use("Agg", force=True)

    initial_cash = float(config.get("initial_cash", 10000.0))
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        _ObservedStrategy,
        pred_file=prediction_path,
        pip_cost=params["pip_cost"],
        rel_volume=params["rel_volume"],
        min_order_volume=params["min_order_volume"],
        max_order_volume=params["max_order_volume"],
        leverage=params["leverage"],
        profit_threshold=params["profit_threshold"],
        min_drawdown_pips=params["min_drawdown_pips"],
        tp_multiplier=params["tp_multiplier"],
        sl_multiplier=params["sl_multiplier"],
        lower_rr_threshold=params["lower_rr_threshold"],
        upper_rr_threshold=params["upper_rr_threshold"],
        max_trades_per_5days=params["max_trades_per_5days"],
        exit_variant="E",
        swap_per_lot_per_day=params["swap_per_lot_per_day"],
    )
    cerebro.adddata(bt.feeds.PandasData(dataname=_broker_frame(broker_label)))
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.set_coc(False)
    spread_cost = params["spread_pips"] * params["pip_cost"]
    slippage_cost = params["slippage_pips"] * params["pip_cost"]
    total_spread = spread_cost + slippage_cost
    commission_per_unit = params["commission_per_lot"] / 100000.0
    cerebro.broker.setcommission(commission=commission_per_unit, margin=None, mult=1.0)
    cerebro.broker.set_slippage_fixed(total_spread / 2.0, slip_open=True, slip_limit=True)

    previous = os.getcwd()
    os.chdir(work_directory)
    try:
        strategy = cerebro.run()[0]
    finally:
        os.chdir(previous)

    trades = _public_trades(strategy.trades)
    realized = float(sum(trade["pnl"] for trade in trades))
    pending_units = float(strategy.position_after_stop)
    entry_sides = [
        "long" if event["request"] == "buy" else "short"
        for event in strategy.events
        if event["request"] in {"buy", "sell"}
    ]
    return {
        "label": "SYNTHETIC",
        "prediction_generator": HARNESS_CONFIG_VALUE,
        "offset_unit": ELAPSED_OFFSET_UNIT,
        "legacy_offset_unit": LEGACY_OFFSET_UNIT,
        "exit_variant": "E",
        "fill_semantics": "close_only_decision_next_open_market_fill",
        "protective_orders": False,
        "source_timezone": source_timezone.strip(),
        "normalized_timezone": "UTC",
        "reserved_start_utc": RESERVED_START_UTC.isoformat(),
        "short_family": short_mode,
        "long_family": long_mode,
        "family_pair": f"{short_mode}/{long_mode}",
        "short_horizons_hours": list(short_horizons),
        "long_horizons_hours": list(long_horizons),
        "plugin_columns": list(predictions.columns),
        "support_origins": [_iso(stamp) for stamp in ideal.index],
        "support_origin_count": int(len(ideal.index)),
        "broker_bar_count": int(len(broker_label)),
        "broker_first": _iso(broker_label.index[0]),
        "broker_last": _iso(broker_label.index[-1]),
        "per_horizon_mae": arm_mae,
        "per_horizon_naive_mae": naive_mae,
        "mae_aggregation": MAE_AGGREGATION,
        "mae_macro_average": _macro(arm_mae),
        "naive_macro_average": _macro(naive_mae),
        "initial_cash": initial_cash,
        "realized_pnl": realized,
        "pending_position_count": 0 if pending_units == 0.0 else 1,
        "pending_position_units": pending_units,
        "marked_equity": float(strategy.marked_equity_before_stop),
        "cash": float(strategy.cash_before_stop),
        "end_exposure_units": pending_units,
        "n_closed_trades": len(trades),
        "entry_sides": entry_sides,
        "trades": trades,
        "fills": strategy.fill_rows,
        "costs": {
            "spread_pips": float(params["spread_pips"]),
            "slippage_pips": float(params["slippage_pips"]),
            "pip_cost": float(params["pip_cost"]),
            "slippage_fixed_per_side": float(total_spread / 2.0),
            "commission_per_unit": float(commission_per_unit),
            "commission_per_lot": float(params["commission_per_lot"]),
            "swap_per_lot_per_day": float(params["swap_per_lot_per_day"]),
            "tp_multiplier": float(params["tp_multiplier"]),
            "sl_multiplier": float(params["sl_multiplier"]),
            "commission_measured": float(sum(row["commission"] for row in strategy.cost_rows)),
            "swap_measured": float(sum(row["swap"] for row in strategy.cost_rows)),
            "gross_pnl_measured": float(sum(row["pnl_gross"] for row in strategy.cost_rows)),
        },
        "position_before_stop": float(strategy.position_before_stop),
        "position_after_stop": float(strategy.position_after_stop),
        "closed_trades_before_stop": int(strategy.closed_trades_before_stop),
        "closed_trades_after_stop": int(strategy.closed_trades_after_stop),
        "stop_close_applicable": bool(strategy.position_before_stop != 0.0),
        "stop_close_obtained_fill": bool(strategy.stop_close_obtained_fill),
        "terminal_convention": TERMINAL_CONVENTION,
        "events": strategy.events,
        "equal_mae_note": EQUAL_MAE_NOTE,
    }
