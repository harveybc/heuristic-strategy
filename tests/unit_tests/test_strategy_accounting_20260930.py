"""Successor account convention. These checks fail on the legacy cash and margin rules."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.account_convention import (
    COMMISSION_AUDIT,
    LOT_UNITS,
    MARGIN_FRACTION_CAP,
    SUCCESSOR,
    SWAP_CLOCK,
    AccountingBroker,
    MarginFractionError,
    commission_cash,
    compute_successor_order_size,
    configure_successor_broker,
    margin_required,
    swap_cash,
    units_within_margin,
    validate_margin_fraction,
)
from app.elapsed_hour_harness import _broker_frame, run_elapsed_hour_harness
from app.factorial_harness import (
    build_manifest,
    execute_manifest,
    family_input,
    scale_on_dev,
)
from app.plugins.plugin_long_short_predictions import Plugin
from app.policies.prediction_entry_exit import PredictionEntryExitParameters
from tests.unit_tests.test_strategy_micro_20260930 import (
    IDEAL_EXIT_STAMP,
    IDEAL_LONG_MIN_AT_EXIT,
    IDEAL_SHORT_AT_EXIT,
    OPEN_ORACLES,
    PERSIST_SHORT_AT_EXIT,
    PERSIST_SL_STAMP,
    SHORT_HORIZONS,
    LONG_HORIZONS,
    SUPPORT_ORIGINS,
    TRADE_CLOCKS,
    TURN_STOP,
    _arm_config,
    _event,
    _frame,
    _micro_bars,
    _ohlc,
    _probe_bars,
)

SUCCESSOR_CASH = 10_000.0
LEVERAGE = 100.0
RATE = 0.00007
BUDGET = SUCCESSOR_CASH * MARGIN_FRACTION_CAP
NAIVE_MAE = {
    "1": 0.006,
    "2": 0.01675,
    "3": 0.01375,
    "4": 0.0105,
    "5": 0.00875,
    "6": 0.008,
    "24": 0.00675,
    "48": 0.013,
    "72": 0.0075,
    "96": 0.0065,
    "120": 0.00675,
    "144": 0.00625,
}
NAIVE_MACRO = 0.009208333333333327
LEGACY_CITATION = "LEGACY docs/audits/evidence/STRATEGY_MICRO_20260930/MICRO.json"
PARENT = "781022a4702244d866cf86bb69343673340e9eff"


def _successor_config(short_family, long_family, **extra):
    config = _arm_config(short_family, long_family, **extra)
    config["accounting_convention"] = "successor"
    config["initial_cash"] = SUCCESSOR_CASH
    return config


def _family_mean(values, horizons):
    return float(sum(values[str(hours)] for hours in horizons) / len(horizons))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError("non-finite measured number")
        return value
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    raise AssertionError(f"unexpected measured type {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    lowered = encoded.lower()
    for token in ("/home", "harvey", "hostname", "postgres"):
        assert token not in lowered
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def _legacy_micro():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "audits"
        / "evidence"
        / "STRATEGY_MICRO_20260930"
        / "MICRO.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _feed(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    indexed.index = pd.to_datetime(indexed.index)
    if indexed.index.tz is None:
        indexed.index = indexed.index.tz_localize("UTC")
    return _broker_frame(indexed)


def _predictions(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "DATE_TIME"
    return frame


def _install(cerebro, params, broker_cls=None):
    rate = params["commission_per_lot"] / LOT_UNITS
    slip = (params["spread_pips"] + params["slippage_pips"]) * params["pip_cost"] / 2.0
    if broker_cls is None:
        return configure_successor_broker(
            cerebro,
            cash=SUCCESSOR_CASH,
            leverage=params["leverage"],
            margin_fraction=params["margin_fraction"],
            commission_rate=rate,
            min_order_volume=params["min_order_volume"],
            slippage_per_side=slip,
        )
    broker = broker_cls(
        margin_fraction=params["margin_fraction"],
        trading_leverage=params["leverage"],
        commission_rate=rate,
        min_order_volume=params["min_order_volume"],
    )
    cerebro.broker = broker
    broker.setcash(SUCCESSOR_CASH)
    broker.set_shortcash(False)
    broker.set_coc(False)
    import backtrader as bt

    broker.setcommission(
        commission=rate,
        margin=None,
        mult=1.0,
        commtype=bt.CommInfoBase.COMM_PERC,
        percabs=True,
        stocklike=True,
        leverage=float(params["leverage"]),
        automargin=False,
    )
    broker.set_slippage_fixed(slip, slip_open=True, slip_limit=True)
    return broker


def _run_book(tmp_path, bars, predictions, *, strategy_cls=None, broker_cls=None, checksubmit=True, name="book"):
    import backtrader as bt
    import matplotlib

    matplotlib.use("Agg", force=True)
    import app.plugins.plugin_long_short_predictions as plugin_module

    plugin_module._QUIET = True
    params = Plugin().params
    work = tmp_path / name
    work.mkdir()
    table = predictions.copy()
    table.index.name = "DATE_TIME"
    pred_path = work / "predictions.csv"
    table.reset_index().to_csv(pred_path, index=False)
    cerebro = bt.Cerebro()
    cls = strategy_cls or Plugin.HeuristicStrategy
    cerebro.addstrategy(
        cls,
        pred_file=str(pred_path),
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
        accounting_convention=SUCCESSOR,
        margin_fraction=params["margin_fraction"],
        commission_per_unit=params["commission_per_lot"] / LOT_UNITS,
        spread_pips=params["spread_pips"],
        slippage_pips=params["slippage_pips"],
    )
    cerebro.adddata(bt.feeds.PandasData(dataname=_feed(bars)))
    broker = _install(cerebro, params, broker_cls)
    if not checksubmit:
        broker.set_checksubmit(False)
    previous = os.getcwd()
    os.chdir(work)
    try:
        strategy = cerebro.run()[0]
    finally:
        os.chdir(previous)
    return strategy


def _states(strategy):
    return [event["state"] for event in strategy.order_events]


def _flat_book(strategy):
    book = strategy.accounting_snapshot["book"]
    cash = float(strategy.broker.getcash())
    equity = float(strategy.broker.getvalue())
    realized = float(sum(trade["pnl"] for trade in strategy.trades))
    assert strategy.position.size == pytest.approx(0.0)
    assert book["flat"] is True
    assert book["equity_gap"] == pytest.approx(0.0, abs=1e-6)
    assert book["flat_cash_gap"] == pytest.approx(0.0, abs=1e-6)
    assert cash == pytest.approx(equity, abs=1e-6)
    assert cash == pytest.approx(SUCCESSOR_CASH + realized, abs=1e-6)
    ledger = strategy.ledger
    commission = sum(row["amount"] for row in ledger if row["kind"] == "commission")
    swap = sum(row["amount"] for row in ledger if row["kind"] == "swap")
    assert commission == pytest.approx(sum(trade["commission"] for trade in strategy.trades))
    assert swap == pytest.approx(sum(trade["swap"] for trade in strategy.trades))
    for trade in strategy.trades:
        assert trade["pnl"] == pytest.approx(trade["gross_pnl"] - trade["commission"] - trade["swap"])
    for row in ledger:
        if row["kind"] in {"spread", "slippage"}:
            assert row["debited_to_cash"] is False
            assert row["included_in_fill_price"] is True
        if row["kind"] == "swap":
            assert row["cash_after"] == pytest.approx(row["cash_before"] - row["amount"])
    return book


class _RejectBroker(AccountingBroker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rejected = False

    def _execute(self, order, ago=None, price=None, cash=None, position=None, dtcoc=None):
        if ago is not None and not self._rejected:
            self._rejected = True
            order.reject()
            self.notify(order)
            self._ococheck(order)
            self._bracketize(order, cancel=True)
            return None
        return super()._execute(
            order, ago=ago, price=price, cash=cash, position=position, dtcoc=dtcoc
        )


class _OnceBuy(Plugin.HeuristicStrategy):
    def next(self):
        if getattr(self, "_sent", False):
            return
        self._sent = True
        self.buy(size=10_000)


class _CancelBuy(_OnceBuy):
    def next(self):
        if getattr(self, "_sent", False):
            return
        self._sent = True
        order = self.buy(size=10_000)
        self.cancel_accepted = bool(self.broker.cancel(order))


class _SpyClose(Plugin.HeuristicStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_cash = []

    def close(self, *args, **kwargs):
        self.decision_cash.append(float(self.broker.getcash()))
        return super().close(*args, **kwargs)


def test_margin_fraction_rejects_values_above_the_cap_and_bools():
    assert validate_margin_fraction(0.05) == pytest.approx(0.05)
    with pytest.raises(MarginFractionError):
        validate_margin_fraction(0.0500001)
    with pytest.raises(MarginFractionError):
        validate_margin_fraction(0.0)
    with pytest.raises(MarginFractionError):
        validate_margin_fraction(True)
    with pytest.raises(MarginFractionError):
        validate_margin_fraction(False)


def test_units_are_notional_over_price_not_a_price_blind_cap():
    assert units_within_margin(1.0, BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE) == 50_000
    assert units_within_margin(2.0, BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE) == 25_000
    slipped = units_within_margin(1.000015, BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE)
    assert slipped == 49_999
    assert margin_required(slipped, 1.000015, LEVERAGE) <= BUDGET + 1e-9
    assert margin_required(50_000, 1.000015, LEVERAGE) > BUDGET
    tight = units_within_margin(1.0, BUDGET, LEVERAGE, BUDGET, RATE)
    assert tight < 50_000
    assert margin_required(tight, 1.0, LEVERAGE) <= BUDGET + 1e-9
    assert margin_required(tight, 1.0, LEVERAGE) + commission_cash(tight, 1.0, RATE) <= BUDGET + 1e-9


def test_minimum_volume_cannot_raise_margin(tmp_path):
    params = PredictionEntryExitParameters()
    size = compute_successor_order_size(
        reward_risk_ratio=3.0,
        price=10.0,
        equity=SUCCESSOR_CASH,
        cash=SUCCESSOR_CASH,
        margin_fraction=0.05,
        leverage=LEVERAGE,
        commission_rate=RATE,
        params=params,
    )
    assert size == 0.0
    assert size < params.min_order_volume


def test_gap_fill_is_margin_and_leaves_no_position(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 10.0, 10.0),
        ]
    )
    predictions = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 10.0, "Prediction_d_1": 10.0}})
    strategy = _run_book(tmp_path, bars, predictions, name="gap")
    assert "Margin" in _states(strategy)
    assert "Completed" not in _states(strategy)
    assert strategy.position.size == pytest.approx(0.0)
    assert strategy.trades == []
    assert strategy.current_direction is None
    assert any(event["action"] == "margin_rejected" for event in strategy.broker.cap_events)
    rejected = next(event for event in strategy.broker.cap_events if event["action"] == "margin_rejected")
    assert rejected["allowed_units"] < Plugin().params["min_order_volume"]


def test_price_two_caps_units_before_costs(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 2.0, 2.0),
            _ohlc("2019-05-01 01:00", 2.0, 2.01),
        ]
    )
    predictions = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 2.04, "Prediction_d_1": 2.04}})
    strategy = _run_book(tmp_path, bars, predictions, name="price2")
    units = abs(float(strategy.position.size))
    entry = float(strategy.position.price)
    allowed = units_within_margin(entry, BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE)
    assert units == allowed
    assert units <= 25_000
    assert units != 50_000
    assert margin_required(units, entry, LEVERAGE) <= BUDGET + 1e-6
    book = strategy.accounting_snapshot["book"]
    assert book["equity_gap"] == pytest.approx(0.0, abs=1e-6)
    assert book["flat"] is False


def test_long_and_short_post_collateral_and_release_it(tmp_path):
    long_bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 1.01),
            _ohlc("2019-05-01 02:00", 1.01, 1.03),
            _ohlc("2019-05-01 03:00", 1.03, 1.03),
        ]
    )
    short_bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 0.99),
            _ohlc("2019-05-01 02:00", 0.99, 0.97),
            _ohlc("2019-05-01 03:00", 0.97, 0.97),
        ]
    )
    long_pred = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 1.03, "Prediction_d_1": 1.03}})
    short_pred = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 0.97, "Prediction_d_1": 0.97}})
    for name, bars, predictions in (
        ("long", long_bars, long_pred),
        ("short", short_bars, short_pred),
    ):
        strategy = _run_book(tmp_path, bars, predictions, name=name)
        entry = next(row for row in strategy.ledger if row["kind"] == "commission")
        units = abs(entry["units"])
        price = entry["price"]
        collateral = margin_required(units, price, LEVERAGE)
        posted = SUCCESSOR_CASH - collateral - entry["amount"]
        assert entry["cash_after"] == pytest.approx(posted, abs=1e-6)
        assert entry["cash_after"] < SUCCESSOR_CASH
        assert entry["cash_after"] > SUCCESSOR_CASH - units * price
        _flat_book(strategy)
        assert float(strategy.broker.getcash()) > entry["cash_after"]


def test_gapped_timestamps_are_not_counted_as_one_bar(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 1.002),
            _ohlc("2019-05-01 05:00", 1.002, 1.02),
        ]
    )
    predictions = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 1.02, "Prediction_d_1": 1.02}})
    strategy = _run_book(tmp_path, bars, predictions, strategy_cls=_SpyClose, name="clock")
    swaps = [row for row in strategy.ledger if row["kind"] == "swap"]
    assert len(swaps) == 1
    row = swaps[0]
    units = abs(float(strategy.position.size))
    assert row["hours"] == pytest.approx(4.0)
    assert row["amount"] == pytest.approx(swap_cash(units, 4.0, 10.0))
    assert row["amount"] != pytest.approx(swap_cash(units, 1.0, 10.0))
    assert row["cash_after"] == pytest.approx(row["cash_before"] - row["amount"])
    assert strategy.decision_cash
    assert strategy.decision_cash[0] == pytest.approx(row["cash_after"])
    assert strategy.position.size != pytest.approx(0.0)


def test_cancelled_order_does_not_fill(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 1.0),
        ]
    )
    predictions = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 1.0, "Prediction_d_1": 1.0}})
    strategy = _run_book(
        tmp_path, bars, predictions, strategy_cls=_CancelBuy, checksubmit=False, name="cancel"
    )
    assert strategy.cancel_accepted is True
    states = set(_states(strategy))
    assert {"in_flight", "Accepted", "Cancelled"} <= states
    assert "Completed" not in states
    assert strategy.position.size == pytest.approx(0.0)
    assert strategy.trades == []


def test_rejected_order_does_not_leave_a_position(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 1.0),
        ]
    )
    predictions = _predictions({"2019-05-01 00:00": {"Prediction_h_1": 1.0, "Prediction_d_1": 1.0}})
    strategy = _run_book(
        tmp_path, bars, predictions, strategy_cls=_OnceBuy, broker_cls=_RejectBroker, name="reject"
    )
    states = set(_states(strategy))
    assert {"in_flight", "Accepted", "Rejected"} <= states
    assert "Completed" not in states
    assert strategy.position.size == pytest.approx(0.0)
    assert strategy.trades == []
    assert strategy.current_direction is None


def test_harness_default_uses_the_successor_book(tmp_path):
    bars = _frame(
        [
            _ohlc("2019-05-01 00:00", 1.0, 1.0),
            _ohlc("2019-05-01 01:00", 1.0, 1.01),
            _ohlc("2019-05-01 02:00", 1.01, 1.02),
        ]
    )
    config = _arm_config(
        "ideal",
        "ideal",
        initial_cash=SUCCESSOR_CASH,
        short_horizons_hours=(1,),
        long_horizons_hours=(2,),
    )
    del config["accounting_convention"]
    result = run_elapsed_hour_harness(bars, config, work_directory=str(tmp_path / "default"))
    assert result["accounting_convention"] == SUCCESSOR
    assert result["initial_cash"] == SUCCESSOR_CASH
    assert result["stop_close_obtained_fill"] is False
    for fill in result["fills"]:
        assert abs(fill["size"]) != pytest.approx(1_000_000.0)
        assert margin_required(fill["size"], fill["price"], LEVERAGE) <= BUDGET * 2


def test_pending_position_survives_stop_inside_the_margin_cap(tmp_path):
    result = run_elapsed_hour_harness(
        _probe_bars(),
        _successor_config("ideal", "ideal", broker_session_end="2019-05-01 01:00"),
        work_directory=str(tmp_path / "probe"),
    )
    units = result["pending_position_units"]
    assert result["n_closed_trades"] == 0
    assert result["realized_pnl"] == pytest.approx(0.0)
    assert units != pytest.approx(0.0)
    assert units != pytest.approx(1_000_000.0)
    assert result["position_before_stop"] == pytest.approx(units)
    assert result["position_after_stop"] == pytest.approx(units)
    assert result["stop_close_obtained_fill"] is False
    assert result["closed_trades_before_stop"] == result["closed_trades_after_stop"] == 0
    fill = result["fills"][0]
    assert fill["price"] == pytest.approx(1.000015)
    assert abs(fill["size"]) == units_within_margin(fill["price"], BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE)
    assert margin_required(units, fill["price"], LEVERAGE) <= BUDGET + 1e-6
    book = result["accounting"]["book"]
    assert book["equity_gap"] == pytest.approx(0.0, abs=1e-6)
    assert book["flat"] is False
    assert book["unrealized_pnl"] == pytest.approx(units * (result["accounting"]["mark_price"] - fill["price"]))
    assert "in_flight" in {event["state"] for event in result["accounting"]["orders"]}
    assert result["broker_last"] == "2019-05-01T01:00:00+00:00"


def test_factorial_manifest_is_ready_and_does_not_run():
    origin = pd.Timestamp("2019-05-14 00:00", tz="UTC")
    index = pd.to_datetime(["2019-05-14 00:00", "2019-05-14 01:00", "2019-05-16 00:00"], utc=True)
    frame = pd.DataFrame({"CLOSE": [1.0, 1.2, 999.0]}, index=index)
    scale = scale_on_dev(frame, [1], source_timezone="UTC")
    mutated = frame.copy()
    mutated.iloc[-1, 0] = 5.0
    assert scale_on_dev(mutated, [1], source_timezone="UTC") == scale
    moved = frame.copy()
    moved.iloc[0, 0] = 1.5
    assert scale_on_dev(moved, [1], source_timezone="UTC") != scale
    manifest = build_manifest([origin], [1], scale)
    assert manifest["cell_count"] == 18
    assert manifest["executed"] is False
    assert manifest["status"] == "B0_NOT_STARTED"
    assert manifest["historic_ceiling_is_b0_budget"] is False
    assert manifest["retained_sweep_cells_not_run"] == 241
    assert manifest["holdout_decisions"] == "OUT_OF_SCOPE"
    assert manifest["b0_budget"] == "OUT_OF_SCOPE"
    assert manifest["parameters_not_chosen_by_pnl"] is True
    assert all(cell["executed"] is False for cell in manifest["cells"])
    paired = [
        cell
        for cell in manifest["cells"]
        if cell["input_kind"] == "mae_matched_noise" and cell["paired_seed"] == 42
    ]
    assert {cell["varying"] for cell in paired} == {"short", "long"}
    assert {cell["seed"] for cell in paired} == {42}
    future = [1.01, 0.99]
    noise = family_input(
        "mae_matched_noise", origin_price=1.0, future_prices=future, target_mae=0.01, seed=42
    )
    again = family_input(
        "mae_matched_noise", origin_price=1.0, future_prices=future, target_mae=0.01, seed=42
    )
    other = family_input(
        "mae_matched_noise", origin_price=1.0, future_prices=future, target_mae=0.01, seed=43
    )
    ideal = family_input("ideal", origin_price=1.0, future_prices=future, target_mae=0.01, seed=None)
    persistence = family_input(
        "persistence", origin_price=1.0, future_prices=future, target_mae=0.01, seed=None
    )
    assert noise == again
    assert noise != other
    assert ideal != persistence
    assert noise != persistence
    assert float(np.mean(np.abs(np.array(noise) - np.array(future)))) == pytest.approx(0.01)
    assert float(np.mean(np.abs(np.array(persistence) - np.array(future)))) == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="B0_NOT_STARTED"):
        execute_manifest(manifest)
    with pytest.raises(ValueError):
        build_manifest([pd.Timestamp("2019-05-16 00:00", tz="UTC")], [1], scale)


def _arm_summary(result):
    book = result["accounting"]["book"]
    ledger = result["accounting"]["ledger"]
    return {
        "initial_cash": result["initial_cash"],
        "margin_fraction": result["accounting"]["margin_fraction"],
        "leverage": result["accounting"]["leverage"],
        "n_closed_trades": result["n_closed_trades"],
        "realized_pnl": result["realized_pnl"],
        "cash": result["cash"],
        "marked_equity": result["marked_equity"],
        "pending_position_units": result["pending_position_units"],
        "entry_sides": result["entry_sides"],
        "mae_macro_average": result["mae_macro_average"],
        "naive_macro_average": result["naive_macro_average"],
        "short_family_mae": _family_mean(result["per_horizon_mae"], SHORT_HORIZONS),
        "long_family_mae": _family_mean(result["per_horizon_mae"], LONG_HORIZONS),
        "per_horizon_mae": result["per_horizon_mae"],
        "per_horizon_naive_mae": result["per_horizon_naive_mae"],
        "fills": result["fills"],
        "trades": result["trades"],
        "flat_cash_gap": book["flat_cash_gap"],
        "equity_gap": book["equity_gap"],
        "collateral": book["collateral"],
        "unrealized_pnl": book["unrealized_pnl"],
        "order_states": sorted({event["state"] for event in result["accounting"]["orders"]}),
        "cap_events": result["accounting"]["cap_events"],
        "cost_sums": {
            kind: float(sum(row["amount"] for row in ledger if row["kind"] == kind))
            for kind in ("commission", "spread", "slippage", "swap")
        },
    }


def _fill_difference(measured, legacy_fills):
    rows = []
    width = max(len(measured), len(legacy_fills))
    for index in range(width):
        successor = measured[index] if index < len(measured) else None
        legacy = legacy_fills[index] if index < len(legacy_fills) else None
        row = {"index": index, "successor": successor, "legacy": legacy}
        if successor is not None and legacy is not None:
            row["same_clock"] = successor["datetime"] == legacy["datetime"]
            row["same_side"] = successor["is_buy"] == legacy["is_buy"]
            row["price_delta"] = successor["price"] - legacy["price"]
            row["size_delta"] = successor["size"] - legacy["size"]
        rows.append(row)
    return {
        "legacy_citation": "LEGACY",
        "successor_count": len(measured),
        "legacy_count": len(legacy_fills),
        "rows": rows,
    }


def test_successor_arms_on_the_legacy_support(tmp_path):
    legacy = _legacy_micro()
    bars = _micro_bars()
    arms = {}
    for short_family, long_family in (
        ("ideal", "ideal"),
        ("persistence", "ideal"),
        ("ideal", "persistence"),
        ("persistence", "persistence"),
    ):
        pair = f"{short_family}/{long_family}"
        arms[pair] = run_elapsed_hour_harness(
            bars,
            _successor_config(short_family, long_family),
            work_directory=str(tmp_path / pair.replace("/", "_")),
        )

    ideal = arms["ideal/ideal"]
    assert ideal["support_origins"] == list(SUPPORT_ORIGINS)
    assert ideal["initial_cash"] == SUCCESSOR_CASH
    assert ideal["accounting_convention"] == SUCCESSOR
    assert ideal["exit_variant"] == "E"
    assert COMMISSION_AUDIT["flat_fee_of_7_currency_units"] is False
    assert COMMISSION_AUDIT["per_side"] is True
    assert COMMISSION_AUDIT["broker_rate_value"] == pytest.approx(RATE)

    for pair, result in arms.items():
        assert tuple(result["entry_sides"]) == OPEN_ORACLES[pair]
        assert result["support_origins"] == list(SUPPORT_ORIGINS)
        assert result["per_horizon_naive_mae"].keys() == NAIVE_MAE.keys()
        for hours, expected in NAIVE_MAE.items():
            assert result["per_horizon_naive_mae"][hours] == pytest.approx(expected)
            assert result["per_horizon_naive_mae"][hours] == pytest.approx(
                legacy["arms"][pair]["per_horizon_naive_mae"][hours]
            )
        assert result["naive_macro_average"] == pytest.approx(NAIVE_MACRO)
        assert result["per_horizon_mae"] == pytest.approx(legacy["arms"][pair]["per_horizon_mae"])
        for event in result["accounting"]["cap_events"]:
            if "margin_used" in event:
                assert event["margin_used"] <= event["margin_budget"] + 1e-6
                assert event["margin_used"] == pytest.approx(
                    margin_required(
                        event["allowed_units"]
                        if event["action"] == "capped"
                        else event["requested_units"],
                        event["price"],
                        LEVERAGE,
                    )
                )
        assert all(abs(fill["size"]) != pytest.approx(1_000_000.0) for fill in result["fills"])

    for pair in ("ideal/ideal", "persistence/ideal"):
        result = arms[pair]
        legacy_fills = legacy["arms"][pair]["fills"]
        assert len(result["fills"]) == len(legacy_fills)
        for measured, previous in zip(result["fills"], legacy_fills):
            assert measured["datetime"] == previous["datetime"]
            assert measured["is_buy"] == previous["is_buy"]
            assert measured["price"] == pytest.approx(previous["price"])
            assert measured["size"] != pytest.approx(previous["size"])
        for trade, clock in zip(result["trades"], TRADE_CLOCKS[pair]):
            assert trade["open_dt"] == clock[0]
            assert trade["close_dt"] == clock[1]
        book = result["accounting"]["book"]
        assert book["flat"] is True
        assert book["flat_cash_gap"] == pytest.approx(0.0, abs=1e-6)
        assert book["equity_gap"] == pytest.approx(0.0, abs=1e-6)
        assert result["cash"] == pytest.approx(result["marked_equity"], abs=1e-6)
        assert result["cash"] == pytest.approx(SUCCESSOR_CASH + result["realized_pnl"], abs=1e-6)
        ledger = result["accounting"]["ledger"]
        assert sum(row["amount"] for row in ledger if row["kind"] == "commission") == pytest.approx(
            sum(trade["commission"] for trade in result["trades"])
        )
        assert sum(row["amount"] for row in ledger if row["kind"] == "swap") == pytest.approx(
            sum(trade["swap"] for trade in result["trades"])
        )
        assert all(
            row["debited_to_cash"] is False
            for row in ledger
            if row["kind"] in {"spread", "slippage"}
        )
        first = result["fills"][0]
        assert first["price"] == pytest.approx(1.000015)
        assert abs(first["size"]) == units_within_margin(first["price"], BUDGET, LEVERAGE, SUCCESSOR_CASH, RATE)
        commission = next(row["amount"] for row in ledger if row["kind"] == "commission")
        assert commission == pytest.approx(commission_cash(first["size"], first["price"], RATE))
        assert commission != pytest.approx(7.0)

    assert all(value == pytest.approx(0.0) for value in ideal["per_horizon_mae"].values())
    flat = arms["persistence/persistence"]
    assert flat["per_horizon_mae"] == pytest.approx(flat["per_horizon_naive_mae"])
    assert flat["n_closed_trades"] == 0
    assert flat["realized_pnl"] == pytest.approx(0.0)
    assert flat["marked_equity"] == pytest.approx(SUCCESSOR_CASH)
    assert arms["ideal/persistence"]["n_closed_trades"] == 0
    assert arms["ideal/persistence"]["realized_pnl"] == pytest.approx(0.0)

    exit_event = _event(ideal, IDEAL_EXIT_STAMP)
    assert exit_event["request"] == "close"
    assert exit_event["position_before"] != pytest.approx(0.0)
    assert exit_event["position_before"] != pytest.approx(1_000_000.0)
    assert min(exit_event["predictions"]["long"]) == pytest.approx(IDEAL_LONG_MIN_AT_EXIT)
    assert exit_event["predictions"]["short"] == pytest.approx(IDEAL_SHORT_AT_EXIT)
    blend = 0.6 * min(exit_event["predictions"]["short"]) + 0.4 * min(exit_event["predictions"]["long"])
    assert blend < TURN_STOP
    mixed = arms["persistence/ideal"]
    persist_exit = _event(mixed, IDEAL_EXIT_STAMP)
    assert persist_exit["request"] == "none"
    assert persist_exit["predictions"]["short"] == pytest.approx([PERSIST_SHORT_AT_EXIT] * 6)
    assert _event(mixed, PERSIST_SL_STAMP)["request"] == "close"
    assert _event(arms["ideal/persistence"], "2019-05-01T00:00:00+00:00")["request"] == "none"

    probe = run_elapsed_hour_harness(
        _probe_bars(),
        _successor_config("ideal", "ideal", broker_session_end="2019-05-01 01:00"),
        work_directory=str(tmp_path / "probe"),
    )
    assert probe["stop_close_obtained_fill"] is False
    assert probe["pending_position_units"] != pytest.approx(1_000_000.0)
    assert probe["n_closed_trades"] == 0

    evidence = Path(__file__).resolve().parents[2] / "docs" / "audits" / "evidence" / "STRATEGY_ACCOUNTING_20260930"
    summaries = {pair: _arm_summary(result) for pair, result in arms.items()}
    differences = {
        pair: _fill_difference(result["fills"], legacy["arms"][pair]["fills"])
        for pair, result in arms.items()
    }
    origin = pd.Timestamp("2019-05-14 00:00", tz="UTC")
    index = pd.to_datetime(["2019-05-14 00:00", "2019-05-14 01:00", "2019-05-16 00:00"], utc=True)
    scale = scale_on_dev(pd.DataFrame({"CLOSE": [1.0, 1.2, 999.0]}, index=index), [1], source_timezone="UTC")
    manifest = build_manifest([origin], [1], scale)
    ran = False
    try:
        execute_manifest(manifest)
        ran = True
    except RuntimeError as exc:
        factorial_error = str(exc)
    assert ran is False
    assert "B0_NOT_STARTED" in factorial_error

    micro = {
        "label": "SYNTHETIC",
        "phase": "MEASURED",
        "experiment": "successor_accounting_variant_E",
        "parent_commit": PARENT,
        "legacy_evidence": LEGACY_CITATION,
        "initial_cash": SUCCESSOR_CASH,
        "margin_fraction": MARGIN_FRACTION_CAP,
        "leverage": LEVERAGE,
        "arms": summaries,
        "fill_differences_versus_legacy": differences,
        "terminal_probe": _arm_summary(probe),
        "factorial_executed": False,
        "b0_status": "B0_NOT_STARTED",
    }
    post = {
        "label": "SYNTHETIC",
        "phase": "MEASURED",
        "experiment": "successor_accounting_variant_E",
        "parent_commit": PARENT,
        "legacy_evidence": LEGACY_CITATION,
        "account_currency": "quote",
        "notional": "units * price",
        "margin": "notional / leverage",
        "margin_fraction": MARGIN_FRACTION_CAP,
        "leverage": LEVERAGE,
        "initial_cash": SUCCESSOR_CASH,
        "swap_clock": SWAP_CLOCK,
        "commission_audit": COMMISSION_AUDIT,
        "commission_is_per_side_notional_rate": True,
        "flat_fee_of_7_currency_units": False,
        "spread_and_slippage_included_in_fill_price": True,
        "short_cash_credit_is_not_profit": True,
        "stop_does_not_fabricate_a_fill": True,
        "arms": {
            pair: {
                "n_closed_trades": row["n_closed_trades"],
                "realized_pnl": row["realized_pnl"],
                "cash": row["cash"],
                "marked_equity": row["marked_equity"],
                "pending_position_units": row["pending_position_units"],
                "mae_macro_average": row["mae_macro_average"],
                "short_family_mae": row["short_family_mae"],
                "long_family_mae": row["long_family_mae"],
                "naive_macro_average": row["naive_macro_average"],
                "entry_sides": row["entry_sides"],
                "fills": row["fills"],
                "flat_cash_gap": row["flat_cash_gap"],
                "equity_gap": row["equity_gap"],
                "cost_sums": row["cost_sums"],
                "order_states": row["order_states"],
                "position_at_0900": (
                    None
                    if pair in {"ideal/persistence", "persistence/persistence"}
                    else _event(arms[pair], IDEAL_EXIT_STAMP)["position_before"]
                ),
            }
            for pair, row in summaries.items()
        },
        "fill_differences_versus_legacy": differences,
        "terminal_probe": {
            "pending_position_units": probe["pending_position_units"],
            "fill_price": probe["fills"][0]["price"],
            "fill_size": probe["fills"][0]["size"],
            "n_closed_trades": probe["n_closed_trades"],
            "realized_pnl": probe["realized_pnl"],
            "cash": probe["cash"],
            "marked_equity": probe["marked_equity"],
            "stop_close_obtained_fill": probe["stop_close_obtained_fill"],
            "equity_gap": probe["accounting"]["book"]["equity_gap"],
        },
        "naive_mae_by_horizon": NAIVE_MAE,
        "factorial": {
            "executed": False,
            "cell_count": manifest["cell_count"],
            "status": manifest["status"],
            "error": factorial_error,
            "historic_ceiling_is_b0_budget": False,
            "retained_sweep_cells_not_run": 241,
            "holdout_decisions": "OUT_OF_SCOPE",
            "b0_budget": "OUT_OF_SCOPE",
        },
        "b0_status": "B0_NOT_STARTED",
        "market_b0_not_run": True,
        "cell_sweep_not_run": True,
    }
    _write_json(evidence / "MICRO.json", micro)
    _write_json(evidence / "POST.json", post)
