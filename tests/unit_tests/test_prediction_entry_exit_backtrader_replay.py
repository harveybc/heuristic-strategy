from __future__ import annotations

import json
from pathlib import Path

import backtrader as bt
import pandas as pd

from app.plugins import plugin_long_short_predictions


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "prediction_entry_exit_backtrader_replay_v1.json"
)


def _rounded(value):
    if value is None:
        return None
    return round(float(value), 12)


def _run_replay(tmp_path: Path, fixture: dict) -> list[dict]:
    predictions = []
    for row in fixture["predictions"]:
        predictions.append(
            {
                "DATE_TIME": row["timestamp"],
                "Prediction_h_1": row["hourly"][0],
                "Prediction_h_2": row["hourly"][1],
                "Prediction_d_1": row["daily"][0],
                "Prediction_d_2": row["daily"][1],
            }
        )
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)

    bars = pd.DataFrame(fixture["bars"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp")
    bars["volume"] = 0.0
    bars["openinterest"] = 0.0

    class RecordingStrategy(
        plugin_long_short_predictions.Plugin.HeuristicStrategy
    ):
        def __init__(self, *args, **kwargs):
            self.replay_events = []
            self._requested_action = None
            super().__init__(*args, **kwargs)

        def buy(self, *args, **kwargs):
            size = kwargs.get("size", args[0] if args else None)
            self._requested_action = ("buy", size)
            return super().buy(*args, **kwargs)

        def sell(self, *args, **kwargs):
            size = kwargs.get("size", args[0] if args else None)
            self._requested_action = ("sell", size)
            return super().sell(*args, **kwargs)

        def close(self, *args, **kwargs):
            result = super().close(*args, **kwargs)
            self._requested_action = ("close", abs(float(self.position.size)))
            return result

        def next(self):
            self._requested_action = None
            timestamp = self.data0.datetime.datetime(0).isoformat()
            close = _rounded(self.data0.close[0])
            position_before = _rounded(self.position.size)

            super().next()

            action, size = self._requested_action or ("none", None)
            self.replay_events.append(
                {
                    "timestamp": timestamp,
                    "close": close,
                    "position_before": position_before,
                    "request": action,
                    "size": _rounded(size),
                    "direction_after": self.current_direction,
                    "take_profit_after": _rounded(self.current_tp),
                    "stop_loss_after": _rounded(self.current_sl),
                }
            )

        def stop(self):
            # The legacy stop hook renders a balance plot. It does not affect
            # requested actions and is intentionally excluded from this replay.
            return None

    plugin_long_short_predictions._QUIET = True
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        RecordingStrategy,
        pred_file=str(prediction_path),
        **fixture["parameters"],
    )
    cerebro.adddata(bt.feeds.PandasData(dataname=bars))
    cerebro.broker.setcash(10_000.0)
    return cerebro.run()[0].replay_events


def test_frozen_backtrader_requested_action_replay(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert _run_replay(tmp_path, fixture) == fixture["expected_events"]
