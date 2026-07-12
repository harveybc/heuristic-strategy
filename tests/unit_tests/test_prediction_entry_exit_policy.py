from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from trading_contracts import (
    DecisionContext,
    MarketSnapshot,
    PositionState,
    PredictionBundle,
    PredictionHorizon,
    ProducerIdentity,
)

from app.policies.prediction_entry_exit import (
    PredictionEntryExitParameters,
    PredictionEntryExitPolicy,
    calculate_entry_geometry,
    compute_legacy_order_size,
    should_early_close,
)


FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "prediction_entry_exit_v1_fixture.json").read_text(
        encoding="utf-8"
    )
)
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
PRODUCER = ProducerIdentity(name="policy-test", version="1.0.0")


def _params(**overrides) -> PredictionEntryExitParameters:
    values = {
        "pip_cost": 0.0001,
        "rel_volume": 0.02,
        "min_order_volume": 10_000.0,
        "max_order_volume": 1_000_000.0,
        "leverage": 100.0,
        "profit_threshold": 5.0,
        "min_drawdown_pips": 10.0,
        "tp_multiplier": 0.9,
        "sl_multiplier": 2.0,
        "lower_rr_threshold": 0.5,
        "upper_rr_threshold": 2.0,
        "max_trades_per_5days": 3,
        "exit_variant": "E",
    }
    values.update(overrides)
    return PredictionEntryExitParameters(**values)


@pytest.mark.parametrize("case", FIXTURE["entry_cases"], ids=lambda case: case["name"])
def test_frozen_entry_geometry(case: dict) -> None:
    result = calculate_entry_geometry(
        current_price=case["current_price"],
        long_horizon_predictions=case["predictions"],
        params=_params(),
    )
    assert result is not None
    expected = case["expected"]
    assert result.direction == expected["direction"]
    assert result.take_profit_price == pytest.approx(expected["take_profit_price"])
    assert result.stop_loss_price == pytest.approx(expected["stop_loss_price"])
    assert result.reward_risk_ratio == pytest.approx(expected["reward_risk_ratio"])


@pytest.mark.parametrize("case", FIXTURE["early_close_cases"])
def test_frozen_early_close_variants(case: dict) -> None:
    assert should_early_close(
        direction=case["direction"],
        variant=case["variant"],
        short_horizon_predictions=case["short"],
        long_horizon_predictions=case["long"],
        stop_loss_price=case["stop"],
        entry_price=case["entry"],
    ) is case["expected"]


@pytest.mark.parametrize("case", FIXTURE["size_cases"])
def test_frozen_legacy_sizing(case: dict) -> None:
    assert compute_legacy_order_size(
        reward_risk_ratio=case["rr"],
        available_cash=case["cash"],
        params=_params(),
    ) == pytest.approx(case["expected"])


def _context(*, side: str = "flat", close: float = 1.1, trades: int = 0) -> DecisionContext:
    market = MarketSnapshot(
        object_id="market-001",
        as_of=NOW,
        producer=PRODUCER,
        trace_id="trace-001",
        config_hash=HASH_A,
        asset_id="fx:EUR/USD",
        timeframe="1h",
        local_features={"close": close, "trades_last_5d": trades},
        data_manifest_hash=HASH_A,
        feature_manifest_hash=HASH_B,
    )
    prediction = PredictionBundle(
        object_id="prediction-001",
        as_of=NOW,
        producer=PRODUCER,
        trace_id="trace-001",
        config_hash=HASH_A,
        asset_id="fx:EUR/USD",
        horizons={
            "short": PredictionHorizon(
                outputs={"path_01": 1.101, "path_02": 1.102},
                output_schema="price_path.v1",
            ),
            "long": PredictionHorizon(
                outputs={"path_01": 1.09, "path_02": 1.12},
                output_schema="price_path.v1",
            ),
        },
        model_artifact_hash=HASH_B,
    )
    return DecisionContext(
        object_id="decision-001",
        as_of=NOW,
        producer=PRODUCER,
        trace_id="trace-001",
        config_hash=HASH_A,
        cell_id="fx:EUR/USD@1h:predictions-v1:heuristic",
        market=market,
        predictions=[prediction],
        position=PositionState(
            side=side,
            units=1.0 if side != "flat" else 0.0,
            average_price=1.1 if side != "flat" else None,
            stop_loss=1.08 if side == "long" else (1.12 if side == "short" else None),
            take_profit=1.12 if side == "long" else (1.08 if side == "short" else None),
        ),
    )


def test_policy_emits_asset_intent_without_broker_units() -> None:
    policy = PredictionEntryExitPolicy(_params(), artifact_hash=HASH_B, producer=PRODUCER)
    intent = policy.decide(_context())
    assert intent.action.value == "target"
    assert intent.target_exposure == 1.0
    assert intent.strategy_rel_volume == 0.02
    assert intent.risk_geometry is not None
    assert intent.risk_geometry.take_profit_price == pytest.approx(1.118)
    assert intent.risk_geometry.stop_price == pytest.approx(1.08)
    assert not hasattr(intent, "units")


def test_policy_frequency_gate_and_position_management() -> None:
    policy = PredictionEntryExitPolicy(_params(), artifact_hash=HASH_B, producer=PRODUCER)
    assert policy.decide(_context(trades=3)).reason_codes == ["frequency_limit"]
    assert policy.decide(_context(side="long", close=1.12)).reason_codes == ["take_profit_hit"]
    assert policy.decide(_context(side="long", close=1.08)).reason_codes == ["stop_loss_hit"]


def test_pure_policy_module_does_not_import_backtrader() -> None:
    code = (
        "import sys; "
        "import app.policies; "
        "assert 'backtrader' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_backtrader_adapter_delegates_to_frozen_core() -> None:
    from app.plugins.plugin_long_short_predictions import Plugin

    params = _params()
    dummy = SimpleNamespace(
        params=SimpleNamespace(**params.__dict__),
        exit_variant=params.exit_variant,
        broker=SimpleNamespace(getcash=lambda: 10_000.0),
    )
    dummy._policy_params = lambda: Plugin.HeuristicStrategy._policy_params(dummy)
    size = Plugin.HeuristicStrategy.compute_size(dummy, 1.25)
    assert size == pytest.approx(20_000.0)
    close = Plugin.HeuristicStrategy._should_early_close_long(
        dummy,
        "D",
        [0.98],
        [0.97],
        0.99,
        1.0,
    )
    assert close is True
