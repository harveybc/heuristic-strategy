from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from trading_contracts import (
    DecisionContext,
    MarketSnapshot,
    PositionState,
    ProducerIdentity,
)

from app.policies import (
    CallablePredictionPathSource,
    CsvPredictionPathSource,
    MappingPredictionPathSource,
    PredictionEntryExitParameters,
    PredictionEntryExitPolicy,
    PredictionPaths,
    build_prediction_bundle,
)


NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
PRODUCER = ProducerIdentity(name="source-parity", version="1.0.0")
PATHS = PredictionPaths.from_sequences(
    short=[1.101, 1.102],
    long=[1.09, 1.12],
)


def _context(bundle) -> DecisionContext:
    market = MarketSnapshot(
        object_id="market-source-parity",
        as_of=NOW,
        producer=PRODUCER,
        trace_id="source-parity-trace",
        config_hash=HASH_A,
        asset_id="fx:EUR/USD",
        timeframe="1h",
        local_features={"close": 1.1, "trades_last_5d": 0},
        data_manifest_hash=HASH_A,
        feature_manifest_hash=HASH_B,
    )
    return DecisionContext(
        object_id="decision-source-parity",
        as_of=NOW,
        producer=PRODUCER,
        trace_id="source-parity-trace",
        config_hash=HASH_A,
        cell_id="fx:EUR/USD@1h:predictions-v1:heuristic",
        market=market,
        predictions=[bundle],
        position=PositionState(side="flat", units=0.0),
    )


def _semantic_projection(intent) -> dict:
    return {
        "action": intent.action.value,
        "target_exposure": intent.target_exposure,
        "strategy_rel_volume": intent.strategy_rel_volume,
        "stop_price": intent.risk_geometry.stop_price,
        "take_profit_price": intent.risk_geometry.take_profit_price,
        "reason_codes": intent.reason_codes,
    }


def test_identical_paths_are_source_substitution_invariant() -> None:
    frame = pd.DataFrame(
        {
            "Prediction_h_1": [PATHS.short[0]],
            "Prediction_h_2": [PATHS.short[1]],
            "Prediction_d_1": [PATHS.long[0]],
            "Prediction_d_2": [PATHS.long[1]],
        },
        index=pd.DatetimeIndex([NOW], name="DATE_TIME"),
    )
    sources = [
        MappingPredictionPathSource("ideal", {NOW: PATHS}),
        CsvPredictionPathSource("csv", frame),
        CallablePredictionPathSource("direct-model", lambda as_of: PATHS),
        CallablePredictionPathSource("provider-api", lambda as_of: PATHS),
    ]
    policy = PredictionEntryExitPolicy(
        PredictionEntryExitParameters(pip_cost=0.0001),
        artifact_hash=HASH_B,
        producer=PRODUCER,
    )

    projections = []
    for source in sources:
        bundle = build_prediction_bundle(
            source,
            as_of=NOW,
            asset_id="fx:EUR/USD",
            producer=PRODUCER,
            trace_id="source-parity-trace",
            config_hash=HASH_A,
            model_artifact_hash=HASH_B,
        )
        assert bundle is not None
        projections.append(_semantic_projection(policy.decide(_context(bundle))))

    assert projections[1:] == projections[:-1]


def test_missing_timestamp_produces_no_bundle() -> None:
    source = MappingPredictionPathSource("empty", {})
    assert build_prediction_bundle(
        source,
        as_of=NOW,
        asset_id="fx:EUR/USD",
        producer=PRODUCER,
        trace_id="source-parity-trace",
        config_hash=HASH_A,
        model_artifact_hash=HASH_B,
    ) is None
