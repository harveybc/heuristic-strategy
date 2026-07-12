from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd
from trading_contracts import (
    PredictionBundle,
    PredictionHorizon,
    ProducerIdentity,
)


@dataclass(frozen=True)
class PredictionPaths:
    short: tuple[float, ...]
    long: tuple[float, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        short: Sequence[float],
        long: Sequence[float],
    ) -> "PredictionPaths":
        return cls(
            short=tuple(float(value) for value in short),
            long=tuple(float(value) for value in long),
        )


@runtime_checkable
class PredictionPathSource(Protocol):
    source_id: str

    def get_paths(self, as_of: datetime) -> PredictionPaths | None:
        ...


class MappingPredictionPathSource:
    """Deterministic source for fixtures, ideal controls, and cached paths."""

    def __init__(
        self,
        source_id: str,
        paths_by_time: Mapping[datetime, PredictionPaths],
    ) -> None:
        self.source_id = source_id
        self._paths_by_time = dict(paths_by_time)

    def get_paths(self, as_of: datetime) -> PredictionPaths | None:
        return self._paths_by_time.get(as_of)


class CsvPredictionPathSource:
    """Adapter for the legacy `Prediction_h_*` / `Prediction_d_*` format."""

    def __init__(self, source_id: str, frame: pd.DataFrame) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("prediction CSV frame must use a DatetimeIndex")
        self.source_id = source_id
        self._frame = frame.copy(deep=True)
        self._short_columns = self._ordered_columns("Prediction_h_")
        self._long_columns = self._ordered_columns("Prediction_d_")

    @classmethod
    def from_csv(cls, source_id: str, path: str) -> "CsvPredictionPathSource":
        frame = pd.read_csv(path, parse_dates=["DATE_TIME"])
        return cls(source_id, frame.set_index("DATE_TIME"))

    def _ordered_columns(self, prefix: str) -> list[str]:
        def suffix(column: str) -> int:
            try:
                return int(column.removeprefix(prefix))
            except ValueError as exc:
                raise ValueError(f"invalid prediction column: {column}") from exc

        return sorted(
            (column for column in self._frame.columns if column.startswith(prefix)),
            key=suffix,
        )

    def get_paths(self, as_of: datetime) -> PredictionPaths | None:
        timestamp = pd.Timestamp(as_of)
        if timestamp not in self._frame.index:
            return None
        row = self._frame.loc[timestamp]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"duplicate prediction timestamp: {timestamp.isoformat()}")
        short = [row[column] for column in self._short_columns]
        long = [row[column] for column in self._long_columns]
        if not short and not long:
            return None
        return PredictionPaths.from_sequences(short=short, long=long)


class CallablePredictionPathSource:
    """Transport-neutral adapter for direct inference or provider clients."""

    def __init__(
        self,
        source_id: str,
        fetch: Callable[[datetime], PredictionPaths | None],
    ) -> None:
        self.source_id = source_id
        self._fetch = fetch

    def get_paths(self, as_of: datetime) -> PredictionPaths | None:
        return self._fetch(as_of)


def build_prediction_bundle(
    source: PredictionPathSource,
    *,
    as_of: datetime,
    asset_id: str,
    producer: ProducerIdentity,
    trace_id: str,
    config_hash: str,
    model_artifact_hash: str,
) -> PredictionBundle | None:
    paths = source.get_paths(as_of)
    if paths is None:
        return None

    def horizon(values: tuple[float, ...]) -> PredictionHorizon:
        return PredictionHorizon(
            outputs={
                f"path_{index:02d}": value
                for index, value in enumerate(values, start=1)
            },
            output_schema="price_path.v1",
        )

    return PredictionBundle(
        object_id=f"{source.source_id}:{asset_id}:{as_of.isoformat()}",
        as_of=as_of,
        producer=producer,
        trace_id=trace_id,
        config_hash=config_hash,
        asset_id=asset_id,
        horizons={
            "short": horizon(paths.short),
            "long": horizon(paths.long),
        },
        model_artifact_hash=model_artifact_hash,
    )
