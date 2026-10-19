"""Source-selectable OHLC loading: ``lake_label`` (legacy default), ``admitted`` or ``lake_mapped``.

``load_ohlc(path, start, end)`` with no ``source`` delegates to ``data.load_ohlc`` unchanged, so the
legacy frame is byte-identical. The two new sources return the same shape (OPEN/HIGH/LOW/CLOSE indexed
by ``DATE_TIME``) but the index is a true UTC bar START:

* ``admitted``     the repaired admitted root (``timestamp`` = UTC bar start, Bid OHLC);
* ``lake_mapped``  the lake labels mapped through a declared clock identity (``clock_identity``).

Both refuse any window reaching TEST (>= 2025-01-01 UTC) before reading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.short_horizon import clock_identity as CI
from app.short_horizon import data as D

SOURCES = ("lake_label", "admitted", "lake_mapped")
DEFAULT_SOURCE = "lake_label"


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) and frame.index.max() >= D.TEST_START:
        raise D.TestRowsError("TEST rows present after filtering")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("bar starts must be unique and increasing")
    if ((frame.index.minute != 0) | (frame.index.second != 0)).any():
        raise ValueError("only whole-hour bar starts are supported")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("non-finite OHLC")
    if ((frame.HIGH < frame[["OPEN", "CLOSE"]].max(axis=1)) |
            (frame.LOW > frame[["OPEN", "CLOSE"]].min(axis=1))).any():
        raise ValueError("incoherent OHLC range")
    return frame


def load_ohlc(path, start, end_exclusive, *, source: str = DEFAULT_SOURCE, clock_rule=None,
              transition_policy=None) -> pd.DataFrame:
    if source == "lake_label":
        return D.load_ohlc(path, start, end_exclusive)
    if source == "admitted":
        frame = CI.read_admitted(path, start, end_exclusive)[CI.OHLC]
    elif source == "lake_mapped":
        if clock_rule is None:
            raise ValueError("lake_mapped needs a declared clock_rule")
        pol = transition_policy or {"nonexistent": "raise", "ambiguous": "raise"}
        frame = CI.map_lake(path, start, end_exclusive, clock_rule, **pol)[CI.OHLC]
    else:
        raise ValueError(f"unknown bar source {source!r}; expected one of {SOURCES}")
    frame = frame.copy()
    frame.index = frame.index.rename("DATE_TIME")
    return _validate(frame)


def resolve_path(template: str, roots=None) -> str:
    return CI._resolve_path(template, roots)


def load_ohlc_from_config(cfg, start, end_exclusive, *, source: str | None = None, roots=None,
                          repo_root=None) -> pd.DataFrame:
    """Load through a bar-sources config (dict or JSON path). Pins are checked when the spec carries them."""
    if not isinstance(cfg, dict):
        cfg = json.loads(Path(cfg).read_text())
    src = source or cfg.get("default_source", DEFAULT_SOURCE)
    if src not in cfg["sources"]:
        raise ValueError(f"source {src!r} not declared in the config")
    spec = cfg["sources"][src]
    path = resolve_path(spec["path"], roots)
    if spec.get("sha256") and D.file_sha256(path) != spec["sha256"]:
        raise ValueError(f"SOURCE_DIGEST: {spec['path']} differs from its pin")
    if src == "lake_mapped" and spec.get("clock_identity"):
        base = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        rec = json.loads((base / spec["clock_identity"]).read_text())
        body = {k: v for k, v in rec.items() if k != "identity_sha256"}
        if CI.canonical_sha256(body) != rec["identity_sha256"] or rec["identity_sha256"] != spec.get("clock_identity_sha256"):
            raise ValueError("CLOCK_IDENTITY_DIGEST: the pinned clock identity does not verify")
        if rec["rule"]["id"] != spec["clock_rule"] or rec["source"]["sha256"] != spec.get("sha256"):
            raise ValueError("CLOCK_IDENTITY_MISMATCH: config rule/source differ from the pinned identity")
    return load_ohlc(path, start, end_exclusive, source=src, clock_rule=spec.get("clock_rule"),
                     transition_policy=spec.get("transition_policy"))
