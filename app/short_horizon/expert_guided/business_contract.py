"""The shared business contract (owner decision 2026-10-29: Musashi's contract adopted wholesale), vendored byte for byte
from agent-multi master 82862801 docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json and pinned by digest.
Nothing here changes the audited engine defaults: the contract is applied only through the declared execution
variant FULL_F4 (c3.EXEC_VARIANTS["full_f4"]) when a config selects it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUSINESS_CONTRACT_PATH = ROOT / "docs/short_horizon/clean_round/contracts/BUSINESS_CONTRACT.json"
BUSINESS_CONTRACT_SHA256 = "73f32e43005da04ae44bcfffb576624513e8040aa8df3f06c521be5b7495b036"   # agent-multi master 82862801
BUSINESS_CONTRACT_SOURCE = {"repo": "agent-multi", "commit": "82862801", "path": "docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json"}


class BusinessContractError(RuntimeError):
    pass


def load(path=BUSINESS_CONTRACT_PATH, expected_sha256: str = BUSINESS_CONTRACT_SHA256) -> dict:
    """The contract document, refused unless its bytes match the pinned digest."""
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha256:
        raise BusinessContractError(f"BUSINESS_CONTRACT_DIGEST:{got}!={expected_sha256}")
    return json.loads(raw)


def profile(doc: dict | None = None) -> dict:
    """The engine-facing values of the contract: leverage, rel_volume, the daily cap (None = no cap), the session
    rules (hours after the open before entries, hours before the close for the forced flat), one position at a time."""
    d = doc if doc is not None else load()
    return {"leverage": float(d["leverage"]), "rel_volume": float(d["rel_volume"]), "max_entries_per_day": d["max_entries_per_day"],
            "entry_wait_after_session_open_hours": float(d["entry_wait_after_session_open_hours"]),
            "forced_flat_before_session_close_hours": float(d["forced_flat_before_session_close_hours"]),
            "max_simultaneous_positions": int(d["max_simultaneous_positions"]), "early_close_enabled": bool(d["early_close"]["enabled"]),
            "schema": d["schema"], "sha256": BUSINESS_CONTRACT_SHA256}
