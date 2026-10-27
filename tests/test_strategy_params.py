"""STRATEGY_PARAMS.json (owner rule 2026-10-27): every coefficient in one validated file; unknown keys, missing keys,
out-of-bounds values and a tied pair that differs are refused; the committed file loads and equals the defaults."""
import json
from pathlib import Path

import pytest

from app.short_horizon.expert_guided import strategy_params as SP

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "docs/short_horizon/clean_round/contracts/STRATEGY_PARAMS.json"


def test_committed_file_loads_and_is_the_canonical_default():
    doc = SP.load(FILE)
    for k, v in SP.DEFAULTS.items():
        assert doc[k] == v, k
    g = SP.genes_c3(doc)
    assert set(g) == {"open_rate", "close_rem_rate_l", "close_rem_rate_s", "close_p_l", "close_p_s", "allow_reverse", "nop_veto"}
    assert len(SP.sha256_of(FILE)) == 64


def test_unknown_missing_out_of_bounds_and_tied_keys_are_refused():
    base = SP.default_doc()
    with pytest.raises(SP.StrategyParamsError, match="UNKNOWN_KEY"):
        SP.validate({**base, "magic_number": 3})
    d = dict(base)
    d.pop("risk_pct")
    with pytest.raises(SP.StrategyParamsError, match="MISSING_KEY"):
        SP.validate(d)
    with pytest.raises(SP.StrategyParamsError, match="OUT_OF_BOUNDS"):
        SP.validate({**base, "risk_pct": 0.5})
    with pytest.raises(SP.StrategyParamsError, match="OUT_OF_BOUNDS"):
        SP.validate({**base, "tp_grid": [0.5, 9.0]})
    with pytest.raises(SP.StrategyParamsError, match="TIED_KEYS"):
        SP.validate({**base, "open_rate_s": 0.4})
    assert SP.validate({**base, "nop_veto": None})["nop_veto"] is None
    full = SP.genes_full(base, "param", "param")
    assert "tp_a_l" in full and "v_min" in full and "observe_bars" in full
