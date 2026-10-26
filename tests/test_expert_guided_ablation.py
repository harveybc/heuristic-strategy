"""E3-M agent-state ablation (--agent-state full|minimal|none) and the split-contract label (--split-contract
dev3y|consolidated). Omitted feedback slots are constant zeros in the network input only: shape and parameter count
never change, labels/partitions and the env keep the real state. ``consolidated`` lives at the period level: TRAIN
2020-2023 with its last 6 months as the inner selection block, 2024 read once, TEST (>= 2025) closed; the default
arm is byte-identical to the module constants.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402
from app.short_horizon.expert_guided import env as E  # noqa: E402

FEATS4 = DG.FEATURES_V4
FEEDBACK_SLOTS = ("position", "unrealized_sigma", "holding_days", "sl_distance_sigma", "tp_distance_sigma",
                  "hours_since_operational_start_days", "hours_to_forced_flat_days", "entries_today_share", "can_enter")
FLAT = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], np.float32)
LONG = np.array([1, 0.5, 0.2, 1.5, 2.0, 1, 1, 0.25, 1], np.float32)
SHORT = np.array([-1, -0.3, 0.1, 1.0, 2.5, 1, 1, 0.25, 1], np.float32)


# ------------------------------------------------------------------ 6. the nine feedback slots and their modes
def test_inventory_names_the_nine_feedback_slots_exactly():
    assert E.STATE_NAMES == FEEDBACK_SLOTS and len(E.STATE_NAMES) == 9
    assert tuple(E.AGENT_STATE_MODES) == ("full", "minimal", "none")
    assert E.state_mask("full") is None                                        # byte-identical default path
    assert E.state_inventory("full") == {n: "included" for n in FEEDBACK_SLOTS}
    m = E.state_mask("minimal")
    assert m.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1] and m.dtype == np.float32
    inv = E.state_inventory("minimal")
    assert [n for n, v in inv.items() if v == "included"] == ["position", "can_enter"]
    assert E.state_mask("none").tolist() == [0] * 9
    assert set(E.state_inventory("none").values()) == {"constant_zero"} and list(E.state_inventory("none")) == list(FEEDBACK_SLOTS)
    with pytest.raises(ValueError, match="AGENT_STATE_UNKNOWN:partial"):
        E.state_mask("partial")
    # masking keeps the length (constant zeros), never drops a slot
    for mode in ("full", "minimal", "none"):
        out = E.masked_state(LONG, E.state_mask(mode))
        assert len(out) == 9
    assert E.masked_state(LONG, None) is LONG
    assert E.masked_state(LONG, E.state_mask("minimal")).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1]


def test_parameter_count_is_identical_across_modes():
    torch = pytest.importorskip("torch")  # noqa: F841
    from app.short_horizon.expert_guided import temporal as T
    n = {mode: sum(p.numel() for p in T.TemporalPolicyNet(15, DG.WIN, 4, n_reg=2, n_state=len(E.STATE_NAMES), units=2).parameters())
         for mode in E.AGENT_STATE_MODES}
    assert len(set(n.values())) == 1 and n["full"] == 120 + 182 + 14 + (4 * 11 + 4) + (2 * 11 + 2)   # trunk + heads on (2 + 9) inputs


class Recorder:
    """Policy stub that records every input vector it is called with."""

    def __init__(self, logits, reg=None):
        self.lg, self.rg, self.seen = np.asarray(logits, np.float32), reg, []

    def __call__(self, x):
        self.seen.append(np.array(x, np.float32))
        n = len(x)
        return np.tile(self.lg, (n, 1)), (None if self.rg is None else np.tile(np.asarray(self.rg, np.float32), (n, 1)))


def test_composed_feeds_constant_zeros_but_routes_and_enforces_with_the_real_state():
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M

    class CX:
        Xw = np.arange(13 * 15, dtype=np.float32)[None, :] / 100.0
    cx = CX()
    for mode in ("none", "minimal"):
        mask = E.state_mask(mode)
        pols = {"open_l": Recorder([0.0, 2.0], [1.0, 2.0]), "open_s": Recorder([2.0, 0.0], [1.0, 2.0]),
                "close_l": Recorder([0.0, 3.0]), "close_s": Recorder([3.0, 0.0])}
        c = M.Composed("FULL", pols, "prob", tau=0.5, mask=mask)
        f = c(cx)
        assert f(0, 0, SHORT, None)[0] == E.HOLD                              # closer_s (real position) says hold
        assert pols["close_s"].seen and not pols["close_l"].seen               # routed by the REAL side
        x = pols["close_s"].seen[-1][0]
        assert np.array_equal(x[:195], cx.Xw[0])
        expect = np.zeros(9, np.float32) if mode == "none" else SHORT * mask
        assert np.array_equal(x[195:], expect)                                 # omitted slots are constant zeros
        assert f(0, 0, LONG, None)[0] == E.CLOSE                              # closer_l exits, arbiter re-opens long -> exit
        assert f(0, 0, FLAT, None) == (E.LONG, 1.0, 2.0)
        xo = pols["open_l"].seen[-1][0]
        assert np.array_equal(xo[195:], np.zeros(9, np.float32) if mode == "none" else FLAT * mask)
    # JOINT under ``none``: the network input is the same for a flat and a positioned state (no feedback at all)
    joint = Recorder([0.0, 1.0, 0.0, 0.0], [1.5, 2.0])
    f = M.Composed("JOINT", {"joint": joint}, "argmax", mask=E.state_mask("none"))(cx)
    f(0, 0, FLAT, None), f(0, 0, LONG, None)
    assert np.array_equal(joint.seen[0], joint.seen[1]) and np.all(joint.seen[0][0, 195:] == 0)
    # the default (mask None) feeds the real state, as before
    joint = Recorder([0.0, 1.0, 0.0, 0.0], [1.5, 2.0])
    M.Composed("JOINT", {"joint": joint}, "argmax")(cx)(0, 0, LONG, None)
    assert np.array_equal(joint.seen[0][0, 195:], LONG)


def test_build_sets_zeroes_the_input_slots_but_labels_by_the_real_partition():
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M

    class CX:
        Xw = np.zeros((3, 13 * 15), np.float32)
        h = l = c = np.ones(10)
    cx = CX()
    vis = [(0, 1, FLAT, E.LONG, 1.0, 2.0, None), (1, 2, LONG, E.HOLD, 1.5, 2.0, None), (2, 3, SHORT, E.CLOSE, 1.5, 2.0, None)]
    full = M.build_sets("FULL", vis, cx, True)
    none = M.build_sets("FULL", vis, cx, True, mask=E.state_mask("none"))
    minimal = M.build_sets("FULL", vis, cx, True, mask=E.state_mask("minimal"))
    assert set(full) == set(none) == set(minimal) == {"open_l", "open_s", "close_l", "close_s"}
    for m in full:
        assert np.array_equal(full[m][1], none[m][1]) and np.array_equal(full[m][1], minimal[m][1])   # same labels
        assert full[m][0].shape == none[m][0].shape == minimal[m][0].shape                           # same shape
        assert np.all(none[m][0][:, 195:] == 0)
    assert np.array_equal(full["close_s"][0][0, 195:], SHORT)
    assert minimal["close_s"][0][0, 195:].tolist() == [-1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert full["close_l"][1].tolist() == [0] and full["close_s"][1].tolist() == [M.EXIT]


# ------------------------------------------------------------------ 7. split contract
def test_split_contracts_dev3y_is_the_default_and_consolidated_folds_inside_train():
    assert tuple(DG.SPLIT_CONTRACTS)[:4] == ("dev3y", "consolidated", "wf2022", "wf2023")          # protocol v11 adds the nested folds after these
    assert set(DG.SPLIT_CONTRACTS) == {"dev3y", "consolidated", "wf2022", "wf2023"} | set(DG.WALK_FORWARD_NESTED)
    d = DG.split_contract("dev3y")
    assert d["periods"] is DG.PERIODS and d["status"] is DG.PERIOD_STATUS and d["scaler_fit"] == DG.PERIODS["TRAIN"]
    c = DG.split_contract("consolidated")
    assert c["train"] == ("2020-01-01", "2024-01-01") and c["validation"] == ("2024-01-01", "2025-01-01")
    assert c["periods"] == {"TRAIN": ("2020-01-01", "2023-07-01"), "INNER_A": ("2023-07-01", "2024-01-01"),
                            "OBS_2024": ("2024-01-01", "2025-01-01")}
    t1 = pd.Timestamp(c["train"][1])
    assert pd.Timestamp(c["periods"]["INNER_A"][0]) == t1 - pd.DateOffset(months=6)         # last 6 months of TRAIN
    assert c["selection"] == "INNER_A" and c["read_once"] == "OBS_2024" and c["report"] == ("TRAIN", "INNER_A")
    assert c["scaler_fit"] == c["periods"]["TRAIN"]                                           # inner block unseen by any statistic
    for arm in DG.SPLIT_CONTRACTS.values():
        assert all(pd.Timestamp(b) <= pd.Timestamp(DG.TEST_CLOSED_FROM) for _, b in arm["periods"].values())
    with pytest.raises(ValueError, match="SPLIT_CONTRACT_UNKNOWN:x"):
        DG.split_contract("x")


def _synthetic_dataset(contract, scaling):
    from tests.test_expert_guided_scaling import WEEKS, synthetic_weeks
    bars, closures = synthetic_weeks(WEEKS)
    cal = DG.calendar_from_closures(closures)
    cal.assign(bars.index)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"DATE_TIME": (bars.index + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                       "OPEN": bars.OPEN.to_numpy(), "HIGH": bars.HIGH.to_numpy(), "LOW": bars.LOW.to_numpy(),
                       "CLOSE": bars.CLOSE.to_numpy()})
    for j, f in enumerate(FEATS4):
        df[f] = rng.normal(size=len(df)) + j
    return DG.build(bars, cal, None, frame=df, features=FEATS4, scaling=scaling, contract=contract), df


def test_build_assigns_periods_and_scaler_fit_by_the_contract_and_records_the_arm():
    ds0, df = _synthetic_dataset("dev3y", "fixed_train")
    assert ds0.meta["periods"] == DG.PERIODS and ds0.meta["period_status"] == DG.PERIOD_STATUS
    assert ds0.meta["split_contract"] == {"arm": "dev3y", "train": ["2020-01-01", "2023-01-01"], "validation": ["2024-01-01", "2025-01-01"],
                                          "selection_period": "INNER_A", "read_once": "OBS_2024", "test_closed_from": "2025-01-01"}
    ts = pd.to_datetime(df["DATE_TIME"], utc=True)
    assert set(ds0.period[(ts >= "2023-07-01") & (ts < "2024-01-01")]) == {"INNER_B"}
    ds1, _ = _synthetic_dataset("consolidated", "fixed_train")
    assert ds1.meta["split_contract"]["arm"] == "consolidated" and ds1.meta["split_contract"]["train"] == ["2020-01-01", "2024-01-01"]
    assert set(ds1.period[(ts >= "2023-01-01") & (ts < "2023-07-01")]) == {"TRAIN"}
    assert set(ds1.period[(ts >= "2023-07-01") & (ts < "2024-01-01")]) == {"INNER_A"}
    assert set(ds1.period[(ts >= "2024-01-01")]) == {"OBS_2024"} and "INNER_B" not in ds1.meta["periods"]
    assert ds1.meta["scaling"]["train"] == ["2020-01-01", "2023-07-01"] and ds0.meta["scaling"]["train"] == ["2020-01-01", "2023-01-01"]
    assert ds1.meta["input_contract"]["scaling"] == "fixed TRAIN-only mean/std (2020-01-01..2023-07-01) for every feature"
    assert ds0.meta["input_contract"]["scaling"] == "fixed TRAIN-only mean/std (2020-2022) for every feature"
    assert set(ds1.rows("INNER_A")) <= set(np.flatnonzero((ts >= "2023-07-01") & (ts < "2024-01-01")))
    # the consolidated constants are fitted on MORE rows (2023H1 joins the fit block) and differ from dev3y's
    assert ds1.meta["scaling"]["fit_rows"] > ds0.meta["scaling"]["fit_rows"]
    assert ds1.meta["scaling"]["features"][FEATS4[0]]["mean"] != ds0.meta["scaling"]["features"][FEATS4[0]]["mean"]
    # the dataset's own period table drives name resolution (what E3-M / Ctx.rollout use)
    assert ds1.meta["periods"]["INNER_A"] == ("2023-07-01", "2024-01-01")


def test_e3m_cli_declares_the_new_flags_with_byte_identical_defaults():
    pytest.importorskip("torch")
    from tools import expert_guided_e3m as M
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert (a.agent_state, a.split_contract, a.scaling, a.no_verify) == ("full", "dev3y", "rolling_z", False)
    b = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--agent-state", "none",
                                     "--split-contract", "consolidated", "--scaling", "fixed_train", "--no-verify"])
    assert (b.agent_state, b.split_contract, b.scaling, b.no_verify) == ("none", "consolidated", "fixed_train", True)
    with pytest.raises(SystemExit):
        M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--agent-state", "partial"])
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "expert_guided_e3m.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--agent-state {full,minimal,none}" in r.stdout and "--split-contract {dev3y,consolidated,wf2022,wf2023," in r.stdout
