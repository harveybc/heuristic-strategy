"""Input-scaling propagation (auditor's defects, 2026-10-21): strict policy enum, TRAIN-only constants that never see
VALIDATION, explicit NaN policy with accounting, propagation through every loader, the SCALER.json artifact next to
the checkpoints, and the reload refusal without it. The integration test runs the real E3-M CLI on a small subset of
the admitted file + sel15 CSVs (built in a temp dir, pins skipped and recorded as verify=false) and checks that the
requested policy is the effective one in the tensors, that SCALER.json carries the same constants and that
checkpoint + scaler reproduce the saved action logits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import data as DG  # noqa: E402

HOUR = pd.Timedelta(hours=1)
ADMITTED_DIR = Path(os.environ.get("EG_ADMITTED_DIR", "/var/tmp/fs-timing/admitted"))
SEL15_DIR = Path(os.environ.get("EG_SEL15_DIR", "/var/tmp/pivot-detector-private/sel15"))
FEATS4 = DG.FEATURES_V4


# ------------------------------------------------------------------ synthetic inputs (no real data, no pins)
def _ns(stamps):
    return np.array([pd.Timestamp(t).value for t in stamps], np.int64)


def hourly_ts(start, n):
    t0 = pd.Timestamp(start, tz="UTC")
    return [t0 + k * HOUR for k in range(n)]


def synthetic_weeks(weeks, seed=0):
    """Hourly bars Sun 22:00 -> Fri 21:00 (120 bars) for every (sunday_22h) in ``weeks``; closures = every gap plus
    one before the first and one after the last week (segments need both ends). Returns (bars, closures)."""
    rng = np.random.default_rng(seed)
    stamps = []
    for w in weeks:
        stamps += hourly_ts(w, 120)
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[1.10, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0006, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0006, n)
    bars = pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=idx.rename("DATE_TIME"))
    opens = [pd.Timestamp(w, tz="UTC") for w in weeks]
    closes = [o + 120 * HOUR for o in opens]
    closures = [[(opens[0] - 48 * HOUR).isoformat(), opens[0].isoformat()]]
    closures += [[closes[i].isoformat(), opens[i + 1].isoformat()] for i in range(len(weeks) - 1)]
    closures += [[closes[-1].isoformat(), (closes[-1] + 48 * HOUR).isoformat()]]
    return bars, closures


WEEKS = (["2020-01-05 22:00", "2020-01-12 22:00", "2020-01-19 22:00", "2020-01-26 22:00", "2020-02-02 22:00", "2020-02-09 22:00"]
         + ["2021-03-07 22:00", "2021-03-14 22:00"] + ["2022-05-01 22:00", "2022-05-08 22:00"]
         + ["2023-01-08 22:00", "2023-01-15 22:00", "2023-01-22 22:00"] + ["2023-07-09 22:00", "2023-07-16 22:00", "2023-07-23 22:00"]
         + ["2023-10-01 22:00", "2023-10-08 22:00"] + ["2024-01-07 22:00", "2024-01-14 22:00", "2024-01-21 22:00"])


def write_synthetic_admitted(dst: Path, bars, closures):
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": bars.index, "available_at": bars.index + HOUR, "BidOpen": bars.OPEN.to_numpy(),
                  "BidHigh": bars.HIGH.to_numpy(), "BidLow": bars.LOW.to_numpy(), "BidClose": bars.CLOSE.to_numpy(),
                  "source_id": "synthetic:test"}).to_parquet(dst / "eurusd.parquet", index=False)
    (dst / "CLOSURE_INTERVALS.json").write_text(json.dumps(closures))


def write_synthetic_v4(dst: Path, bars, seed=1, warmup=30):
    """features_all.parquet stamped at bar START (as the real v4 matrix) with a leading NaN warm-up."""
    rng = np.random.default_rng(seed)
    n = len(bars)
    F = pd.DataFrame({f: rng.normal(size=n) * (j + 1) + j for j, f in enumerate(FEATS4)})
    F.iloc[:warmup, 2] = np.nan                                  # hurst-like leading warm-up
    (dst / "eval_input").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"timestamp": bars.index, "OPEN": bars.OPEN.to_numpy(), "HIGH": bars.HIGH.to_numpy(),
                       "LOW": bars.LOW.to_numpy(), "CLOSE": bars.CLOSE.to_numpy()})
    for f in FEATS4:
        df[f] = F[f].to_numpy()
    df.to_parquet(dst / "eval_input" / "features_all.parquet", index=False)
    (dst / "FROZEN_SUBSET.json").write_text(json.dumps({"selected": FEATS4}))
    return F


@pytest.fixture(scope="module")
def synthetic_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    bars, closures = synthetic_weeks(WEEKS)
    write_synthetic_admitted(root / "admitted", bars, closures)
    F = write_synthetic_v4(root / "v4", bars)
    return root / "admitted", root / "v4", bars, F


# ------------------------------------------------------------------ 1. strict policy enum
def test_unknown_policy_is_refused_before_any_fit_or_read():
    for bad in ("rolling", "fixed", "", None, "FIXED_TRAIN"):
        with pytest.raises(ValueError, match="SCALING_UNKNOWN"):
            DG.check_scaling(bad)
    with pytest.raises(ValueError, match="SCALING_UNKNOWN:bogus"):
        DG.build(None, None, None, scaling="bogus")                       # refused before touching bars/cal/csv
    for fn in (lambda: DG.load("/nonexistent/a", "/nonexistent/b", scaling="bogus"),
               lambda: DG.load_v4("/nonexistent/a", "/nonexistent/b", scaling="bogus"),
               lambda: DG.load_store("/nonexistent/a", "/nonexistent/b/x.parquet", ["f"], scaling="bogus")):
        with pytest.raises(ValueError, match="SCALING_UNKNOWN:bogus"):   # not FileNotFoundError: refused before I/O
            fn()
    assert DG.SCALINGS == ("fixed_train", "rolling_z")
    for ok in DG.SCALINGS:
        assert DG.check_scaling(ok) == ok


# ------------------------------------------------------------------ 4. TRAIN-only constants, NaN policy
def _panel(n=900, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.array([1.0, 5.0, 0.01]) + np.array([0.0, 100.0, -3.0])
    ts = _ns(hourly_ts("2022-12-15 00:00", n))                          # crosses 2023-01-01 at row 408
    return X, ts


def test_validation_rows_do_not_change_mean_or_std():
    X, ts = _panel()
    names = ["a", "b", "c"]
    w1, ok1, d1 = DG.fixed_train_windows(X, ts, names)
    X2 = X.copy()
    X2[ts >= DG._utc_ns("2023-01-01")] *= 10.0                            # VALIDATION-side rows changed
    X2[ts >= DG._utc_ns("2023-01-01")] += 7.0
    w2, ok2, d2 = DG.fixed_train_windows(X2, ts, names)
    for f in names:
        assert d1["features"][f]["mean"] == d2["features"][f]["mean"]
        assert d1["features"][f]["std"] == d2["features"][f]["std"]
    tr = ts < DG._utc_ns("2023-01-01")
    assert np.array_equal(w1[tr], w2[tr]) and np.array_equal(ok1, ok2)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    assert np.allclose([d1["features"][f]["mean"] for f in names], mu)
    assert np.allclose([d1["features"][f]["std"] for f in names], sd)
    assert d1["fit_rows"] == int(tr.sum()) and d1["ddof"] == 0


def test_same_raw_value_maps_to_the_same_scaled_value_from_two_origins():
    X, ts = _panel()
    i, j = 200, 800                                                       # TRAIN row and a VALIDATION row
    X[j] = X[i]
    w, ok, d = DG.fixed_train_windows(X, ts, ["a", "b", "c"])
    assert ok[i] and ok[j]
    assert np.array_equal(w[i, -1], w[j, -1])                             # current-bar slot of both rows
    mu = np.array([d["features"][f]["mean"] for f in ("a", "b", "c")])
    sd = np.array([d["features"][f]["std"] for f in ("a", "b", "c")])
    assert np.allclose(w[j, -1], (X[i] - mu) / sd, rtol=1e-6)
    assert np.allclose(w[j, 0], (X[j - 2 * 12] - mu) / sd, rtol=1e-6)     # oldest slot = row j - stride*(win-1)


def test_constant_column_rule_std_zero_to_one():
    X, ts = _panel()
    X[:, 1] = 42.0
    w, ok, d = DG.fixed_train_windows(X, ts, ["a", "const", "c"])
    assert d["features"]["const"] == {"scale": "fixed_train_z", "mean": 42.0, "std": 1.0, "std_raw": 0.0, "variance": 0.0,
                                      "rule": "std_zero_to_one", "nan_share_train": 0.0, "fit_rows": d["fit_rows"]}
    assert d["features"]["a"]["rule"] == "z"
    assert np.all(w[ok][:, :, 1] == 0.0)


def test_nan_accounting_and_typed_rejections():
    X, ts = _panel()
    names = ["a", "b", "c"]
    tr = ts < DG._utc_ns("2023-01-01")
    X[10:30, 1] = np.nan                                                  # a gap inside TRAIN
    X[500, 0] = np.nan                                                    # a NaN outside TRAIN
    w, ok, d = DG.fixed_train_windows(X, ts, names)
    nb = d["features"]["b"]
    assert nb["nan_share_train"] == pytest.approx(20 / tr.sum()) and nb["fit_rows"] == int(tr.sum()) - 20
    assert nb["mean"] == pytest.approx(np.nanmean(X[tr, 1])) and nb["std"] == pytest.approx(np.nanstd(X[tr, 1]))
    assert d["features"]["a"]["nan_share_train"] == 0.0
    # rows excluded because a slot of their window is NaN: rows 10..30 and 500 reach back 24 rows at stride 2
    nan_rows = set()
    for r in list(range(10, 30)) + [500]:
        nan_rows |= {r + 2 * k for k in range(13)}
    expect = sorted(x for x in nan_rows if x < len(X) and x >= 24)
    assert d["rows_excluded_nan_windows"] == len(expect) and d["rows_excluded_warmup"] == 24
    assert not ok[expect].any() and ok[expect[-1] + 1]
    assert d["nan_policy"].startswith("nanmean/nanstd over TRAIN")
    # typed rejections
    Xa = X.copy()
    Xa[tr, 2] = np.nan
    with pytest.raises(ValueError, match="SCALING_FEATURE_ALL_NAN:c"):
        DG.fixed_train_windows(Xa, ts, names)
    with pytest.raises(ValueError, match="SCALING_TRAIN_EMPTY"):
        DG.fixed_train_windows(X[~tr], ts[~tr], names)
    Xi = X.copy()
    Xi[100, 0] = np.inf
    with pytest.raises(ValueError, match="SCALING_NONFINITE_CONSTANT:a"):
        DG.fixed_train_windows(Xi, ts, names)


# ------------------------------------------------------------------ 2. propagation through load_v4
def test_load_v4_propagates_fixed_train_and_refuses_unknown(synthetic_dirs):
    adm, v4, bars, F = synthetic_dirs
    ds = DG.load_v4(adm, v4, verify=False, scaling="fixed_train")
    sc = ds.meta["scaling"]
    assert sc["policy"] == "fixed_train" and sc["method"] == "fixed_train" and sc["train"] == ["2020-01-01", "2023-01-01"]
    assert ds.meta["sources"]["loader"] == "v4" and ds.meta["sources"]["verified"] is False
    # constants = nanmean/nanstd of the v4 rows whose DATE_TIME (= bar start + 1 h) is inside TRAIN, warm-up rows dropped
    dt = bars.index + HOUR
    fin = np.isfinite(F.to_numpy(float)).all(1)
    first = int(np.argmax(fin))
    tr = (dt >= pd.Timestamp("2020-01-01", tz="UTC")) & (dt < pd.Timestamp("2023-01-01", tz="UTC"))
    keep = np.zeros(len(bars), bool)
    keep[first:] = True
    for j, f in enumerate(FEATS4):
        col = F[f].to_numpy(float)[tr & keep]
        assert sc["features"][f]["mean"] == pytest.approx(np.nanmean(col))
        assert sc["features"][f]["std"] == pytest.approx(np.nanstd(col))
    # every finite window value is (x - mean) / std of its raw feature
    r = int(np.flatnonzero(ds.row_ok)[50])
    b = int(ds.row_bar[r])
    raw = F.to_numpy(float)[first:][r]
    assert b >= 0
    mu = np.array([sc["features"][f]["mean"] for f in FEATS4])
    sd = np.array([sc["features"][f]["std"] for f in FEATS4])
    assert np.allclose(ds.windows[r, -1], (raw - mu) / sd, rtol=1e-5)
    # rolling default unchanged, unknown refused
    ds_r = DG.load_v4(adm, v4, verify=False)
    assert ds_r.meta["scaling"]["policy"] == "rolling_z" and ds_r.meta["scaling"]["constants"] is None
    assert ds_r.meta["input_contract"]["scaling"].startswith("causal rolling z-score")
    with pytest.raises(ValueError, match="SCALING_UNKNOWN:rolling"):
        DG.load_v4(adm, v4, verify=False, scaling="rolling")


# ------------------------------------------------------------------ 3. scaler artifact and reload refusal
def test_scaler_artifact_round_trip_and_refusals(synthetic_dirs, tmp_path):
    adm, v4, bars, F = synthetic_dirs
    ds = DG.load_v4(adm, v4, verify=False, scaling="fixed_train")
    with pytest.raises(ValueError, match="SCALER_MISSING"):
        DG.read_scaler(tmp_path)
    path = DG.write_scaler(tmp_path, ds, rung="TEST")
    assert path.name == DG.SCALER_FILE
    art = DG.read_scaler(tmp_path)
    assert art["policy"] == "fixed_train" and art["features"] == FEATS4 and art["n_features"] == 4
    assert art["fit_period"] == ["2020-01-01", "2023-01-01"] and art["fit_rows"] == ds.meta["scaling"]["fit_rows"]
    sc = ds.meta["scaling"]["features"]
    assert art["mean"] == [sc[f]["mean"] for f in FEATS4] and art["std"] == [sc[f]["std"] for f in FEATS4]
    assert art["variance"] == [sc[f]["variance"] for f in FEATS4] and art["rule"] == [sc[f]["rule"] for f in FEATS4]
    assert art["nan_share_train"] == [sc[f]["nan_share_train"] for f in FEATS4]
    assert set(art["sources"]["sha256"]) == {"eurusd.parquet", "CLOSURE_INTERVALS.json", "features_all.parquet", "FROZEN_SUBSET.json"}
    assert "commit" in art["code"] and art["split_contract"]["arm"] == "dev3y" and art["run"] == {"rung": "TEST"}
    assert art["window"] == {"observations": DG.WIN, "stride_rows": DG.STRIDE, "includes_current_completed_bar": True}
    assert DG.verify_scaler(art, ds) is art
    # names/order differ -> refused
    bad = dict(art, features=list(reversed(art["features"])))
    with pytest.raises(ValueError, match="SCALER_FEATURES_MISMATCH"):
        DG.verify_scaler(bad, ds)
    # policy differs -> refused (no hidden refit)
    ds_r = DG.load_v4(adm, v4, verify=False)
    with pytest.raises(ValueError, match="SCALER_POLICY_MISMATCH"):
        DG.verify_scaler(art, ds_r)
    # constants differ -> refused
    bad = dict(art, mean=[m + 1e-3 for m in art["mean"]])
    with pytest.raises(ValueError, match="SCALER_CONSTANTS_MISMATCH"):
        DG.verify_scaler(bad, ds)
    # rolling_z artifact: policy persisted, no constants, window recorded
    DG.write_scaler(tmp_path / "roll", ds_r)
    ar = DG.read_scaler(tmp_path / "roll")
    assert ar["policy"] == "rolling_z" and ar["mean"] is None and ar["std"] is None and ar["rolling_window_rows"] == DG.SCALE_W
    assert "no constants exist" in ar["note"]
    assert DG.verify_scaler(ar, ds_r) is ar
    with pytest.raises(ValueError, match="SCALER_POLICY_MISMATCH"):
        DG.verify_scaler(ar, ds)
    # load_with_scaler refuses arms/loaders the caller does not support
    with pytest.raises(ValueError, match="SCALER_LOADER_UNSUPPORTED:v4"):
        DG.load_with_scaler(tmp_path, adm, v4)


# ------------------------------------------------------------------ 5. integration: the E3-M CLI on a real-data subset
CHUNKS = [("2020-01-01", "2020-03-01"), ("2023-01-01", "2023-02-12"), ("2023-07-01", "2023-08-13"), ("2024-01-01", "2024-02-11")]


def build_real_fixture(dst: Path):
    """Subset of the admitted file + closures + sel15 CSVs over ``CHUNKS`` (receipts dropped, pins skipped)."""
    adm, sel = dst / "admitted", dst / "sel15"
    adm.mkdir(parents=True), sel.mkdir(parents=True)
    pq = pd.read_parquet(ADMITTED_DIR / "eurusd.parquet")
    t = pd.DatetimeIndex(pq["timestamp"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    keep = np.zeros(len(pq), bool)
    for a, b in CHUNKS:
        keep |= (t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))
    pq[keep].reset_index(drop=True).to_parquet(adm / "eurusd.parquet", index=False)
    (adm / "CLOSURE_INTERVALS.json").write_bytes((ADMITTED_DIR / "CLOSURE_INTERVALS.json").read_bytes())
    for role in ("train", "validation"):
        df = pd.read_csv(SEL15_DIR / f"{role}.csv")
        dt = pd.to_datetime(df["DATE_TIME"], utc=True) - HOUR
        k = np.zeros(len(df), bool)
        for a, b in CHUNKS:
            k |= (dt >= pd.Timestamp(a, tz="UTC")) & (dt < pd.Timestamp(b, tz="UTC"))
        df[k].to_csv(sel / f"{role}.csv", index=False)
    return adm, sel


@pytest.mark.slow
def test_e3m_cli_fixed_train_policy_is_effective_persisted_and_reloadable(tmp_path):
    pytest.importorskip("torch")
    if not (ADMITTED_DIR / "eurusd.parquet").exists() or not (SEL15_DIR / "train.csv").exists():
        pytest.skip("admitted file / sel15 CSVs not available on this host")
    adm, sel = build_real_fixture(tmp_path / "fx")
    out = tmp_path / "e3m"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, str(ROOT / "tools" / "expert_guided_e3m.py"), "run", "--admitted", str(adm), "--csv-dir", str(sel),
           "--out", str(out), "--scaling", "fixed_train", "--max-epochs", "1", "--variants", "JOINT", "--cpu", "--no-verify"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    e3 = json.loads((out / "E3M.json").read_text())
    assert e3["scaling"] == "fixed_train" == e3["meta"]["scaling"]["policy"]
    assert e3["verify"] is False and e3["split_contract"]["arm"] == "dev3y" and e3["split_contract"]["test_closed_from"] == "2025-01-01"
    assert e3["agent_state"]["mode"] == "full" and set(e3["agent_state"]["inventory"].values()) == {"included"}
    assert e3["scaler_file"] == "SCALER.json" and (out / "SCALER.json").exists() and (out / "PROBE.json").exists()
    # the artifact next to the checkpoints carries the dataset's constants in tensor order
    art = DG.read_scaler(out)
    # the fixed scaler differenced the non-stationary inputs (recorded decision), so names carry "@diff"
    st = art["stationarity"]
    assert st["policy"] == "diff" and "HIGH" in st["differenced"] and st["features"]["HIGH"]["decision"] == "DIFF"
    assert art["features"] == [f"{f}@diff" if f in st["differenced"] else f for f in DG.FEATURES] and art["run"]["rung"] == "E3M"
    assert art["sources"]["verified"] is False and art["split_contract"]["arm"] == "dev3y"
    # effective policy in the tensors: recompute (x - mean) / std from the raw CSV rows
    ds, art2 = DG.load_with_scaler(out, adm, sel, verify=False)
    assert art2 == art
    raw = pd.concat([pd.read_csv(sel / "train.csv"), pd.read_csv(sel / "validation.csv")], ignore_index=True)
    X = raw[DG.FEATURES].to_numpy(float)
    for f in st["differenced"]:                                           # same transform as the loader
        j = DG.FEATURES.index(f)
        X[:, j] = np.r_[np.nan, np.diff(X[:, j])]
    ts = pd.to_datetime(raw["DATE_TIME"], utc=True)
    tr = ((ts >= pd.Timestamp("2020-01-01", tz="UTC")) & (ts < pd.Timestamp("2023-01-01", tz="UTC"))).to_numpy()
    mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
    sd = np.where(sd > 0, sd, 1.0)
    assert np.allclose(art["mean"], mu) and np.allclose(art["std"], sd) and art["fit_rows"] == int(tr.sum())
    rows = np.r_[ds.rows("TRAIN")[:3], ds.rows("INNER_A")[:3], ds.rows("INNER_B")[:2], ds.rows("OBS_2024")[:3]]
    assert len(rows) == 11
    for r_ in rows:
        for k in (0, 5, 12):
            i = r_ - 2 * (12 - k)
            assert np.allclose(ds.windows[r_, k], (X[i] - mu) / sd, rtol=1e-5, atol=1e-6)
    # checkpoint + scaler reproduce the saved logits on the same rows (fresh process, fresh dataset)
    import torch
    from app.short_horizon.expert_guided import temporal as T
    from tools.expert_guided_e3m import MODULES
    probe = json.loads((out / "PROBE.json").read_text())["probe"]
    assert probe and set(probe) == {f"JOINT_joint_{s}" for s in (20261019, 20261020, 20261021)}
    Xw = ds.windows.reshape(len(ds.windows), -1).astype(np.float32)
    for key, pr in probe.items():
        nc, nr = MODULES["JOINT"]["joint"]
        net = T.TemporalPolicyNet(ds.windows.shape[2], DG.WIN, nc, n_reg=nr, n_state=9, **e3["arch"])
        net.load_state_dict(torch.load(out / f"{key}.pt"))
        pol = T.TorchPolicy(net)
        Xp = np.stack([np.r_[Xw[r_], np.zeros(9, np.float32)] for r_ in pr["rows"]]).astype(np.float32)
        lg, rg = pol(Xp)
        assert np.allclose(lg, np.asarray(pr["logits"]), atol=1e-5) and np.allclose(rg, np.asarray(pr["reg"]), atol=1e-5)
        assert [int(ds.row_ts[r_]) for r_ in pr["rows"]] == pr["row_ts_ns"]
    # refusals of the reload path: missing artifact, another feature order, another policy
    (out / "SCALER.json").rename(out / "SCALER.json.bak")
    with pytest.raises(ValueError, match="SCALER_MISSING"):
        DG.load_with_scaler(out, adm, sel, verify=False)
    (out / "SCALER.json.bak").rename(out / "SCALER.json")
    with pytest.raises(ValueError, match="SCALER_FEATURES_MISMATCH"):
        DG.verify_scaler(dict(art, features=art["features"][::-1]), ds)
    with pytest.raises(ValueError, match="SCALER_POLICY_MISMATCH"):
        DG.verify_scaler(art, DG.load(adm, sel, verify=False, scaling="rolling_z"))


def test_stationarity_rule_differences_a_random_walk_and_keeps_white_noise():
    rng = np.random.default_rng(0)
    n = 3000
    ts = np.arange(n, dtype=np.int64) * DG.H_NS + DG._utc_ns("2020-01-01")
    walk = np.cumsum(rng.normal(size=n)) + 100.0                          # a price-like level
    noise = rng.normal(size=n)
    X = np.stack([walk, noise], 1)
    X2, feats, table = DG.stationarity_decisions(X, ts, ["lvl", "ret"], ("2020-01-01", "2020-04-01"))
    assert feats == ["lvl@diff", "ret"] and table["lvl"]["decision"] == "DIFF" and table["ret"]["decision"] == "NONE"
    assert np.isnan(X2[0, 0]) and np.allclose(X2[1:, 0], np.diff(walk)) and np.array_equal(X2[:, 1], noise)
    with pytest.raises(ValueError, match="SCALING_NONSTATIONARY_INPUT:lvl"):
        DG.stationarity_decisions(X, ts, ["lvl", "ret"], ("2020-01-01", "2020-04-01"), policy="refuse")
    X3, feats3, table3 = DG.stationarity_decisions(X, ts, ["lvl", "ret"], ("2020-01-01", "2020-04-01"), policy="keep")
    assert feats3 == ["lvl", "ret"] and table3["lvl"]["decision"] == "KEEP_NONSTATIONARY" and np.array_equal(X3, X)
    # a column already decided upstream (catalog) is measured but never re-decided
    _, feats4, table4 = DG.stationarity_decisions(X, ts, ["lvl@asinh", "ret"], ("2020-01-01", "2020-04-01"))
    assert feats4 == ["lvl@asinh", "ret"] and table4["lvl@asinh"]["decision"] == "NONE"
    with pytest.raises(ValueError, match="NONSTATIONARY_UNKNOWN"):
        DG.stationarity_decisions(X, ts, ["lvl", "ret"], ("2020-01-01", "2020-04-01"), policy="bogus")


def test_eligible_from_removes_rows_from_the_agent_and_is_hashed(synthetic_dirs):
    adm, v4, _bars, _F = synthetic_dirs
    ds0 = DG.load_v4(adm, v4, scaling="fixed_train", verify=False)
    ds1 = DG.load_v4(adm, v4, scaling="fixed_train", verify=False, eligible_from="2020-01-20")
    cut = DG._utc_ns("2020-01-20")
    early = ds1.row_ts < cut
    assert early.any() and not ds1.observable()[early].any() and not ds1.labelled()[early].any()
    assert ds1.observable()[~early].sum() == ds0.observable()[~early].sum()
    assert ds1.meta["eligible_from"] == "2020-01-20" and set(ds1.meta["support_hash"]) == set(ds1.meta["periods"])
    assert ds1.meta["support_hash"]["TRAIN"] != ds0.meta["support_hash"]["TRAIN"]
    assert ds1.meta["support_hash"]["OBS_2024"] == ds0.meta["support_hash"]["OBS_2024"]
