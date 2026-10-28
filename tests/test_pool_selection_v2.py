"""Unit tests for tools/pool_selection_v2.py: the pure parts of the protocol (purged folds, block permutation,
family rules, catalog handling, inner-block iteration choice, decision rules). No admitted data is read."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.pool_selection_v2 as P  # noqa: E402


def test_purged_folds_are_contiguous_disjoint_and_embargoed():
    n, k, emb = 1000, 4, 50
    folds = list(P.purged_folds(n, k, emb))
    assert [f[0] for f in folds] == list(range(k))
    cover = np.concatenate([va for _, _, va in folds])
    assert np.array_equal(np.sort(cover), np.arange(n))             # validation folds tile the rows exactly once
    for _, tr, va in folds:
        assert not set(tr) & set(va)
        lo, hi = va.min(), va.max()
        assert not ((tr >= lo - emb) & (tr <= hi + emb)).any()       # embargo on both sides
        assert np.array_equal(va, np.arange(lo, hi + 1))              # contiguous


def test_block_permutation_moves_whole_blocks_jointly():
    rng = np.random.default_rng(3)
    n, block = 432, 72                                                # six whole blocks
    idx = P.block_perm(n, rng, block)
    assert np.array_equal(np.sort(idx), np.arange(n))                 # a permutation
    # consecutive rows inside a block stay consecutive after the permutation
    starts = idx[::block][: n // block]
    for s_ in starts:
        pos = np.where(idx == s_)[0][0]
        assert np.array_equal(idx[pos: pos + block], np.arange(s_, min(s_ + block, n)))
    assert not np.array_equal(idx, np.arange(n))                      # it did permute


def test_family_rule_splits_inventory_and_keeps_prefix_families():
    assert P.family_of("RSI") == "inv.momentum" and P.family_of("MACD_Signal") == "inv.trend"
    assert P.family_of("ATR") == "inv.volatility" and P.family_of("HIGH@diff") == "inv.raw"
    assert P.family_of("fx.gbpusd.ret_4") == "fx.gbpusd" and P.family_of("px.close_loc_12") == "px.close_loc"
    assert P.family_of("hurst_rs_w96") == "fractal" and P.family_of("hod_sin") == "calendar"


def test_catalog_companions_inherit_and_decomposition_families_are_keyed_by_parent(tmp_path):
    cat = pd.DataFrame([
        {"column": "wv.lc.A4", "family": "wavelet", "parent": "log_close", "component": "A4"},
        {"column": "wv.r1.D1", "family": "wavelet", "parent": "logret_1", "component": "D1"},
        {"column": "px.logret_1", "family": "price_pool", "parent": "Bid OHLC", "component": "px.logret_1"},
    ])
    f = tmp_path / "cat.csv"
    cat.to_csv(f, index=False)
    out = P.load_catalog(str(f), ["wv.lc.A4@diff", "wv.r1.D1", "px.logret_1", "RSI"])
    r = out.set_index("column")
    assert r.loc["wv.lc.A4@diff", "family"] == "wavelet/log_close" and r.loc["wv.lc.A4@diff", "component"] == "A4"
    assert r.loc["wv.r1.D1", "family"] == "wavelet/logret_1"
    assert r.loc["px.logret_1", "family"] == "px.logret" and pd.isna(r.loc["px.logret_1", "parent"])
    assert r.loc["RSI", "family"] == "inv.momentum"


def test_hgb_iterations_are_chosen_on_the_inner_block_only():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    tr, inner = np.arange(0, 3000), np.arange(3000, 4000)
    m = P.HGBModel("cls", 0).fit(X[tr], y[tr], X[inner], y[inner])
    assert 1 <= m.k <= P.HGB["max_iter"]
    p = m.predict(X[inner])
    assert p.shape == (len(inner), 2) and np.allclose(p.sum(1), 1)
    # without an inner block the model keeps every iteration
    m2 = P.HGBModel("cls", 0).fit(X[tr], y[tr])
    assert m2.k == m2.m.n_iter_


def test_prior_and_losses():
    y_tr = np.array([0, 0, 0, 1])
    y_va = np.array([0, 1])
    pl = P.prior_loss("cls", y_tr, y_va)
    assert pl == pytest.approx(-(np.log(0.75) + np.log(0.25)) / 2)
    assert P.prior_loss("reg", np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 0.0
    assert P.loss_of("reg", np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0


def _stub_engine(families, decomp=None):
    e = object.__new__(P.Engine)
    e.families = families
    e.decomp = decomp or {}
    e.informative = list(P.TASKS)
    e.encodings = {}
    e.low_window_coverage = {}
    e.cols = [c for cs in families.values() for c in cs]
    e.pool_cols = list(e.cols) + ["p1"]                                   # parents are pool columns too
    return e


def test_decision_rule_requires_lo90_above_zero_and_picks_the_best_arm_for_a_decomposition(tmp_path, monkeypatch):
    e = _stub_engine({"famA": ["a1", "a2"], "wav/log_close": ["w1", "w2"]},
                     {"wav/log_close": {"parent": ["log_close"], "parent_cols": ["p1"], "components": {"D1": "w1", "A4": "w2"}}})
    rows = []
    # famA: consistent gain in ctx core15 on opener3 -> RETAINED
    for k, (none, with_) in enumerate([(0.50, 0.45), (0.52, 0.47), (0.49, 0.44)]):
        rows.append({"stage": "contrast", "family": "famA", "ctx": "core15", "arm": "none", "task": "opener3", "fold": k, "val_loss": none, "n_cols": 15})
        rows.append({"stage": "contrast", "family": "famA", "ctx": "core15", "arm": "with", "task": "opener3", "fold": k, "val_loss": with_, "n_cols": 17})
    # wav: P+Cgroup best, Cgroup inconsistent (one fold hurts) -> retained arm P+Cgroup
    for k, (none, cg, pcg) in enumerate([(0.50, 0.48, 0.46), (0.50, 0.53, 0.47), (0.50, 0.49, 0.45)]):
        rows.append({"stage": "contrast", "family": "wav/log_close", "ctx": "full", "arm": "none", "task": "hit", "fold": k, "val_loss": none, "n_cols": 100})
        rows.append({"stage": "contrast", "family": "wav/log_close", "ctx": "full", "arm": "Cgroup", "task": "hit", "fold": k, "val_loss": cg, "n_cols": 102})
        rows.append({"stage": "contrast", "family": "wav/log_close", "ctx": "full", "arm": "P_decided+Cgroup", "task": "hit", "fold": k, "val_loss": pcg, "n_cols": 103})
    monkeypatch.setattr(e, "rows", lambda: rows)
    dec = e.decisions(quiet=True)
    assert dec["famA"]["status"] == "RETAINED" and dec["famA"]["retained_cols"] == ["a1", "a2"]
    assert dec["wav/log_close"]["status"] == "RETAINED" and dec["wav/log_close"]["retained_arm"] == "P_decided+Cgroup"
    assert dec["wav/log_close"]["retained_cols"] == ["p1", "w1", "w2"]
    # a single-fold gain with a loss in another fold does not retain
    e2 = _stub_engine({"famB": ["b1"]})
    rows2 = [{"stage": "contrast", "family": "famB", "ctx": "core15", "arm": a, "task": "opener3", "fold": k, "val_loss": v, "n_cols": 1}
             for k, (none, with_) in enumerate([(0.50, 0.40), (0.50, 0.60), (0.50, 0.50)]) for a, v in (("none", none), ("with", with_))]
    monkeypatch.setattr(e2, "rows", lambda: rows2)
    assert e2.decisions(quiet=True)["famB"]["status"] == "NOT_RETAINED_WITH_THIS_EVIDENCE"


def test_encoding_decision_keeps_base_unless_a_variant_wins_with_lo90():
    e = _stub_engine({"hod": ["hod_sin", "hod_cos"]})
    e.encodings = {"hod": {"base": ["hod_sin", "hod_cos"], "variants": {"onehot24": ["o1"], "sessions": ["s1"]}, "structural": True}}
    rows = []
    for k, (b, oh, se) in enumerate([(0.50, 0.47, 0.50), (0.50, 0.46, 0.49), (0.50, 0.48, 0.52)]):
        for arm, v in (("base", b), ("onehot24", oh), ("sessions", se)):
            rows.append({"stage": "encoding", "family": "hod", "arm": arm, "model": "hgb", "task": "opener3", "fold": k, "val_loss": v})
    out = e.encoding_decisions(rows)
    by = {(r["variant"], r["model"]): r for r in out}
    assert by[("onehot24", "hgb")]["decision"] == "USE_onehot24" and by[("onehot24", "hgb")]["lo90"] > 0
    assert by[("sessions", "hgb")]["decision"] == "USE_onehot24"      # the family decision is shared across rows
    rows_flat = [dict(r, val_loss=0.50) for r in rows]                 # no variant helps -> base stays
    assert all(r["decision"] == "KEEP_BASE_ENCODING" for r in e.encoding_decisions(rows_flat))


def test_lo90_is_a_one_sided_lower_bound():
    assert P.Engine.lo90([0.1, 0.1, 0.1]) == pytest.approx(0.1)
    assert P.Engine.lo90([0.3, -0.1, 0.1]) < 0.1 and P.Engine.lo90([5.0]) == float("-inf")


def test_parent_columns_resolve_by_base_name_after_a_scale_transform():
    pool = ["px.logret_1", "px.rv_24", "RAW_LOG_RANGE@log", "rg.gmm.p0", "rg.gmm.p1"]
    fam = {"px.logret_1": "px.logret", "px.rv_24": "px.rv", "RAW_LOG_RANGE@log": "inv.raw",
           "rg.gmm.p0": "gmm/x", "rg.gmm.p1": "gmm/x"}
    got = P.resolve_parent_cols(["logret_1+rv_24+log_range"], pool, fam, "gmm/x")
    assert got == ["px.logret_1", "px.rv_24", "RAW_LOG_RANGE@log"]
    # a column of the family itself is never its own parent
    assert P.resolve_parent_cols(["rg.gmm.p0"], pool, fam, "gmm/x") == []
