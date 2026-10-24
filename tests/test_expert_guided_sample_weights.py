"""``--sample-weights`` (multires only) scales every flat training row's CE term by its label fitness; defaults stay
byte-identical (``sample_weight=None`` is the untouched code path). The clean-round emitter addresses the recipe at the
package root (``<root>/recipe/``), where the digest check reads it, not under ``<root>/data/``."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from app.short_horizon.expert_guided import learn as L  # noqa: E402
from tools import expert_guided_clean_round as CR  # noqa: E402
from tools import expert_guided_e3m as M  # noqa: E402


def _data(seed=0, n=240, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def _fit(**kw):
    X, y = _data()
    model, hist, info = L.fit(X, y, X[:60], y[:60], seed=7, n_cls=2, hidden=(8,), max_epochs=3, patience=5, **kw)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, hist, info


def test_default_path_is_untouched_by_the_new_argument():
    a, ha, ia = _fit()
    b, hb, ib = _fit(sample_weight=None)
    assert all(torch.equal(a[k], b[k]) for k in a) and ha == hb and "sample_weights" not in ia and "sample_weights" not in ib


def test_sample_weights_change_the_fit_and_are_declared():
    X, y = _data()
    w = np.where(y == 1, 0.05, 1.0).astype(np.float32)
    a, _, _ = _fit()
    b, _, info = _fit(sample_weight=w)
    assert info["sample_weights"]["rows"] == len(y) and info["sample_weights"]["min"] == pytest.approx(0.05)
    assert any(not torch.equal(a[k], b[k]) for k in a)
    c, _, _ = _fit(sample_weight=np.ones(len(y), np.float32), class_weight=np.array([1.0, 1.0]))
    d, _, _ = _fit(class_weight=np.array([1.0, 1.0]))
    assert all(torch.allclose(c[k], d[k], atol=1e-5) for k in c)      # unit weights = the plain class-weighted mean (float path aside)


class _Labels:
    def weight(self, bar, r):
        return 0.25


def test_row_weight_only_for_flat_multires_rows_when_enabled():
    lab = _Labels()
    assert M.row_weight(lab, "multires", True, None, 5, 3) == 0.25
    assert M.row_weight(lab, "multires", True, {"side": 1}, 5, 3) == 1.0          # positioned rows weigh 1
    assert M.row_weight(lab, "multires", False, None, 5, 3) == 1.0                # flag off
    assert M.row_weight(lab, "opportunity", True, None, 5, 3) == 1.0              # other modes have no fitness
    assert M.row_weight(None, "multires", True, None, 5, 3) == 1.0


def test_flag_default_off_and_refused_outside_multires():
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c"])
    assert a.sample_weights is False and M.check_sample_weights(a) is False
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--labels", "multires", "--sample-weights"])
    assert M.check_sample_weights(a) is True
    a = M.build_parser().parse_args(["run", "--admitted", "a", "--csv-dir", "b", "--out", "c", "--labels", "opportunity", "--sample-weights"])
    with pytest.raises(SystemExit, match="SAMPLE_WEIGHTS_REQUIRE_MULTIRES"):
        M.check_sample_weights(a)


def _spec():
    return {"model": {"units": 2, "depth": 2, "head": 16, "max_epochs": 5, "core": None}, "regularization": {"dropout": 0.3, "wd": 0.01},
            "recipe": {"path": "/x/CONSUMER_RECIPE.json"}, "store": {"path": "/x/store.parquet"}, "scaling": "fixed_train",
            "nonstationary": "diff", "split_contract": "consolidated", "exec_contract": "sac", "min_orders_per_week": 2.0}


def test_emitter_addresses_the_recipe_at_the_package_root():
    s = _spec()
    cmd = " ".join(CR.e3m_cmd(s, {"name": "zz"}, "OUT", "$Q/data", recipe_root="$Q"))
    assert "--recipe $Q/recipe/CONSUMER_RECIPE.json" in cmd and "--store $Q/data/store/store.parquet" in cmd and "data/recipe" not in cmd
    legacy = " ".join(CR.e3m_cmd(s, {"name": "zz"}, "OUT", "ROOT"))
    assert "--recipe ROOT/recipe/CONSUMER_RECIPE.json" in legacy
    mr = " ".join(CR.e3m_cmd(s, {"name": "mr", "labels": "multires", "multires_thresholds": "/y/TH.json", "sample_weights": True}, "OUT", "$Q/data", recipe_root="$Q"))
    assert "--labels multires" in mr and "--multires-thresholds $Q/recipe/TH.json" in mr and mr.endswith("--sample-weights") or "--sample-weights" in mr
    plain = " ".join(CR.e3m_cmd(s, {"name": "mr", "labels": "multires"}, "OUT", "$Q/data", recipe_root="$Q"))
    assert "--sample-weights" not in plain
