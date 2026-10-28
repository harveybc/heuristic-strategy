import copy
import json
from pathlib import Path

import numpy as np
import pytest

PARENT = Path(__file__).resolve().parents[2] / "docs/short_horizon/clean_round/evidence/a1/RECIPE_v2_a1.json"


def test_retire_only_constant_and_preserve_survivors():
    from tools.retire_constant_recipe import derive, encoded, sha_bytes
    parent = json.loads(PARENT.read_text())
    snapshot = copy.deepcopy(parent)
    rec, files = derive(parent, "a" * 64, np.array([1., np.nan, 1.]))
    i = parent["list"]["columns"].index("px.er_1")
    keep = list(range(i)) + list(range(i + 1, 121))
    assert parent == snapshot
    assert rec["list"]["n"] == 120
    assert rec["identity"] != parent["identity"]
    assert rec["list"]["columns"] == [parent["list"]["columns"][k] for k in keep]
    for field in ("mean", "std", "std_raw", "variance", "features", "fit_rows_per_feature", "nan_share_train"):
        assert rec["scaler"][field] == [parent["scaler"][field][k] for k in keep]
    for field in ("store", "business", "regularization", "labels"):
        assert rec[field] == parent[field]
    assert "support_hash" not in rec["split"]
    assert sorted(j for group in rec["representation"]["resolved"]["indices"] for j in group) == list(range(120))
    assert rec["representation"]["params_per_module"]["open_l"] == parent["representation"]["params_per_module"]["open_l"] - 6
    for name, digest in (("FEATURE_LIST.json", rec["list"]["sha256"]),
                         ("GROUPS.json", rec["representation"]["groups_file_sha256"]),
                         ("SCALER.json", rec["scaler"]["artifact_sha256"])):
        assert sha_bytes(encoded(files[name])) == digest


@pytest.mark.parametrize("values", [[1., 2.], [np.nan], [1., np.inf], [0., 0.]])
def test_refuse_nonconstant_or_wrong_constant(values):
    from tools.retire_constant_recipe import derive
    with pytest.raises(SystemExit, match="RECIPE_RETIRE"):
        derive(json.loads(PARENT.read_text()), "a" * 64, np.asarray(values))


def test_changed_parent_refused():
    from tools.retire_constant_recipe import derive
    parent = json.loads(PARENT.read_text())
    parent["scaler"]["mean"][0] += 1
    with pytest.raises(SystemExit, match="PARENT_IDENTITY"):
        derive(parent, "a" * 64, np.ones(3))
