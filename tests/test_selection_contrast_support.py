"""Exercise the real queue builder without importing or fitting any model."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


def engine():
    source = (Path(__file__).parents[1] / "tools/pool_selection_v2.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Engine")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "arms")
    scope = {"TASKS": {"task": "cls"}, "CORE15": ["context"], "FOLDS_BASE": 6,
             "FOLDS_CONTRAST": 3, "SUBGROUPS": {"details": ("detail",)}}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "real_engine_arms", "exec"), scope)
    fake = type("SyntheticEngine", (), {"arms": scope["arms"], "informative_tasks": lambda self: ["task"]})()
    fake.a = SimpleNamespace(stage="contrast")
    fake.pool_cols = fake.cols = fake.temporal_cols = ["context", "parent", "detail", "approx"]
    fake.families = {"decomp": ["detail", "approx"], "raw": ["parent"], "other": ["context"]}
    fake.decomp = {"decomp": {"parent_cols": ["parent"], "components": {"detail": "detail", "approx": "approx"}}}
    return fake


def groups():
    result = defaultdict(list)
    for unit in engine().arms():
        if unit[1] == "contrast" and unit[-1]["family"] == "decomp":
            result[(unit[-1]["ctx"], unit[5])].append(unit)
    return result


def test_all_arms_have_exact_same_row_columns():
    for units in groups().values():
        row_sets = [tuple(u[-1]["row_cols"]) for u in units]
        assert len(set(row_sets)) == 1
        assert set(row_sets[0]) == {"context", "parent", "detail", "approx"}
        assert all(u[-1]["support_protocol"] == "family_union_v1" for u in units)


def test_pairing_does_not_change_which_features_an_arm_consumes():
    for units in groups().values():
        by_arm = {u[-1]["arm"]: u[4] for u in units}
        assert by_arm["none"] == ["context"]
        assert by_arm["P"] == ["context", "parent"]
        assert by_arm["C:detail"] == ["context", "detail"]
        assert by_arm["P+Cgroup"] == ["context", "parent", "detail", "approx"]


def test_no_stage_fold_or_key_is_lost():
    g = groups()
    assert len(g) == 6  # two contexts, three folds
    assert all(len(units) == 7 for units in g.values())
    all_units = engine().arms()
    assert len({u[0] for u in all_units}) == len(all_units)
