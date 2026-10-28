import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def source_scope():
    source = (Path(__file__).parents[1] / "tools/pool_selection_v2.py").read_text()
    wanted = {"resolve_pool_file", "load_pool", "decided_columns"}
    defs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = {"Path": Path, "json": json, "pd": SimpleNamespace(DataFrame=object),
             "DG": SimpleNamespace(sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "real_pool_loader", "exec"), scope)
    return scope


def store(tmp_path, named=True):
    right, wrong = tmp_path / "full.parquet", tmp_path / "old-fit.parquet"
    wrong.write_bytes(b"wrong")
    right.write_bytes(b"right")
    m = {"file_sha256": hashlib.sha256(b"right").hexdigest(), "columns": ["present", "absent"]}
    if named:
        m["file"] = right.name
    (tmp_path / "MANIFEST.json").write_text(json.dumps(m))
    return right, wrong


def test_declared_file_wins_over_glob_order(tmp_path):
    right, _ = store(tmp_path)
    assert source_scope()["resolve_pool_file"](tmp_path)[0] == right


def test_no_filename_requires_exact_digest_match(tmp_path):
    right, _ = store(tmp_path, named=False)
    assert source_scope()["resolve_pool_file"](tmp_path)[0] == right


def test_explicit_wrong_file_rejected(tmp_path):
    _, wrong = store(tmp_path)
    with pytest.raises(ValueError, match="POOL_FILE_HASH_MISMATCH"):
        source_scope()["resolve_pool_file"](wrong)


def test_modified_declared_file_rejected(tmp_path):
    right, _ = store(tmp_path)
    right.write_bytes(b"changed")
    with pytest.raises(ValueError, match="POOL_FILE_HASH_MISMATCH"):
        source_scope()["resolve_pool_file"](tmp_path)


def test_ambiguous_digest_without_filename_rejected(tmp_path):
    right, _ = store(tmp_path, named=False)
    (tmp_path / "copy.parquet").write_bytes(right.read_bytes())
    with pytest.raises(ValueError, match="POOL_FILE_AMBIGUOUS"):
        source_scope()["resolve_pool_file"](tmp_path)


def test_missing_columns_are_not_silently_dropped(tmp_path):
    store(tmp_path)
    scope = source_scope()
    scope["pd"].read_parquet = lambda p: SimpleNamespace(columns=["present"])
    with pytest.raises(ValueError, match="POOL_COLUMNS_MISSING"):
        scope["load_pool"]([str(tmp_path)])
