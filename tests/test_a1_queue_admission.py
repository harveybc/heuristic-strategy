import ast
import inspect
import json
from types import SimpleNamespace

import pytest

from tools import a1_dependent_queue as queue


def runner(tmp_path, monkeypatch):
    q = queue.Q({"work": str(tmp_path)})
    calls = []
    monkeypatch.setattr(queue.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0))
    return q, calls


def test_receipted_child_owns_single_admission(tmp_path, monkeypatch):
    q, calls = runner(tmp_path, monkeypatch)
    cmd = [queue.PY, "tools/receipted_run.py", "run", "--mem", "4G"]
    q.sh(cmd, "fit", tmp_path, child_admits=True)
    assert calls[0][0] == cmd
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_preparation_keeps_admission(tmp_path, monkeypatch):
    q, calls = runner(tmp_path, monkeypatch)
    cmd = [queue.PY, "tools/consumer_store.py", "--out", "result"]
    q.sh(cmd, "prepare", tmp_path)
    assert calls[0][0][0].endswith("crispdm-run")
    assert calls[0][0][-len(cmd):] == cmd


@pytest.mark.parametrize("cmd", [["python", "train.py"], [queue.PY, "tools/receipted_run.py", "config"]])
def test_other_commands_cannot_skip_parent_admission(tmp_path, monkeypatch, cmd):
    q, calls = runner(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="CHILD_ADMISSION_REQUIRES_RECEIPTED_RUN"):
        q.sh(cmd, "bad", tmp_path, child_admits=True)
    assert not calls


def test_both_gpu_stages_delegate_to_receipted_runner():
    tree = ast.parse(inspect.getsource(queue))
    delegated = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("t2_strict", "t3_satoshi"):
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "sh":
                    if "receipted_run.py" in ast.unparse(call) and '"run"' in ast.unparse(call).replace("'", '"'):
                        assert any(k.arg == "child_admits" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords)
                        delegated.append(fn.name)
    assert sorted(delegated) == ["t2_strict", "t3_satoshi"]


def test_resume_skips_done_preserves_failure_log(tmp_path, monkeypatch):
    q, calls = runner(tmp_path, monkeypatch)
    (q.st / "consumer_store").touch()
    q.step("consumer_store", lambda: pytest.fail("completed store repeated"))
    log = tmp_path / "a1q" / "prepare.log"
    log.write_text("previous failure\n")
    q.sh([queue.PY, "prepare.py"], "prepare", tmp_path)
    assert log.read_text() == "previous failure\n"


def test_lists_preserve_aggregate_order_and_replay(tmp_path):
    q = queue.Q({"work": str(tmp_path)})
    fin = tmp_path / "selection_v5" / "final"
    fin.mkdir(parents=True)
    (fin / "SHORTLIST.json").write_text(json.dumps({"lists": {"core15": ["b", "a"], "shortlist_strict": ["x"]}}))
    q.lists()
    first = {p.name: p.read_bytes() for p in fin.glob("FEATURE_LIST_*.json")}
    q.lists()
    assert first == {p.name: p.read_bytes() for p in fin.glob("FEATURE_LIST_*.json")}
    assert json.loads(first["FEATURE_LIST_core15.json"])["features"] == ["b", "a"]
