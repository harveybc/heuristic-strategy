"""The behaviour-matrix queue driver (tools/behaviour_matrix_driver.sh) exits on its own under ONESHOT once its queue
has drained (F10 2026-10-29: an empty queue printed one blank line into PENDING.txt and `grep -c ""` counted it, so the
f439d37 and 51afc0f tier-1 drivers looped after their last cell until a STOP flag was set)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_oneshot_driver_exits_when_the_queue_has_drained(tmp_path):
    r = tmp_path / "root"
    (r / "queue").mkdir(parents=True)
    (r / "queue" / "empty.json").write_text("[]")
    (r / "code").symlink_to(ROOT)
    env = dict(os.environ, ROOT=str(r), RES=str(r / "results"), ONESHOT="1", JOBS="2", PY=sys.executable, CR="/bin/false", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run(["bash", str(ROOT / "tools/behaviour_matrix_driver.sh")], env=env, timeout=240, capture_output=True, text=True)
    log = (r / "LOG").read_text()
    assert p.returncode == 0 and "queue drained" in log and "ONESHOT driver exits" in log, log
    pending = (r / "PENDING.txt").read_text()
    assert pending.strip() == ""                                       # nothing pending, yet the file holds a newline: the old count read 1
    assert subprocess.run(["grep", "-c", "", str(r / "PENDING.txt")], capture_output=True, text=True).stdout.strip() == "1"
    assert subprocess.run(["grep", "-c", ".", str(r / "PENDING.txt")], capture_output=True, text=True).stdout.strip() == "0"
