"""Formatting and axis-mapping checks for the price-path figures (owner style)."""
import numpy as np
import pandas as pd

import tools.short_horizon_price_path_report as R


def test_spanish_number_format():
    assert R.es(0.0021, 4) == "0,0021"
    assert R.es(31482.73, 2) == "31.482,73"
    assert R.es(-1540.4, 0) == "-1.540"


def test_monotone_check_detects_non_monotone_axes():
    t = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.1, 2.0, 3.2], "c": [0.0, 1.0, 0.9, 3.0]})
    ok = R.monotone_check(t, ["a", "b"])
    assert ok["monotone"] and ok["max_decrease_b"] == 0.0
    assert ok["max_rel_dev_b_vs_a"] == np.float64(max(abs(1.1 / 1 - 1), 0, abs(3.2 / 3 - 1)))
    bad = R.monotone_check(t, ["a", "c"])
    assert not bad["monotone"] and abs(bad["max_decrease_c"] - 0.1) < 1e-12
