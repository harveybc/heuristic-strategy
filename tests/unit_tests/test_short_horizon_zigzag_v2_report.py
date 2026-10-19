"""ZigZag v2 report helpers: the union front keeps exactly the nondominated points (Sharpe max, others min)."""
from tools.short_horizon_zigzag_v2_report import nondominated


def test_union_front_is_nondominated_with_sharpe_maximized():
    rows = [{"obj": [10.0, 5.0]}, {"obj": [9.0, 4.0]}, {"obj": [8.0, 6.0]}, {"obj": [10.0, 5.0 + 1e-9]}]
    keep = nondominated(rows, 2)
    assert [r["obj"] for r in keep] == [[10.0, 5.0], [9.0, 4.0]]
    rows3 = [{"obj": [1.0, 5.0, 3.0]}, {"obj": [1.0, 5.0, 2.0]}, {"obj": [0.5, 1.0, 9.0]}]
    assert [r["obj"] for r in nondominated(rows3, 3)] == [[1.0, 5.0, 2.0], [0.5, 1.0, 9.0]]
