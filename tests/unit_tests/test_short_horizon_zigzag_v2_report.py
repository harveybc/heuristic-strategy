"""ZigZag v2 report helpers: the union front keeps exactly the nondominated points (Sharpe max, others min)."""
from tools.short_horizon_zigzag_v2_report import nondominated


def test_union_front_is_nondominated_with_sharpe_maximized():
    rows = [{"obj": [10.0, 5.0]}, {"obj": [9.0, 4.0]}, {"obj": [8.0, 6.0]}, {"obj": [10.0, 5.0 + 1e-9]}]
    keep = nondominated(rows, 2)
    assert [r["obj"] for r in keep] == [[10.0, 5.0], [9.0, 4.0]]
    rows3 = [{"obj": [1.0, 5.0, 3.0]}, {"obj": [1.0, 5.0, 2.0]}, {"obj": [0.5, 1.0, 9.0]}]
    assert [r["obj"] for r in nondominated(rows3, 3)] == [[1.0, 5.0, 2.0], [0.5, 1.0, 9.0]]


def test_ties_collapse_and_a_bound_is_reported_only_when_every_tied_set_sits_on_it(tmp_path):
    import json
    import math
    from tools.short_horizon_zigzag_v2_report import load_chains
    d = tmp_path / "A4" / "seed1"
    d.mkdir(parents=True)
    pop = [[math.log(2), math.log(1.0), 1.0], [math.log(2), math.log(50.0), 1.0],   # tied: depth/backstep on bound, deviation not
           [math.log(10), math.log(20.0), 3.0]]
    fit = [[15.0, 4.0], [15.0, 4.0], [12.0, 3.0]]
    d.joinpath("checkpoint.json").write_text(json.dumps({"population": pop, "fitness": fit, "hv": [1.0] * 30, "generation": 29,
                                                         "evals": 3, "sims": 3}))
    chains, reps, _ = load_chains(tmp_path, "A4")
    by = {tuple(r["obj"]): r for r in reps}
    tied = by[(15.0, 4.0)]
    assert tied["n_tied"] == 2 and tied["tied_ranges"]["deviation_pips"] == [1.0, 50.0]
    assert tied["on_bound"] == ["depth=lower", "backstep=lower"]
    assert by[(12.0, 3.0)]["on_bound"] == []
