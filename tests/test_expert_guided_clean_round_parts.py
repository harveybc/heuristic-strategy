"""Parts of the clean expert-guided round (owner 2026-10-22): grouped small extractors, multi-expert
training-only demonstrations with per-expert support purge, window layout on the store path. No data, no GPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.short_horizon.expert_guided import env as E  # noqa: E402
from app.short_horizon.expert_guided import temporal as T  # noqa: E402
from tools import expert_guided_e3m as M  # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ grouped extractors
def test_groups_must_partition_the_features_exactly_once():
    with pytest.raises(ValueError, match="TRUNK_GROUPS_MUST_PARTITION_FEATURES"):
        T.TemporalTrunk(5, groups=[[0, 1], [2, 3]])                       # feature 4 missing
    with pytest.raises(ValueError, match="TRUNK_GROUPS_MUST_PARTITION_FEATURES"):
        T.TemporalTrunk(4, groups=[[0, 1, 2], [2, 3]])                    # feature 2 twice
    with pytest.raises(ValueError, match="TRUNK_GROUP_UNITS_INVALID"):
        T.TemporalTrunk(4, groups=[[0, 1], [2, 3]], group_units=[2])


def test_grouped_extractor_parameter_count_is_small_and_exact():
    # 3 groups of sizes 3, 2, 1 with 2 units each: conv(in=g, out=2, k=3) = 2*g*3 + 2 -> 20 + 14 + 8 = 42
    tr = T.TemporalTrunk(6, units=2, core=8, depth=2, groups=[[0, 1, 2], [3, 4], [5]])
    assert n_params(tr.branch) == 42
    assert n_params(tr.fuse) == 8 * 6 * 3 + 8                            # 6 = sum of group units
    assert n_params(tr.core) == 8 * 8 * 3 + 8 and n_params(tr.core2) == 8 * 8 * 3 + 8
    # the per-feature default is unchanged (registered design)
    assert n_params(T.TemporalTrunk(4, units=2)) == 96


def test_grouped_trunk_is_causal_and_reads_only_its_group():
    torch.manual_seed(0)
    tr = T.TemporalTrunk(5, units=2, core=4, depth=2, dropout=0.0, groups=[[0, 3], [1, 2, 4]]).eval()
    w = torch.randn(2, 24, 5)
    base = tr(w)
    for j in range(24):                                                    # changing slot j never changes earlier outputs
        w2 = w.clone()
        w2[:, j, :] += 3.0
        x = w2.transpose(1, 2)
        out = torch.cat([conv(x.index_select(1, getattr(tr, f"gidx{k}"))) for k, conv in enumerate(tr.branch)], 1)
        x0 = w.transpose(1, 2)
        out0 = torch.cat([conv(x0.index_select(1, getattr(tr, f"gidx{k}"))) for k, conv in enumerate(tr.branch)], 1)
        assert torch.all((out - out0).abs().sum(1)[:, :j] == 0)
    # receptive field with depth 2 (dilations 1, 2, 4, 8) = 31 >= 24: the oldest slot reaches the current step
    w2 = w.clone()
    w2[:, 0, :] += 3.0
    assert not torch.allclose(tr(w2), base)
    # group 0 extractor sees features 0 and 3 only: perturbing feature 1 leaves its output unchanged
    x = w.transpose(1, 2)
    x2 = x.clone()
    x2[:, 1, :] += 9.0
    g0 = tr.branch[0](x.index_select(1, tr.gidx0))
    assert torch.allclose(g0, tr.branch[0](x2.index_select(1, tr.gidx0)))


def test_policy_net_with_groups_round_trips_its_state_dict():
    torch.manual_seed(1)
    kw = dict(units=2, core=8, depth=2, head=16, groups=[[0, 1], [2], [3, 4, 5]], group_units=[2, 2, 3],
              group_names=["a", "b", "c"])
    net = T.TemporalPolicyNet(6, 24, 2, n_reg=2, n_state=9, **kw).eval()
    net2 = T.TemporalPolicyNet(6, 24, 2, n_reg=2, n_state=9, **kw).eval()
    net2.load_state_dict(net.state_dict())
    x = torch.randn(3, 24 * 6 + 9)
    lg, rg = net(x)
    lg2, rg2 = net2(x)
    assert torch.allclose(lg, lg2) and torch.allclose(rg, rg2) and lg.shape == (3, 2)


# ------------------------------------------------------------------ multi-expert labels
FLAT = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], np.float32)
POS = np.array([1, 0.5, 0.2, 1.5, 2.0, 1, 1, 0.25, 1], np.float32)


class Orc:
    def __init__(self, action):
        self.a = action

    def action(self, bar, pos, can_enter, sigma):
        return self.a, 1.5, 2.0


class Track:
    def __init__(self, action, ok_bars):
        self.oracle, self.ok = Orc(action), set(ok_bars)

    def valid(self, bar, r):
        return bar in self.ok


class CX:
    def __init__(self):
        self.Xw = np.zeros((10, 3), np.float32)
        self.oracle = Orc(E.LONG)
        self.ds = type("DS", (), {"sigma": np.ones(20)})()
        self.h = np.array([1.0] * 12)
        self.l = np.array([1.0] * 12)
        self.c = np.array([1.0] * 12)


def test_dagger_rows_get_one_label_per_valid_expert_and_positioned_rows_one_exit_label():
    cx = CX()
    tracks = [Track(E.LONG, {1, 2}), Track(E.SHORT, {1}), Track(E.HOLD, {2})]
    vis = [(0, 1, FLAT, E.HOLD, 1.5, 2.0, None),                 # valid for tracks 0 and 1 -> 2 rows
           (1, 2, FLAT, E.HOLD, 1.5, 2.0, None),                 # valid for tracks 0 and 2 -> 2 rows
           (2, 3, FLAT, E.HOLD, 1.5, 2.0, None),                 # valid for no track -> dropped
           (3, 4, POS, E.HOLD, 1.5, 2.0, (1, 1.0, 0.9, 1.1, 8))]  # positioned -> one exit-rule row
    D = M.build_sets("FULL", vis, cx, False, tracks=tracks)
    assert len(D["open_l"][1]) == 4 and len(D["open_s"][1]) == 4
    assert D["open_l"][1].tolist() == [1, 0, 1, 0] and D["open_s"][1].tolist() == [0, 1, 0, 0]
    assert len(D["close_l"][1]) == 1
    # without tracks the single-oracle path is unchanged (one label per flat row, purge by the dataset's mask only)
    D1 = M.build_sets("FULL", vis, cx, False)
    assert len(D1["open_l"][1]) == 3 and D1["open_l"][1].tolist() == [1, 1, 1]


def test_round0_rows_of_an_extra_demonstrator_are_purged_by_its_own_support():
    cx = CX()
    tr = Track(E.SHORT, {1})
    vis = [(0, 1, FLAT, E.SHORT, 1.5, 2.0, None), (1, 2, FLAT, E.SHORT, 1.5, 2.0, None)]
    D = M.build_sets("FULL", vis, cx, True, track=tr)
    assert len(D["open_s"][1]) == 1 and D["open_s"][1].tolist() == [1]


def test_merge_sets_concatenates_per_module():
    cx = CX()
    a = M.build_sets("FULL", [(0, 1, FLAT, E.LONG, 1.5, 2.0, None)], cx, True)
    b = M.build_sets("FULL", [(1, 2, FLAT, E.SHORT, 1.5, 2.0, None)], cx, True)
    m = M.merge_sets(a, b)
    assert m["open_l"][1].tolist() == [1, 0] and m["open_s"][1].tolist() == [0, 1]
