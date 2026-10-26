"""Declared trunk variant ``residual_tcn`` (owner 2026-10-26: adopt Musashi's minimal grouped TCN): the default
u-design stays byte-identical (same state_dict and outputs as the committed implementation, and the sealed C v5
checkpoints still load strictly); the variant reproduces Musashi's GroupedTCNExtractor sequence output exactly
(his class copied below with attribution; torch only); it is causal; both readouts work; parameter counts."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.short_horizon.expert_guided import temporal as T  # noqa: E402

RECIPE = ROOT / "docs/short_horizon/clean_round/evidence/a1/RECIPE_v2_a1.json"
CORE19 = ROOT / "docs/short_horizon/clean_round/FEATURE_LIST_core19.json"
CKPT_DIR = Path("/var/tmp/expert-guided-clean-a1-20261024/out_from_host/out/A_zz_hold")


# --- Musashi's extractor, copied from agent-multi agent_plugins/grouped_tcn_orders.py @e18d00d4 (SB3 base removed) ---
class _MCausalConv1d(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation=1):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class _MResidualCausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.first = _MCausalConv1d(width, width, 3, dilation)
        self.second = _MCausalConv1d(width, width, 3, dilation)

    def forward(self, x):
        return F.silu(x + self.second(F.silu(self.first(x))))


class _MGroupedTCNExtractor(nn.Module):
    def __init__(self, resolved, fusion_units=8):
        super().__init__()
        self.branches = nn.ModuleList()
        for i, (idx, u, k) in enumerate(zip(resolved["indices"], resolved["units"], resolved["kinds"])):
            self.register_buffer(f"gidx{i}", torch.tensor(idx, dtype=torch.long), persistent=False)
            self.branches.append(_MCausalConv1d(len(idx), u, 3 if k == "conv" else 1))
        self.fusion = _MCausalConv1d(sum(resolved["units"]), fusion_units, 1)
        self.core = nn.Sequential(_MResidualCausalBlock(fusion_units, 1), _MResidualCausalBlock(fusion_units, 2))

    def sequence(self, x):                      # x: (B, T, F)
        x = x.transpose(1, 2)
        z = torch.cat([F.silu(branch(x.index_select(1, getattr(self, f"gidx{i}")))) for i, branch in enumerate(self.branches)], dim=1)
        return self.core(F.silu(self.fusion(z)))

    def forward(self, x):
        return self.sequence(x).flatten(1)


def _resolved():
    rec = json.loads(RECIPE.read_text())
    return rec, rec["representation"]["resolved"]


def _net(res, n_feat, **kw):
    return T.TemporalPolicyNet(n_feat, 13, 2, n_reg=2, n_state=9, units=2, depth=2, head=16, groups=res["indices"],
                               group_units=res["units"], group_names=res["names"], group_kinds=res["kinds"], **kw)


def _committed_temporal():
    """The implementation committed before this change (git HEAD~ at the time of writing is not stable; use the
    merge-base version pinned by content: the file at the first commit that contains 'core_blocks' is excluded)."""
    rev = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", "core_blocks", "--", "app/short_horizon/expert_guided/temporal.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    ref = f"{rev[0]}~1" if rev else "HEAD"       # the last committed file WITHOUT the variant
    src = subprocess.run(["git", "show", f"{ref}:app/short_horizon/expert_guided/temporal.py"], cwd=ROOT, capture_output=True, text=True).stdout
    if "core_blocks" in src:
        pytest.skip("no committed baseline without the variant")
    if not src:
        pytest.skip("committed baseline not available in this checkout")
    p = Path(__file__).resolve().parent / "_temporal_baseline_tmp.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("temporal_baseline", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p.unlink()
    return m


def test_default_path_is_byte_identical_to_the_committed_implementation():
    rec, res = _resolved()
    B = _committed_temporal()
    torch.manual_seed(7)
    old = B.TemporalPolicyNet(121, 13, 2, n_reg=2, n_state=9, units=2, depth=2, head=16, groups=res["indices"],
                              group_units=res["units"], group_names=res["names"], group_kinds=res["kinds"])
    torch.manual_seed(7)
    new = _net(res, 121)
    assert list(old.state_dict().keys()) == list(new.state_dict().keys())
    assert all(torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values()))
    x = torch.randn(5, 13 * 121 + 9)
    old.eval(); new.eval()
    lo, ro = old(x); ln, rn = new(x)
    assert torch.equal(lo, ln) and torch.equal(ro, rn)


@pytest.mark.skipif(not (CKPT_DIR / "FULL_open_l_20261019.pt").exists(), reason="sealed checkpoint not on this host")
def test_sealed_c_v5_checkpoint_still_loads_strictly_into_the_default_net():
    _, res = _resolved()
    e3m = json.loads((CKPT_DIR / "E3M.json").read_text())
    a = e3m["arch"]
    net = T.TemporalPolicyNet(121, 13, 2, n_reg=2, n_state=9, units=a["units"], branch=a["branch"], core=a["core"], depth=a["depth"],
                              head=a["head"], groups=a["groups"], group_units=a["group_units"], group_names=a["group_names"], group_kinds=a["group_kinds"])
    sd = torch.load(CKPT_DIR / "FULL_open_l_20261019.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)


def test_residual_tcn_matches_musashis_extractor_exactly():
    rec, res = _resolved()
    torch.manual_seed(1)
    m = _MGroupedTCNExtractor(res, fusion_units=8).eval()
    mine = T.TemporalTrunk(121, units=2, dropout=0.3, groups=res["indices"], group_units=res["units"], group_kinds=res["kinds"],
                           trunk="residual_tcn", fusion_units=8, core_blocks=2, readout="flatten", win=13).eval()
    # map his state_dict names onto mine: branches.i.conv -> branch.i ; fusion.conv -> fuse ; core.j.{first,second}.conv -> blocks.j.{first,second}
    mapped = {}
    for k, v in m.state_dict().items():
        k2 = k.replace("branches.", "branch.").replace("fusion.conv.", "fuse.").replace(".conv.", ".")
        k2 = k2.replace("core.", "blocks.")
        mapped[k2] = v
    missing, unexpected = mine.load_state_dict(mapped, strict=True), None
    x = torch.randn(4, 13, 121)
    with torch.no_grad():
        a = m.sequence(x)
        b = mine.sequence(x)
        assert torch.allclose(a, b, atol=1e-6, rtol=0), float((a - b).abs().max())
        assert torch.allclose(m(x), mine(x), atol=1e-6, rtol=0)
    assert mine(x).shape == (4, 13 * 8)


def test_residual_tcn_is_causal_and_last_readout_works():
    _, res = _resolved()
    torch.manual_seed(3)
    tr = T.TemporalTrunk(121, units=2, groups=res["indices"], group_units=res["units"], group_kinds=res["kinds"],
                         trunk="residual_tcn", readout="last", win=13).eval()
    x = torch.randn(2, 13, 121); y = x.clone(); y[:, -1, :] += 5.0          # perturb only the LAST slot
    with torch.no_grad():
        sa, sb = tr.sequence(x), tr.sequence(y)
    assert torch.allclose(sa[:, :, :-1], sb[:, :, :-1], atol=1e-6)         # earlier slots never see a later one
    assert not torch.allclose(sa[:, :, -1], sb[:, :, -1])
    assert tr(x).shape == (2, 8)
    with pytest.raises(ValueError, match="TRUNK_FLATTEN_NEEDS_WINDOW"):
        T.TemporalTrunk(121, groups=res["indices"], group_units=res["units"], group_kinds=res["kinds"], trunk="residual_tcn")
    with pytest.raises(ValueError, match="TRUNK_VARIANT_UNKNOWN"):
        T.TemporalTrunk(121, trunk="dense")


def _core19_groups(rec, core):
    g = rec["representation"]["groups"]["groups"]; kinds = rec["representation"]["groups"]["kinds"]
    sub = {k: [c for c in v if c in core] for k, v in g.items()}; sub = {k: v for k, v in sub.items() if v}
    extra = [c for c in core if not any(c in v for v in g.values())]
    if extra:
        sub["inv.momentum"] = sub.get("inv.momentum", []) + extra       # MACD / ADX / DI are momentum indicators (declared)
    cols = [c for v in sub.values() for c in v]
    idx = {c: i for i, c in enumerate(cols)}
    return cols, [[idx[c] for c in v] for v in sub.values()], [2] * len(sub), [kinds.get(k, "conv") for k in sub], list(sub)


def test_parameter_counts_are_reported():
    rec, res = _resolved()                      # the 19-column list was retired by the owner (2026-10-26); only the 121-column recipe is counted
    out = {}
    for name, (nf, groups, gu, gk) in {"recipe_v2_121": (121, res["indices"], res["units"], res["kinds"])}.items():
        base = T.TemporalPolicyNet(nf, 13, 2, n_reg=2, n_state=9, units=2, depth=2, head=16, groups=groups, group_units=gu, group_kinds=gk)
        tcn = T.TemporalPolicyNet(nf, 13, 2, n_reg=2, n_state=9, units=2, head=16, groups=groups, group_units=gu, group_kinds=gk,
                                  trunk="residual_tcn", fusion_units=8, core_blocks=2, readout="flatten")
        tcn_last = T.TemporalPolicyNet(nf, 13, 2, n_reg=2, n_state=9, units=2, head=16, groups=groups, group_units=gu, group_kinds=gk,
                                       trunk="residual_tcn", fusion_units=8, core_blocks=2, readout="last")
        out[name] = {"u_design_opener": T.param_table(base)[-1]["params"], "residual_tcn_flatten_opener": T.param_table(tcn)[-1]["params"],
                     "residual_tcn_last_opener": T.param_table(tcn_last)[-1]["params"], "n_feat": nf, "n_groups": len(groups)}
    assert out["recipe_v2_121"]["u_design_opener"] == 1382                   # the registered count (E3M.json params FULL open_l)
    (ROOT / "docs/short_horizon/clean_round/evidence/residual_tcn").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/short_horizon/clean_round/evidence/residual_tcn/PARAM_COUNTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
