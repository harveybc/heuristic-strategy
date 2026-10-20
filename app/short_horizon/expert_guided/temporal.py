"""Minimal causal temporal trunk for windowed time-series observations (owner rule 2026-10-20).

The window is a (time, features) tensor and is never flattened into a Dense layer. Layout, with ``u`` units
per component (the ladder starts at u = 2):

  1. one causal Conv1D branch per feature: 1 -> u filters, kernel 3, dilation 1, GELU   (grouped conv)
  2. concatenation of the branches (time, n_feat * u) + Dropout
  3. causal fusion Conv1D across features: n_feat * u -> u, kernel 3, dilation 2, GELU
  4. causal Conv1D u -> u, kernel 3, dilation 4, GELU
  5. the last time step (the current completed bar): receptive field 1 + 2 + 4 + 8 = 15 >= 13 slots

Weights are shared over time (the same filters slide over the window) and every convolution is padded on
the past side only, so position t never sees a later slot. Agent state, if any, is concatenated AFTER the
trunk, never mixed into the window. No BatchNorm.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

KERNEL = 3
DILATIONS = (1, 2, 4)


def receptive_field(kernel: int = KERNEL, dilations=DILATIONS) -> int:
    return 1 + sum((kernel - 1) * d for d in dilations)


class CausalConv1d(nn.Conv1d):
    def __init__(self, c_in, c_out, kernel, dilation=1, groups=1):
        super().__init__(c_in, c_out, kernel, dilation=dilation, groups=groups)
        self.left = (kernel - 1) * dilation

    def forward(self, x):                       # x: (B, C, T)
        return super().forward(F.pad(x, (self.left, 0)))


class TemporalTrunk(nn.Module):
    def __init__(self, n_feat: int, units: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_feat, self.units = n_feat, units
        self.branch = CausalConv1d(n_feat, n_feat * units, KERNEL, DILATIONS[0], groups=n_feat)
        self.drop = nn.Dropout(dropout)
        self.fuse = CausalConv1d(n_feat * units, units, KERNEL, DILATIONS[1])
        self.core = CausalConv1d(units, units, KERNEL, DILATIONS[2])
        self.out_dim = units

    def forward(self, w):                       # w: (B, T, n_feat)
        x = w.transpose(1, 2)
        x = self.drop(F.gelu(self.branch(x)))
        x = F.gelu(self.fuse(x))
        x = F.gelu(self.core(x))
        return x[:, :, -1]


class TemporalPolicyNet(nn.Module):
    """Temporal trunk (+ agent state after the trunk) -> class logits (+ bounded SL/TP sigma-multiple heads).
    Same forward contract as ``learn.PolicyNet``: returns (logits, reg or None)."""

    def __init__(self, n_feat: int, win: int, n_cls: int, n_reg: int = 0, n_state: int = 0, units: int = 2,
                 dropout: float = 0.1, reg_bounds=((0.25, 3.0), (0.5, 3.0))):
        super().__init__()
        self.n_feat, self.win, self.n_state = n_feat, win, n_state
        self.trunk = TemporalTrunk(n_feat, units, dropout)
        d = self.trunk.out_dim + n_state
        self.cls = nn.Linear(d, n_cls)
        self.n_reg = n_reg
        if n_reg:
            self.reg = nn.Linear(d, n_reg)
            self.register_buffer("lo", torch.tensor([b[0] for b in reg_bounds[:n_reg]], dtype=torch.float32))
            self.register_buffer("hi", torch.tensor([b[1] for b in reg_bounds[:n_reg]], dtype=torch.float32))

    def split(self, x):
        """Accepts (B, T, F) windows, or a flat (B, T*F + n_state) vector that is reshaped back to (B, T, F)."""
        if x.dim() == 3:
            return x, None
        w = x[:, : self.win * self.n_feat].reshape(-1, self.win, self.n_feat)
        s = x[:, self.win * self.n_feat:] if self.n_state else None
        return w, s

    def forward(self, x):
        w, s = self.split(x)
        z = self.trunk(w)
        if self.n_state:
            z = torch.cat([z, s], 1)
        logits = self.cls(z)
        if not self.n_reg:
            return logits, None
        return logits, self.lo + (self.hi - self.lo) * torch.sigmoid(self.reg(z))


def param_table(model: nn.Module) -> list[dict]:
    """Per-module trainable parameter counts (leaf modules with parameters), plus the total."""
    rows = []
    for name, m in model.named_modules():
        own = [p for p in m.parameters(recurse=False) if p.requires_grad]
        if own:
            rows.append({"layer": name, "type": type(m).__name__, "params": int(sum(p.numel() for p in own))})
    rows.append({"layer": "TOTAL", "type": "", "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad))})
    return rows
