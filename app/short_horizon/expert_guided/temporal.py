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

import numpy as np
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


class TorchPolicy:
    """Numpy-in/numpy-out eval-mode wrapper of a trained TemporalPolicyNet (same call contract as NumpyPolicy)."""

    def __init__(self, model: nn.Module):
        self.m = model.cpu().eval()

    def __call__(self, x):
        with torch.no_grad():
            lg, rg = self.m(torch.as_tensor(np.asarray(x, np.float32)))
        return lg.numpy(), (None if rg is None else rg.numpy())


try:                                              # SB3 is optional for E0/E1
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class TemporalExtractor(BaseFeaturesExtractor):
        """SB3 features extractor: flat obs [win*n_feat window, n_state agent state] -> reshaped (win, n_feat) ->
        temporal trunk -> concat(state). features_dim = units + n_state."""

        def __init__(self, observation_space, n_feat: int, win: int, n_state: int = 9, units: int = 2,
                     dropout: float = 0.1):
            super().__init__(observation_space, units + n_state)
            self.n_feat, self.win, self.n_state = n_feat, win, n_state
            self.trunk = TemporalTrunk(n_feat, units, dropout)

        def forward(self, obs):
            w = obs[:, : self.win * self.n_feat].reshape(-1, self.win, self.n_feat)
            return torch.cat([self.trunk(w), obs[:, self.win * self.n_feat:]], 1)
except ImportError:                               # pragma: no cover
    TemporalExtractor = None


class TorchActor:
    """Frozen deterministic SAC actor: tanh(mu(latent_pi(extractor(x)))) rebuilt from an actor state_dict (the SB3
    actor itself is not deep-copyable); eval mode, so dropout is off."""

    def __init__(self, sd, n_feat: int, win: int, n_state: int, units: int, net_arch, act_dim: int = 3):
        import gymnasium as gym
        space = gym.spaces.Box(-np.inf, np.inf, (win * n_feat + n_state,), np.float32)
        self.ext = TemporalExtractor(space, n_feat, win, n_state, units)
        self.ext.load_state_dict({k[len("features_extractor."):]: v for k, v in sd.items() if k.startswith("features_extractor.")})
        layers, d = [], self.ext.features_dim
        for h in net_arch:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.lat = nn.Sequential(*layers)
        self.lat.load_state_dict({k[len("latent_pi."):]: v for k, v in sd.items() if k.startswith("latent_pi.")})
        self.mu = nn.Linear(d, act_dim)
        self.mu.load_state_dict({k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")})
        for m in (self.ext, self.lat, self.mu):
            m.cpu().eval()

    def __call__(self, x):
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(x, np.float32)).reshape(1, -1)
            return torch.tanh(self.mu(self.lat(self.ext(z)))).numpy()[0]
