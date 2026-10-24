"""Small torch networks and the owner's training rule, shared by E0 (classifier) and E1 (policy).

Owner regularization: AdamW (weight decay 1e-4), dropout 0.1, NO BatchNorm. Owner's early stopping: the
monitor is the mean of the TRAIN loss and the INNER loss at each epoch, both measured at the epoch end in
inference mode with the same loss, the same weighting and the same checkpoint (Musashi F06); the running
(dropout-on) training loss is kept separately. Patience 10, min_delta 1e-8, the best checkpoint is restored.
Every epoch is logged with its TRAIN-INNER gap; maximum budget, best epoch and stop cause are returned.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn


class PolicyNet(nn.Module):
    """MLP trunk -> class logits (+ optional bounded regression heads for SL/TP sigma multiples)."""

    def __init__(self, n_in: int, n_cls: int, n_reg: int = 0, hidden=(128, 64), dropout: float = 0.1,
                 reg_bounds=((0.25, 3.0), (0.5, 3.0))):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.cls = nn.Linear(d, n_cls)
        self.n_reg = n_reg
        if n_reg:
            self.reg = nn.Linear(d, n_reg)
            lo = torch.tensor([b[0] for b in reg_bounds[:n_reg]], dtype=torch.float32)
            hi = torch.tensor([b[1] for b in reg_bounds[:n_reg]], dtype=torch.float32)
            self.register_buffer("lo", lo)
            self.register_buffer("hi", hi)

    def forward(self, x):
        z = self.trunk(x)
        logits = self.cls(z)
        if not self.n_reg:
            return logits, None
        return logits, self.lo + (self.hi - self.lo) * torch.sigmoid(self.reg(z))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _masked_huber(rg, r, m):
    """Per-row Huber loss over the regression targets. ``m`` is a per-row mask (n,) applied to the row's summed loss
    (the original contract) or a per-target mask (n, n_reg) (auxiliary targets with their own validity)."""
    per = nn.functional.smooth_l1_loss(rg, r, reduction="none")
    return (per * m).sum(1) if m.dim() == 2 else per.sum(1) * m


def _loss(model, X, y, R, Rm, w, device, batch=8192, train=False):
    """Mean loss over (X, y[, R masked by Rm]) in inference mode: CE (class weights w) + Huber on R."""
    model.eval()
    tot, n = 0.0, 0
    ce = nn.CrossEntropyLoss(weight=w, reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            yb = torch.as_tensor(y[i:i + batch], device=device)
            lg, rg = model(xb)
            l = ce(lg, yb).item()
            if R is not None and rg is not None:
                rb = torch.as_tensor(R[i:i + batch], device=device)
                mb = torch.as_tensor(Rm[i:i + batch], device=device)
                l += _masked_huber(rg, rb, mb).sum().item()
            tot += l
            n += len(xb)
    if w is not None:   # weighted CE normalised by the summed weights, as torch's mean reduction
        ws = w.detach().cpu().numpy()[np.asarray(y, int)].sum()
        return tot / max(ws, 1e-12)
    return tot / max(n, 1)


def fit(X, y, Xi, yi, *, seed, n_cls=3, R=None, Rm=None, Ri=None, Rmi=None, class_weight=None, hidden=(128, 64),
        dropout=0.1, lr=1e-3, wd=1e-4, batch=256, max_epochs=200, patience=10, min_delta=1e-8, device="cpu",
        log_every=None, init_state=None, net=None, sample_weight=None):
    """``net``: optional zero-argument factory returning a model with PolicyNet's forward contract (e.g. the
    temporal trunk of ``temporal.py``); default None keeps the MLP path unchanged. ``sample_weight`` (per training row,
    default None = unchanged loss): the CE term becomes sum(s * w_y * l) / sum(s * w_y), the class-weighted mean with
    every row scaled by s (multires label fitness); the monitored TRAIN/INNER losses stay unweighted by s."""
    set_seed(seed)
    n_reg = 0 if R is None else R.shape[1]
    model = (PolicyNet(X.shape[1], n_cls, n_reg, hidden, dropout) if net is None else net()).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = None if class_weight is None else torch.as_tensor(np.asarray(class_weight, np.float32), device=device)
    ce = nn.CrossEntropyLoss(weight=w)
    St = None if sample_weight is None else torch.as_tensor(np.asarray(sample_weight, np.float32), device=device)
    ce_rows = None if St is None else nn.CrossEntropyLoss(weight=w, reduction="none")
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, device=device)
    Rt = None if R is None else torch.as_tensor(R, device=device)
    Rmt = None if Rm is None else torch.as_tensor(Rm, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist, best, best_state, best_ep, wait, stop = [], math.inf, None, 0, 0, "max_epochs"
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        run, nb = 0.0, 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            lg, rg = model(Xt[idx])
            if St is None:
                loss = ce(lg, yt[idx])
            else:
                yb, sb = yt[idx], St[idx]
                cw = torch.ones_like(sb) if w is None else w[yb]
                loss = (sb * ce_rows(lg, yb)).sum() / torch.clamp((sb * cw).sum(), min=1e-12)
            if Rt is not None:
                loss = loss + _masked_huber(rg, Rt[idx], Rmt[idx]).sum() / max(len(idx), 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        tr = _loss(model, X, y, R, Rm, w, device)
        inn = _loss(model, Xi, yi, Ri, Rmi, w, device)
        score = 0.5 * (tr + inn)
        hist.append({"epoch": ep, "running_train_loss": run / max(nb, 1), "train_loss": tr, "inner_loss": inn,
                     "score": score, "gap_inner_minus_train": inn - tr})
        if score < best - min_delta:
            best, best_ep, wait = score, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                stop = "patience"
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"epochs": len(hist), "best_epoch": best_ep, "best_score": best, "stop_reason": stop,
            "max_epochs": max_epochs, "patience": patience, "min_delta": min_delta, "seconds": time.time() - t0,
            "monitor": "0.5*(TRAIN loss + INNER_A loss), both inference mode at the epoch end",
            "optimizer": {"name": "AdamW", "lr": lr, "weight_decay": wd}, "dropout": dropout,
            "hidden": list(hidden) if net is None else None, "architecture": type(model).__name__,
            "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "batchnorm": False, "batch": batch, "class_weight": None if class_weight is None else list(map(float, class_weight)),
            **({"sample_weights": {"rows": int(len(sample_weight)), "mean": float(np.mean(sample_weight)), "min": float(np.min(sample_weight)),
                                   "max": float(np.max(sample_weight))}} if sample_weight is not None else {})}
    return model, hist, info


def predict(model, X, device="cpu", batch=8192):
    model.eval()
    P, Rg = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            lg, rg = model(torch.as_tensor(X[i:i + batch], device=device))
            P.append(torch.softmax(lg, 1).cpu().numpy())
            if rg is not None:
                Rg.append(rg.cpu().numpy())
    return np.concatenate(P), (np.concatenate(Rg) if Rg else None)


def fit_temperature(logp_inner: np.ndarray, y_inner: np.ndarray) -> float:
    """Temperature T > 0 minimising the INNER_A log-loss of softmax(log p / T) (grid + refine)."""
    def ll(T):
        z = logp_inner / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y_inner)), y_inner], 1e-12, 1)))
    grid = np.exp(np.linspace(np.log(0.25), np.log(8), 61))
    T = grid[int(np.argmin([ll(t) for t in grid]))]
    fine = T * np.exp(np.linspace(-0.06, 0.06, 25))
    return float(fine[int(np.argmin([ll(t) for t in fine]))])


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(P, 1e-12, 1)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class NumpyPolicy:
    """Fast per-step CPU forward pass of a trained PolicyNet (eval mode: dropout off)."""

    def __init__(self, model: PolicyNet):
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        self.W = []
        i = 0
        while f"trunk.{i}.weight" in sd:
            self.W.append((sd[f"trunk.{i}.weight"].T, sd[f"trunk.{i}.bias"]))
            i += 3
        self.cw, self.cb = sd["cls.weight"].T, sd["cls.bias"]
        self.reg = ("reg.weight" in sd)
        if self.reg:
            self.rw, self.rb, self.lo, self.hi = sd["reg.weight"].T, sd["reg.bias"], sd["lo"], sd["hi"]

    @staticmethod
    def _gelu(x):
        from scipy.special import erf          # exact GELU, as torch's default
        return 0.5 * x * (1 + erf(x / math.sqrt(2)))

    def __call__(self, x):
        z = np.asarray(x, np.float32)
        for W, b in self.W:
            z = self._gelu(z @ W + b)
        lg = z @ self.cw + self.cb
        r = None
        if self.reg:
            r = self.lo + (self.hi - self.lo) / (1 + np.exp(-(z @ self.rw + self.rb)))
        return lg, r
