"""Common-random-number noise for oracle degradation (seeded, reusable across levels)."""
from __future__ import annotations

import numpy as np

from .blocks import AMBIGUOUS, HIGH_FIRST, LOW_FIRST

AMBIGUOUS_COIN_SEED = 20261018


def draws(seed: int, n_blocks: int, n_horizons: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"z_high": rng.standard_normal(n_blocks), "z_low": rng.standard_normal(n_blocks),
            "u_flip": rng.uniform(size=n_blocks),
            "z_path": rng.standard_normal((n_blocks, n_horizons))}


def ambiguous_coin(n: int) -> np.ndarray:
    bits = np.random.default_rng(AMBIGUOUS_COIN_SEED).integers(0, 2, n)
    return np.where(bits == 1, HIGH_FIRST, LOW_FIRST)


def resolve_ambiguous(ft: np.ndarray, coin: np.ndarray) -> np.ndarray:
    ft = np.asarray(ft)
    return np.where(ft == AMBIGUOUS, coin, ft).astype(int)


def apply_range_noise(H, L, ft, d, sigma, p_flip):
    H_hat = np.asarray(H) + np.asarray(sigma) * d["z_high"]
    L_hat = np.asarray(L) + np.asarray(sigma) * d["z_low"]
    flip = d["u_flip"] < p_flip
    ft_hat = np.where(flip, -np.asarray(ft), ft).astype(int)
    return H_hat, L_hat, ft_hat


def apply_path_noise(targets, d, sigma):
    return np.asarray(targets) + np.asarray(sigma)[:, None] * d["z_path"][:, :targets.shape[1]]


def blend(truth, naive, w):
    """``naive + w (truth - naive)``: w = 0 is the naive, w = 1 the exact oracle.

    A deterministic, model-like degradation: errors shrink toward the naive
    instead of being independent of it. Where the naive is absent the forecast
    is absent unless w == 1. Absent truth stays absent.
    """
    truth = np.asarray(truth, float)
    naive = np.broadcast_to(np.asarray(naive, float), truth.shape)
    if w == 1.0:
        return truth.copy()
    return naive + w * (truth - naive)


def flip_to_accuracy(ft, eligible, u, target):
    """Flip first-touch labels so accuracy on ``eligible`` rows is ``target`` exactly.

    The k = round((1 - target) * n) eligible rows with the smallest ``u`` are
    flipped, so a lower target flips a superset of a higher one (common random
    numbers) and rows outside ``eligible`` are never touched.
    """
    ft = np.asarray(ft).astype(int)
    idx = np.flatnonzero(np.asarray(eligible, bool))
    k = int(round((1.0 - float(target)) * len(idx)))
    order = idx[np.argsort(np.asarray(u)[idx], kind="stable")]
    out = ft.copy()
    out[order[:k]] = -out[order[:k]]
    return out


def path_draws(seed: int, n_blocks: int, horizons: int, stream: int = 0) -> np.ndarray:
    """Standard normals for every predicted price of every block's path.

    ``stream`` 0 is the original i.i.d. stream (unchanged); other streams are
    independent (used for the random-walk error component).
    """
    key = [int(seed), int(horizons), 2026_10_19] + ([int(stream)] if stream else [])
    return np.random.default_rng(key).standard_normal((n_blocks, horizons))


def blend_path(truth, ref, w, eps_sigma, z):
    """Model-like A1: ``ref + w (truth - ref) + eps``; w = 0 is persistence, w = 1 the truth."""
    truth = np.asarray(truth, float)
    ref = np.asarray(ref, float)[:, None]
    return np.where(np.isfinite(truth), ref + w * (truth - ref) + eps_sigma * np.asarray(z), np.nan)


def random_walk_error(z, step):
    """A2: Gaussian random walk along the horizon (std grows like sqrt(h), autocorrelated)."""
    return step * np.cumsum(np.asarray(z, float), axis=1)
