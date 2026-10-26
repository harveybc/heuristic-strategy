"""Ensemble strategy frozen FROM ITS MEMBERS (audit A5-8: the ensemble re-ran a NSGA-II search on the calibration block
and became a noisier, re-optimised object: INNER_B mean -1.41 vs -0.67 for the seeds). Rule: no new search. The
ensemble's strategy is the member strategy closest to the gene-wise median of the members' genes (standardised by the
members' gene spread), so it is always an actual frozen member point; non-numeric genes take the majority value."""
from __future__ import annotations

import numpy as np

RULE = ("ensemble strategy = the member (seed) strategy whose genes are closest to the gene-wise median of all members, "
        "standardised by the members' spread; no re-search on any block; ties -> the lowest seed")


def strategy_from_members(member_docs: dict) -> dict:
    """``member_docs`` = {seed: strategy dict with a numeric/boolean 'genes' mapping (and any other frozen fields)}.
    Returns {"seed": chosen seed, "strategy": its dict, "rule", "distances"}; refuses when members disagree on gene names."""
    seeds = sorted(member_docs, key=lambda s: int(s))
    if not seeds:
        raise ValueError("ENSEMBLE_NO_MEMBERS")
    names = None
    rows = []
    for s in seeds:
        g = member_docs[s].get("genes") or {}
        if not g:
            raise ValueError(f"ENSEMBLE_MEMBER_WITHOUT_GENES:{s}")
        if names is None:
            names = list(g)
        elif list(g) != names:
            raise ValueError(f"ENSEMBLE_GENE_NAMES_DIFFER:{s}")
        rows.append([float(g[k]) if isinstance(g[k], (int, float, bool, np.floating, np.integer)) else np.nan for k in names])
    G = np.asarray(rows, float)
    med = np.nanmedian(G, 0)
    spread = np.nanstd(G, 0, ddof=0)
    spread[~np.isfinite(spread) | (spread == 0)] = 1.0
    D = np.nansum(((G - med) / spread) ** 2, 1)
    i = int(np.argmin(D))
    return {"seed": seeds[i], "strategy": member_docs[seeds[i]], "rule": RULE, "gene_names": names,
            "median_genes": {k: float(v) for k, v in zip(names, med)}, "distances": {s: float(d) for s, d in zip(seeds, D)}}
