# ZigZag "hindsight expert" study — demonstrations and the confirmation-lag gap

**DEVELOPMENT.** Preparation for learning-from-demonstrations (the RL training itself is Musashi's
lane). We deliver the expert demonstrations, the realistic ceiling of any causal detector, and the
gap a learned policy must close. Nothing here is trading-eligible.

Tables: `results/zigzag/TABLAS_ZIGZAG.md`. Figures:
- Pareto front: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/zigzag/zigzag_frente_pareto.png
- Hindsight vs confirmed vs smoke-probe detector (the gap): https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/zigzag/zigzag_brecha.png
- Sharpe vs pivot-timing error: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/zigzag/zigzag_ruido_temporal.png
- Sharpe vs pivot-detection precision: https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/zigzag/zigzag_deteccion.png
- Demonstrations: `results/zigzag/demonstrations/expert_trades.csv` + `MANIFEST.json`

## 1. The indicator (`app/short_horizon/zigzag.py`)

MetaTrader ZigZag with Depth, Deviation (pips) and Backstep, replicated from the public MetaQuotes
ZigZag.mq5 two-pass structure: candidate low/high maps (Depth-window extreme, Deviation test of the
current bar against that extreme, Backstep erasure of earlier candidates), then alternating selection
that **replaces the last extreme of the current leg** until the opposite extreme appears. Not diffed
bit-for-bit against a specific file version; point = 1 pip. Tests on hand-built series cover
alternation, the Deviation/Depth semantics, replacement, confirmation, blocks, orders, delay/jitter and
the sealed demonstrations writer (`tests/unit_tests/test_short_horizon_zigzag.py`).

**The indicator is NON-CAUSAL. It repaints:** a pivot is known only after a Deviation-sized reversal
and Depth/Backstep bars, and the last extreme is rewritten until the leg ends. It must never be a model
input; it defines labels and the HINDSIGHT_NON_CAUSAL_EXPERT.

Two versions: **(a) hindsight** — final repainted pivots at their own bar; **(b) causal-confirmed** —
the same pivots timestamped at the first bar at which the pivot is present in the ZigZag computed on
the prefix and locked by the opposite extreme beyond Backstep reach (replayed from the previous pivot).
(b) is the realistic ceiling for any causal detector.

## 2. Strategy and rules

EUR/USD H1 lake file (pinned digest), 2020–2024, dev session calendar (entry delay 12 h, forced flat
12 h). At each valley: long; at each peak: short; one position at a time (the block ends at the next
signal or the forced flat). Units from rel_volume 0,10 × percentual_order_size 1,0 at 1:100 (1:10 /
1:50 reported). SL = entry ∓ 2·σ̂ (σ̂ = causal expected 24 h volatility, m fixed a priori). TP: (a) the
next pivot level ∓ 1 pip (**hindsight, labelled**); (b) k·σ̂ with k chosen on 2020–2022 from
{0,5; 1; 1,5; 2}. Realistic costs, order rules (minimum distance, effectiveness ≤ 1,5 × causal range),
≥ 3 orders/week and ≤ 4/day; front points outside the rule are REJECTED.

## 3. DEAP NSGA-II on version (a), 2020–2022 (objectives: Sharpe ↑, max drawdown % ↓)

16 individuals, 15 generations; 16 front points, all accepted by the frequency rule (7,8–14,8
orders/week). Decision Sharpe 15,4–17,9 with max DD 4,9–8,6 %; validated on 2023 at 16,9–19,5. Chosen
by 2023 Sharpe: **Depth 3, Deviation 36,3 pips, Backstep 2** (6 371 pivots; 2023 Sharpe 19,48; 13,0
orders/week). Confirmed once on 2024: Sharpe 18,91, max DD 3,2 %, 14,7 orders/week, 97 % TP exits
(1:10 / 1:50: 18,9 / 18,9). The hindsight net profit compounds to absurd levels (10⁸–10²⁵ USD): it is
an expert, not a result.

## 4. The gap (a) − (b), same parameters

| Params | Confirmation lag median / mean / p90 (bars) | Period | (a) Sharpe | (b) Sharpe | Gap |
|---|---|---|---:|---:|---:|
| D3 / 36,3 / B2 (chosen) | 6 / 6,9 / 12 | 2020–2022 | 17,09 | −1,03 | 18,1 |
| | | 2023 | 19,48 | −2,14 | 21,6 |
| | | 2024 | 18,91 | −2,16 | 21,1 |
| D4 / 29,7 / B3 | 8 / 9,8 / 17 | 2024 | 19,25 | −2,57 | 21,8 |
| D3 / 30,6 / B4 | 9 / 10,9 / 18 | 2024 | 18,92 | −1,31 | 20,2 |

(b) exits 97–99 % at the time stop (the next confirmed signal arrives before a 2·σ̂ TP), k = 2 chosen
on 2020–2022 (all k lose).

**Decomposition from the timing-noise study** (hindsight TP kept, signals delayed/jittered): delaying
the pivots by the median lag (6 bars) still leaves Sharpe ≈ 8–9, and 12 bars ≈ 6; jitter ±4 bars
around the true pivot ≈ 8. **Timing alone costs about half the expert (19 → 6–9); the other half is the
knowledge of the next pivot level used as TP**, which the confirmed version does not have (6–9 → −2).
No break-even exists on the timing axis within 12 bars: with the future TP known, late entries still
pay. Any learned policy must therefore reproduce both the pivot timing and the **extent** of the
following leg.

## 5. Causal pivot detector — SMOKE_PROBE_WEAK_DETECTOR (not evidence about the achievable ceiling)

**Owner correction 2026-10-19.** What ran here is a 5,7-second MLP on 20 hand-made features at t with no
lookback window, substitute data, 30 epochs and one seed: a smoke probe of the pipeline (seal, adapter,
GPU admission, strategy wiring), **not an experiment about what a causal detector can achieve**. A proper
windowed model on the selected features is pending. The numbers and the seal are kept as the probe's record.

- **Target** (documented classifier target): y_t ∈ {none, valley within (t, t+k], peak within
  (t, t+k]} from the hindsight pivots of the chosen parameters; k ∈ {2, 4} bars; one signal per k bars.
- **Features at t only** — *substitute causal lake features*, not Musashi's 19-feature manifest: the
  I6-DMO input bundle (INPUT.npz with the 2ff34fa2… manifest) is resolved from a private deployment
  spec and is **not present on this host** (the local i6dmo scratch root is empty here; only test fixtures were
  found), so that arm is BLOCKED_ON_THIS_HOST and the detector uses 20 causal price features
  (multi-scale returns, position in the 24/120 h range, draw-down/up, Parkinson 6/24, efficiency ratio,
  hour/dow). Labelled SUBSTITUTE_CAUSAL_LAKE_FEATURES_NOT_MUSASHI_MANIFEST in the seal.
- **Model**: MLP 64-32, class-weighted cross-entropy, 30 epochs, trained 2020–2022, threshold and k
  chosen on 2023 by the reversal strategy's Sharpe (TP = 2·σ̂ as in (b)), confirmed once on 2024.
  Predictions sealed (`DETECTOR_SEAL.json`, sha256).
- **GPU role used**: option (2), the coordinator's RTX 4070 Laptop (8 GB): a single `nvidia-smi` read
  showed 1,2 GB used / 0–23 % util; option (1) was not probed (no sanctioned handle for the secondary
  worker in this session); the 5090 was never touched. Admission: `crispdm-run -m 2200M -t 120s` =
  1,25 × the measured 1,74 GB RSS of a 3-epoch measurement run; peak RSS 1,74 GB, peak VRAM 35 MiB,
  wall 15,8 s (training 5,7 s). One job, no monitoring loop.

| k | Period | AUC valley / peak (base 0,5) | Log-loss (base rate) | Precision at τ = 0,6 | Sharpe | Orders/wk | Valid |
|---:|---|---|---|---:|---:|---:|---|
| 2 | 2020–2022 | 0,688 / 0,691 | 1,039 (0,948) | 0,425 | −0,11 | 9,6 | sí |
| 2 | 2023 | 0,664 / 0,664 | 1,073 (0,924) | 0,360 | **−0,35** | 9,5 | sí |
| 2 | 2024 (once) | 0,671 / 0,673 | 1,061 (0,941) | 0,397 | **−0,61** | 8,7 | sí |
| 4 | 2023 | 0,571 / 0,564 | 1,099 (1,097) | 0,37–0,39 | −4,4 … +0,2 | 1–14 | mostly no |

The probe ranks pivot bars above chance (AUC 0,66–0,69) but its calibrated log-loss is worse than the
base rate (class weighting) and its precision at the trade threshold is 0,36–0,43 against a k-bar base
rate of 0,40. **A weak probe did not close the gap (−0,35 / −0,61 vs (b) −2,1 / −2,2 vs (a) 19); a proper
windowed model on the selected features is pending.** Nothing here bounds the achievable ceiling.

## 6. Demonstration dataset for Musashi

`results/zigzag/demonstrations/expert_trades.csv`: 3 791 expert trades 2020–2024 (t_open, side, units,
entry, SL, TP, t_close, reason), sha256 `352ddf6a55300864e11abe790b8bad396742abe4fd4d2aaef1012127ba544f76`,
manifest with parameters (Depth 3, Deviation 36,3, Backstep 2), data digest, costs, sizing, and the
explicit label **HINDSIGHT_NON_CAUSAL_EXPERT**: the trades use the final repainted pivots and the next
pivot level as TP and are not reproducible causally.

## Limits
- The MT ZigZag replication follows the published algorithm structure, not a byte-level diff.
- One instrument; 2024 looked at once (DEVELOPMENT). Costs ASSUMED_NOT_BROKER_VERIFIED; clock under repair.
- The learned arm on Musashi's 19-feature manifest is BLOCKED_ON_THIS_HOST (bundle absent); the
  substitute-feature run is a SMOKE_PROBE_WEAK_DETECTOR and says nothing about the manifest or the ceiling.
- Compute: NSGA 13 + 85 s, confirmed 89 s, noise 57 s, demos < 10 s CPU; GPU as above.
