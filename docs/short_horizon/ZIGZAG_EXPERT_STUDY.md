# ZigZag "hindsight expert" study — demonstrations and the confirmation-lag gap

**DEVELOPMENT.** Period labels (Musashi audit, 2026-10-19): 2020–2022 decision; **2023 = JOINT development**
(it chooses the expert); **2024 = PREVIOUSLY OBSERVED development** — never a confirmation. Sections 1–7
are v1; section 8 (NSGA-II v2) supersedes the v1 expert and its parameter search.

Preparation for learning-from-demonstrations (the RL training itself is Musashi's
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
orders/week). 2024 (PREVIOUSLY OBSERVED development, not a confirmation — wording corrected 2026-10-19): Sharpe 18,91, max DD 3,2 %, 14,7 orders/week, 97 % TP exits
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
  chosen on 2023 (JOINT development) by the reversal strategy's Sharpe (TP = 2·σ̂ as in (b)), read on 2024 (PREVIOUSLY OBSERVED development).
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
| 2 | 2024 (previously observed) | 0,671 / 0,673 | 1,061 (0,941) | 0,397 | **−0,61** | 8,7 | sí |
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

## 7. PARITY_CHECK — the sealed series replayed in gym-fx

The expert trade series (sha256 `352ddf6a…`) and the smoke-probe series were replayed as candidate
policies in the gym-fx simulation through agent-multi's replay worker (`sweep_worker.py` @ `e4d01fa0`,
read-only worktrees of both repositories; the bundle-level `sweep` wrapper needs a fitted policy
bundle, so its simulation half was driven directly with a flat gym-fx configuration). Same
instrument, 2024, same bars (lake, pinned digest), fixed 90 000 units (no compounding), matched
per-fill costs (3,5 USD/lot/side; half spread 0,3 pip + 0,2 pip slippage as `slippage_perc`), no swap.
Because gym-fx has no per-trade SL/TP, the policy compared is **hold-to-reversal** (long at a valley /
short at a peak, fills at the next bar open) in both simulators — labelled PARITY_ONLY_NOT_A_STRATEGY;
it is not the expert's SL/TP result of §3. Table: `results/zigzag/parity/PARITY_TABLE.md`, raw
`PARITY.json`.

| Series | Metric | Ours | gym-fx | Δ % |
|---|---|---:|---:|---:|
| expert | Net USD | 143 408 | 147 351 | +2,7 % |
| expert | Sharpe | 12,92 | 13,80 | +6,8 % |
| expert | Max DD USD / % | 805,5 / 3,92 | 805,1 / 3,90 | −0,05 % |
| expert | Trades | 1 279 blocks (53 clipped at a session close) | 1 287 closed trades (1 288 position changes) | |
| probe | Net USD | 3 368 | 8 082 | +140 % |
| probe | Sharpe | 0,74 | 1,33 | |
| probe | Max DD USD | 5 458 | 4 577 | −16 % |
| probe | Trades | 537 blocks | 232 closed trades (233 changes) | |

Reading. The expert series agrees to 2,7 % in net and to the dollar in max drawdown: the price path,
the next-bar-open fill and the per-fill costs are the same in both simulators. The residual is the
**session calendar**: our simulator force-flats at every session close (53 expert blocks clipped at
the weekly gap, re-entered after it, paying two extra fills) while gym-fx holds through the gap. The
probe series disagrees because its signals repeat the same direction (long, long, …): our simulator
closes and re-opens a block at every signal (537 fills, 8 216 USD of costs) while gym-fx's
`legacy_directional_v1` ignores a same-direction decision (232 trades) — a contract difference in
**what a repeated decision means**, not a price or cost discrepancy. Every other difference
(intrabar stop-before-TP rule, limit trade-through, hour-dependent spread, swap with a triple day,
compounding sizing, ruin stop-out, daily cap, frequency rule) is disabled or absent on one side and
listed in `PARITY.json`. Verdict: PARITY_CHECK COMPARABLE for the expert under the hold-to-reversal
contract; the SL/TP, swap and calendar features of this lane have no counterpart in gym-fx and must
stay on this side.


## 8. NSGA-II v2 — redo of the parameter search (supersedes §3 and the v1 demonstrations)

Tables: `results/zigzag/v2/TABLAS_ZIGZAG_V2.md` (generated from `FINAL.json`); read ledger
`READ_LEDGER.jsonl`. Figure (fronts per daily cap, the confirmed front, (a)-vs-(b) per (A) point, hypervolume
insets): https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/zigzag/v2/zigzag_v2_frentes_pareto.png

### 8.1 Por qué el primer NSGA-II era insuficiente

1. **Presupuesto**: 16 individuos × 15 generaciones, una semilla, sin medida de convergencia.
2. **Rangos y escala**: Depth 3–60, Deviation 1–60 pips, Backstep 1–10, en escala lineal; el punto elegido
   (Depth 3) estaba **sobre la cota inferior**. Para un experto con retrospectiva, más pivotes = más
   operaciones perfectas hasta que el tope diario se activa: el óptimo lo fija el **tope de frecuencia**, no
   la información.
3. **Ejecución**: el camino fijado (`signal_blocks` + `entry_delay_bars = 1`) ejecutaba en **b + 2** y no
   aplicaba la regla de no reabrir en las 12 h tras la reapertura de la sesión. Un pivote retrospectivo es una
   **etiqueta**, no un llenado.
4. **Definición de confirmación**: v1 esperaba al **siguiente pivote final** (+ Backstep); la regla
   cognoscible es el primer candidato opuesto (+ Backstep). Probado contra la repetición por fuerza bruta de
   cada prefijo (`confirmation_bars_fast`).
5. **Etiquetas de periodo**: 2024 se llamó "confirmación"; es desarrollo ya observado.

### 8.2 What v2 does

- **Executable path (additive, v1 byte-identical under a test)**: `signal_blocks(..., executable=True)` with
  `zigzag_opt.executable_profile()` (entry_delay_bars = 0): decision after the completed bar b, fill at the
  open of b + 1; entries before the session's operational start (12 h after a reopen) are refused. Used for
  both (A) and (B).
- **Genes**: log-scale Depth 2–200 and Deviation 1–300 pips (pip = 0,0001; **10 MetaTrader points per pip**,
  so 10–3 000 points), Backstep 1–30 with Backstep < Depth repaired (Backstep = Depth − 1, written back).
  (B) adds the TP multiple k_tp ∈ [0,25; 3].
- **Budget**: population 64, ≥ 80 generations, **3 independent seeds** per optimization; hypervolume of the
  first front per generation; converged when the last 20 generations change it < 1 %.
- **Optimizations**: (A) hindsight expert, Sharpe ↑ / max DD % ↓, with the daily cap at **1, 2 and 4**
  orders/day; (B) causally confirmed version, Sharpe ↑ / max DD % ↓ / **median confirmation lag ↓**. The
  frequency rule (≥ 3/week, ≤ 4/day), SL/TP/units rules, ruin stop-out and realistic costs are inside every
  evaluation; invalid → penalty (Sharpe −10, DD 100 %, lag 1 000).
- **Boundary rule**: a front point on a bound → widen once (Depth 1–400, Deviation 0,5–600, Backstep 0–60,
  k_tp 0,1–6) and rerun the 3 seeds.

### 8.3 Results

**Convergence.** All 21 chains (12 first runs + 9 widened reruns) stopped CONVERGED at generation 80 with a
hypervolume change of 0,000–0,007 % over the last 20 generations. The (A) fronts reach their final
hypervolume within 0–5 generations (they are tiny: 2–5 distinct objective vectors); (B) needs 15–37
generations, and its seeds end at different hypervolumes (702 978 / 705 159 / 647 122 in the widened run):
the confirmed landscape is multimodal, so the union of the three seeds is reported.

**Deviation is nearly inert.** Many parameter sets give identical objective vectors (e.g. 102 sets with
Deviation 0,5–588,7 pips at Depth 2). This is MetaTrader's own semantics: Deviation only limits how far the
current bar may sit from the Depth-window extreme, it is not a minimum swing size. Ties are collapsed to one
representative and a bound is reported only when every tied set sits on it.

**Boundary report.** First run: A4 on Depth = 2 & Backstep = 1, B4 on Backstep = 1 (A1, A2: none). Widened
reruns (A4, B4, and A2 because its raw front touched Deviation = 1 before ties were collapsed): A4 and B4 still
put points on **Backstep = 0**, which is the **structural floor** (Backstep cannot be negative), not a search
bound; Depth is interior (3 for A4, 2–22 for B4). No further widening is possible.

**(A) hindsight expert — chosen by 2023 Sharpe (JOINT development); 2024 read once (PREVIOUSLY OBSERVED):**

| Daily cap | Chosen (Depth / Deviation / Backstep) | Sharpe 2020–22 | Sharpe 2023 | Sharpe 2024 | Max DD 2024 | Orders/wk 2024 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 6 / 19,6 pips (196 pts) / 3 | 13,74 | 14,19 | 14,48 | 3,21 % | 4,6 |
| 2 | 5 / 11,7 pips (117 pts) / 3 | 15,77 | 16,98 | 17,24 | 3,21 % | 8,3 |
| 4 | **3 / 40,7 pips (407 pts) / 0** | 17,67 | 18,23 | 18,16 | 2,35 % | 16,4 |

The optimum moves to shorter Depth and Backstep as the business cap relaxes — the expert's "best" parameters
are set by the cap, as suspected; demonstrations must state the cap they were made under.

**(B) causally confirmed version — chosen by 2023 Sharpe:** Depth 20 / Deviation 214,6 pips (2 146 pts) /
Backstep 5 / k_tp 1,12: Sharpe 0,43 (2020–22), 0,52 (2023), **−1,01 (2024)**, max DD 48,8 %, median lag 20
bars. The whole (B) front has Sharpe −2,0 … +0,65 on 2020–22 with max DD 33–80 %; short-lag points
(1–5 bars) lose, the few positive ones need lags of 20–33 bars and do not hold on 2024.

**(b) for every (A) front point (cap 4):** Sharpe −3,73 … −1,40 on 2020–22, lag 2–6 bars; the best expert
(Sharpe 17,67) has a confirmed counterpart at −1,70 (2020–22) / −1,65 (2023).

**Cost of the extra bar.** Same rules, fill at b + 1 vs b + 2: chosen B4 0,43 vs 0,34 (2020–22), 0,52 vs 0,58
(2023); the A4 expert's (b) −1,70 vs −1,64; v1's (b) −1,38 vs −0,90 (2020–22). The extra bar costs at most
≈ 0,1 Sharpe and sometimes helps: it neither created nor hides an edge. The reopen rule refused 48–479 entries
per period.

**Confirmation lag, v1 vs v2 definition** (v1 parameters, 3 899 pivots 2020–22): median 6 → 5 bars, mean
6,79 → 6,07, p90 11 → 10. v1 overstated the lag by about one bar.

**Demonstrations v2** (`results/zigzag/v2/demonstrations_v2/`): 4 185 expert trades 2020–2024 for Depth 3 /
Deviation 40,7 pips / Backstep 0 at cap 4, executable path, sha256
`38125b0b368a29b425b4f4ac192fe380b343894fbceedfe17c1e9b85c6564a24`, label HINDSIGHT_NON_CAUSAL_EXPERT. The v1
file (`352ddf6a…`) is kept unchanged as v1.

### 8.4 Invalidation note (Musashi F08)

The expert hash changed (v1 `352ddf6a…` → v2 `38125b0b…`). Every **target-dependent** stage of the timing
selection that used the v1 expert as its target — labels, target statistics, feature screening against
the target, and any model fitted or selected on them — is **invalidated** and must be rerun against v2
(or explicitly pinned to v1). Stages that do not read the target (feature availability, causality audits,
data manifests) are unaffected.

### 8.5 Limits

- 2023 chose every point (joint development); 2024 was already read in v1 and once more per chosen point
  here (ledger). No held-out confirmation exists in this study.
- Deviation is a near-redundant gene under MT semantics; the effective search is over Depth and Backstep.
- Backstep = 0 is a floor of the algorithm, so the A4/B4 optima touching it are structural, not a search
  artefact that a wider range could move.
- Costs ASSUMED_NOT_BROKER_VERIFIED; vendor clock under repair; one instrument.
- Compute: 21 chains × 5 184 evaluations (≈ 1 000–4 500 simulations each after the cache), 85-s CPU chunks
  under `crispdm-run -m 272M` (1,25 × the 212 MB peak measured inside a chunk), 4 chains in parallel, no GPU.
