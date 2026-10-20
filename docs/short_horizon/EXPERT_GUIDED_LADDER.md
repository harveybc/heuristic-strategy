# Expert-guided learning ladder E0–E2 (ZigZag v2 hindsight expert) — closure 2026-10-19

DEVELOPMENT only. TEST 2025 never read. No order sent anywhere. No DOIN change, no change to Musashi's checkouts
or campaign. Terminology: learning from hindsight-expert demonstrations / interactive imitation (DAgger), not
human-preference RLHF.

**Outcome: E0 shows a learnable statistical signal (gate passed); E1 does not (gate failed) → the pre-registered
ladder STOPS at E1. E2 was then run as a DECLARED DEVIATION by lead decision ("E2 run by lead decision after E1
gate failure; not a gate pass"): demonstrations do not make SAC profitable or consistently better — SAC+demo beats
plain SAC on INNER_B (−2.89 vs −3.72) but loses on 2024 (−3.08 vs −2.56); both overfit TRAIN (Sharpe 8–9).**

## Inputs and pins

| Object | Identity |
|---|---|
| Bars | admitted root `eurusd.parquet` sha256 `c80847bb…`, UTC bar start, available +1 h (`bar_source.load_ohlc(source="admitted")`) |
| Calendar | complement of `CLOSURE_INTERVALS.json` `71078e8f…` (every admitted bar lies in a segment); entry delay 12 h after reopen, forced flat 12 h before closure |
| Observations | Musashi's prepared `train.csv` `3d8a9dcd…` / `validation.csv` `52ef0adc…` of the closed campaign (CONTRACT `37e06192…`, selection `553fc047…`), read-only; 15 inputs, 13 observations at stride 2 incl. the current completed bar, causal rolling z-score over rows [i-256, i); DATE_TIME = bar start + 1 h, verified by OHLC equality on every row |
| Agent state (E1) | position, unrealized P&L / σ, holding time, SL and TP distances / σ, hours since operational start, hours to forced flat, entries used today, entry-gate flag |
| Expert | MetaTrader ZigZag Depth 3 / Deviation 40.7 pips / Backstep 0, executable path (decide after bar b, fill at open b+1), cap 4/day; recomputed on the admitted clock (F08): demonstrations `v2-admitted` sha256 `25f89856…`, 4,237 trades (v2 on the lake label clock `38125b0b…`, 4,185 trades, kept) |
| Confirmed comparator | same pivots signalled at the knowable confirmation bar, TP 1.5σ (v2 (b) choice), SL 2σ |
| Costs | `RealisticProfile` (hour-dependent spread, market/stop slippage, commission per side, swap with triple rollover), ASSUMED_NOT_BROKER_VERIFIED |
| Sizing | units = ⌊1.0 × 0.10 × balance × 100 / price⌋, integer, minimum 1 |
| Periods | TRAIN 2020–2022 fit · INNER_A 2023-H1 monitor/calibration/threshold · INNER_B 2023-H2 scored after freeze (2023 = joint development: it chose the expert) · 2024 previously observed development, read once after the freeze |

Population (rows by period, after exclusions counted by reason in `E0_TABLAS.md`): TRAIN 14,562 · INNER_A 2,461 ·
INNER_B 2,436 · 2024 4,871. Exclusions: closed entry gate (business mask, Musashi V3-01) 3,913 / 650 / 650 / 1,350;
scaling warm-up 256 (TRAIN); no causal σ 109 (TRAIN); censored labels 3 (2024); label support crossing the period
end 3 / 6 / 8 / 0.

## E0 imitability (pre-registered gate G1 ∧ G2) — PASS

Student: MLP (128, 64), dropout 0.1, AdamW wd 1e-4, no BatchNorm, unweighted CE, owner's stopping rule (mean of TRAIN
and INNER_A loss in inference mode, patience 10), 3 seeds; temperature and signal threshold chosen on INNER_A only.

| INNER_B | value |
|---|---|
| G1 log-loss − TRAIN prior (hindsight target, calibrated) | −62.7 / −73.4 / −65.3 millinats (3/3 seeds) |
| G2 PR-AUC long / short (prevalence 0.115) | ≈ 0.28 / 0.25, weekly block-bootstrap 95 % lower bound of the excess > 0 in 3/3 seeds |
| Confirmed (reactive) target, same model | log-loss −207 to −219 millinats, PR-AUC ≈ 0.55 (a causal function of price is far easier) |

Executed in the v2 block simulator (Sharpe daily √365; mean of 3 seeds for students and random):

| Policy | TRAIN | INNER_A | INNER_B | 2024 (1 read) | 2024 net USD | 2024 orders/wk |
|---|---:|---:|---:|---:|---:|---:|
| Hindsight expert (TP = next pivot) | 18.23 | 18.58 | 20.24 | 18.07 | 4.4e9 | 16.1 |
| Hindsight timing, causal TP 1.5σ | 16.32 | 15.78 | 17.83 | 15.99 | 8.6e7 | 16.3 |
| Confirmed expert | −1.60 | −0.98 | −3.58 | −3.88 | −8,306 | 16.0 |
| **Student of the hindsight expert** | 4.79 | −0.10 | −0.23 | **+0.94** (+0.85 / +0.59 / +1.37) | +3,602 | 3.8 |
| Student of the confirmed expert | −1.35 | 0.57 | −1.74 | −0.74 | −3,467 | 6.1 |
| Random at the TRAIN prior | −2.02 | −3.16 | −2.02 | −3.60 | −7,957 | 15.8 |
| No trade | 0 | 0 | 0 | 0 | 0 | 0 |

Imitability gap on INNER_B: expert 20.24 · confirmed −3.58 · student −0.23 → expert − student ≈ 20.5 Sharpe.
The student overfits (TRAIN Sharpe 4.8, TRAIN log-loss −300 millinats vs −65 on INNER).
Figures: [gap](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_brecha_imitabilidad.png),
[curves](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_curvas_entrenamiento.png),
[PR-AUC excess](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/e0_prauc_exceso.png),
[tables](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e0/E0_TABLAS.md).

## E1 behaviour cloning + DAgger-lite (pre-registered gate I1 ∧ I2) — FAIL → STOP

Environment `app/short_horizon/expert_guided/env.py` (tests first: 14 tests on fill at next open, SL/TP on every
entry, units formula, 12 h reopen refusal, forced flat, 4 entries/day cap, reverse = close + entry both costed,
same-direction ignored, stop-before-TP, ledger reconciliation, label support/censoring, oracle, prefix invariance).
Parity: the expert replayed in this env gives exactly the block simulator's Sharpe (18.58 / 20.24 / 18.07).
Policy: same MLP on [window, agent state] → {hold, long, short, close} + TP/SL σ multiples; class-weighted CE
(1/√freq) + Huber; round 0 = BC on the expert's own trajectory; rounds 1–3 = the policy acts, the hindsight
LegOracle relabels each visited state in the policy's own position/gate; final = round 3 (fixed in advance).

| Policy (bar env) | TRAIN | INNER_A | INNER_B | 2024 (1 read) |
|---|---:|---:|---:|---:|
| Expert demonstrations | 18.23 | 18.58 | 20.24 | 18.07 |
| Leg oracle (relabeller) | 11.81 | 13.09 | 12.43 | 12.36 |
| Confirmed expert | −1.55 | −0.80 | −3.64 | −3.83 |
| BC round 0 (3 seeds) | 1.08 | −0.93 | −1.43 | not read |
| **DAgger round 3 (3 seeds)** | 6.37 | −3.30 | **−4.00** | **−5.21** |
| Random at the TRAIN action prior | −1.29 | 0.32 | −3.27 | −1.64 |

- I1: imitation log-loss on its own INNER_B states minus prior = +0.09 / +0.07 / +0.04 nats (fails 3/3). It falls
  every round (BC +6.5 → +0.35 → +0.19 → +0.07): DAgger does reduce the covariate shift.
- I2: −4.00 < confirmed −3.64 and < random −3.27 (fails).
- Mechanism: the relabeller teaches closing wrong-side positions; the student cannot tell the side, so it churns
  ~15 entries/week (2024: 809 entries, 708 own closes, 168 refused by the daily cap); gross is already −2,298 USD
  before 5,795 USD of costs. TRAIN Sharpe rises 1.6 → 6.4 while INNER falls (overfit).
Figures: [Sharpe by round](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_sharpe_por_ronda.png),
[imitation and frequency by round](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_imitacion_por_ronda.png),
[curves per round](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/e1_curvas_entrenamiento.png),
[tables](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e1/E1_TABLAS.md).

## E2 — SAC with vs without demonstrations (DECLARED DEVIATION, not a gate pass)

Lead decision after the E1 gate failure: the owner's question is whether expert guidance helps SAC versus SAC
without it, and E2 learns from reward. Fixed design ledgered before the first fit (`e2/run/TRIAL_LEDGER.jsonl`,
event `DECLARED_DEVIATION`); nothing tuned on INNER_B or 2024. Same bar env, inputs, business rules and costs as
E0/E1; episode = one random TRAIN week, capital 10,000 USD; reward = net mark-to-market change, UNSHAPED; action =
[target position with a 1/3 dead band, TP σ in [0.25, 3], SL σ in [0.5, 3]]. SB3 2.9.0 SAC (no sb3_contrib),
`DemoSAC` subclass (`app/short_horizon/expert_guided/sac_demo.py`): with zeros it is SAC's own update (arm SAC);
arm SAC_DEMO adds a demo replay (25 % of each batch, linear decay to 0 over 50k updates) and a Q-filtered BC actor
term (λ 1.0, same decay) on the expert's trajectory (admitted clock, 18,255 TRAIN transitions). Extractor
Linear(204,128)+ReLU+Dropout 0.1, [64, 64], AdamW wd 1e-4, no BatchNorm; 100k transitions, 3 seeds; early stopping
on the frozen-checkpoint mean of TRAIN and INNER_A net Sharpe every 5k, patience 10.

| Policy (bar env, Sharpe) | TRAIN | INNER_A | INNER_B | 2024 (1 read) | 2024 net USD | 2024 entries/wk | 2024 gross / costs USD |
|---|---:|---:|---:|---:|---:|---:|---|
| SAC without demonstrations (3 seeds) | 7.82 | −1.37 | −3.72 | −2.56 | −6,917 | 12.3 | −1,185 / 5,732 |
| SAC + demonstrations (3 seeds) | 8.54 | −0.85 | −2.89 | −3.08 | −7,418 | 12.8 | −779 / 6,639 |
| E0 hindsight student (reference, same accounting) | 4.96 | −0.16 | 0.10 | 1.23 | +5,239 | **2.6 (invalid < 3/wk)** | 8,531 / 3,293 |
| Confirmed expert | −1.55 | −0.80 | −3.64 | −3.83 | −8,262 | 16.0 | 994 / 9,256 |
| Random at the TRAIN action prior | −1.29 | 0.31 | −3.27 | −1.64 | −5,957 | 10.7 | 151 / 6,108 |
| Expert as representable in the action space | 16.68 | 16.58 | 18.33 | 16.44 | — | 16.2 | — |

- Per seed, INNER_B: SAC −4.80 / −3.43 / −2.94; SAC_DEMO −2.90 / −2.38 / −3.39. 2024: SAC_DEMO −1.7 / −2.6 / −5.0
  (see `E2_TABLAS.md`). The INNER_B advantage of the demo arm does not hold on 2024: no consistent benefit.
- Both arms churn: in 2024 about 650–680 entries, most closed by the agent or by reversal (SAC 255 + 218, SAC_DEMO
  146 + 429 of 679); gross is already negative before costs.
- Early stopping never triggered (patience 10 not reached); the chosen checkpoints are late (80k–100k) because the
  rising TRAIN Sharpe dominates the mean; TRAIN − INNER_A gap −7.4 to −11.5 Sharpe at the chosen checkpoint.
- Q-filter: it kept about half of the demo rows early (26–34 of 64 at update 0) and fewer as the weight decayed.
- The E0 student in the bar env keeps a positive 2024 (+1.23) but trades 2.6 entries/week (fails the ≥ 3/week rule
  in this accounting; in E0's block simulator it traded 3.8/week): it is a reference, not a valid strategy.
- gym-fx parity: NOT done (not cheap: gym-fx df1b602 uses a different action contract and envelope); the
  agent-multi ab265fad / gym-fx df1b602 protections were not inspected. No change was made to either repository.
Figures: [SAC with vs without demonstrations](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e2/e2_sac_con_sin_demostraciones.png),
[early-stopping curves](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e2/e2_curvas_parada_temprana.png),
[tables](https://github.com/harveybc/heuristic-strategy/blob/satoshi/expert-guided-rl-20261019/docs/short_horizon/results/expert_guided/e2/E2_TABLAS.md).

## Limits (read before using any number)

1. 2023 is joint development (it chose the expert); 2024 was already observed in earlier studies. No pristine
   confirmation exists. The E0 2024 result (+0.94) is one read of 3 seeds with about 200 trades and no Sharpe
   interval; it is a lead, not evidence of an edge.
2. Two E0 measurement pilots (2 and 6 epochs) also scored INNER_B / 2024 with throw-away models; nothing was
   chosen from them; their ledgers are kept (`e0/run/PILOT_MEASURE_*`). The E1 and E2 pilots used `--measure`
   (TRAIN/INNER_A only). E2 is a declared deviation from the pre-registered stop, not a gate pass.
3. E0 students execute in the block simulator (hold until the next signal, SL, TP or forced flat, as the expert);
   E1 policies in the bar env with a CLOSE action. Same costs, rules and sizing; parity of the expert is exact.
4. I1 compares a class-weighted fit's probabilities with an unweighted prior: it is strict on calibration. I2 fails
   on its own, so the decision does not rest on I1.
5. One model family (MLP), one budget, declared choices: oracle re-entry needs ≥ 1σ of hindsight move left, SL
   label 2σ, TP/SL bounds [0.25, 3] / [0.5, 3] σ, student levels clipped into the effective band. Each is a trial in
   the ledgers (`e0/run/TRIAL_LEDGER.jsonl`, `e1/run/TRIAL_LEDGER.jsonl`).
6. Leverage 1:100 with rel_volume 0.10 gives 80 %+ drawdowns for losing policies and astronomical compounded TRAIN
   net for the expert; read Sharpe, not net, across policies. Costs and margin are assumptions, not broker facts.
7. This lane's ≤ 4/day, ≥ 3/week contract is separate from Musashi's ≤ 1/day SAC contract (F03).

## Resources and reproduction

Secondary worker RTX 4090 (one nvidia-smi read: 14 MiB used), `crispdm-run`, one job at a time. cgroup peaks:
E0 1.24 GB (cap 1600M = 1.25 × measured), E1 1.55 GB (pilot 1.47 GB, cap 1840M), E2 1.86 GB (pilot 1.60 GB,
cap 2000M); wall 20 s, 80 s and 45 min. Coordinator:
tests, reports and the demonstrations seal only, each under a 2G cap.

```bash
python tools/expert_guided_e0.py run --admitted <ADMITTED_ROOT> --csv-dir <SEL15_DIR> --out <run>/e0
python tools/expert_guided_e1.py run --admitted <ADMITTED_ROOT> --csv-dir <SEL15_DIR> --out <run>/e1
python tools/expert_guided_e2.py run --admitted <ADMITTED_ROOT> --csv-dir <SEL15_DIR> --e0-run <run>/e0 --out <run>/e2
python tools/expert_guided_report.py e0|e1|e2 --run <run>/eN --out docs/short_horizon/results/expert_guided/eN
python tools/expert_guided_demonstrations.py --admitted <ADMITTED_ROOT> --csv-dir <SEL15_DIR> --out <dir>
```
