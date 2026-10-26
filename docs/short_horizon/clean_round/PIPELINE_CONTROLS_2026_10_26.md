# Pipeline controls, 2026-10-26 (DIAGNOSTIC)

**Status: DIAGNOSTIC.** Nothing here is a result, a release or evidence about the market. Every store, manifest, recipe and
run directory in this report has `DIAGNOSTIC` in its identity (`recipe_DIAGNOSTIC_<KIND>_<hash>`), `scientific_release:
false`, and lives in its own directory tree, away from the real clean-round outputs. The originals (store, manifest, fold
recipe, admitted bars) were only read. No TEST row (>= 2025-01-01) was read: the store ends 2024-12-31 and every admitted
read goes through the read-time filter. OBS_2024 was touched only by the E3-M run's own single read, which is part of
the unchanged code path. It is not reported here.

## Why

The team concluded several times that "the inputs carry no predictive signal". Nobody had shown that the pipeline can
learn and exploit a signal that is **known** to exist. A null result from a pipeline that has never passed a positive
control tells us nothing about the inputs. These controls put known signals (and known non-signals) through the
**same code path** as the real experiments, so that any failure can be located inside the pipeline.

## What was run (identical code path, declared deviations)

* **Code path.** `tools/expert_guided_e3m.py run` with exactly the arguments of the WF_V13 / WF_C2_13 units
  (`tools/expert_guided_clean_round.py e3m_cmd`, arm `zz_hold` with labels `zigzag`, closer rule `hold_through`):
  recipe-bound (`--recipe`, fold `wf2022`), `--variants FULL --units 2 --depth 2 --head 16 --dropout 0.3 --wd 0.01
  --max-epochs 1000` (early stopping patience 10, the inherited rule), `--scaling fixed_train --nonstationary diff
  --agent-state none --exec-contract sac --min-orders-per-week 2.0 --strategy-fitness sharpe_lcb`. The contract is
  `ACTION_CONTRACT_v1_3.json` (identity `action_564e6b858bb08951`) or `ACTION_CONTRACT_C2_v1_3.json`
  (`action_5540240a4750af96`). C2 uses a stamped outcome table built by `tools/expert_guided_c2.py outcomes`
  (engine replay). Three DAgger rounds. The fold is `wf2022`: TRAIN 2020-01..2021-07, INNER_A 2021H2 (early stopping,
  Platt, strategy search), INNER_B = the 2022 validation year (scored after the freeze).
* **Epochs.** At most 1000 epochs with patience 10, as in the real units. Every fit stopped by patience. Openers stopped
  at 28-150 epochs and closers at 2-123 (per-fit numbers are in each `SUMMARY_*.json`).
* **Declared reductions.** One seed (20261019) per unit; the real stages use 5 seeds and decide by the seed ensemble.
  There is no seed ensemble here: every number is a single seed's frozen policy. CPU only, `OMP_NUM_THREADS=4` for
  training (the first attempt with 1 thread was too slow; I stopped it after 3 fits and reran it from scratch).
  Memory cap 3G under `crispdm-run`. Training runs went strictly one at a time. Short analyses that train nothing
  (oracle replays, head probes, learner checks, trading reports; cap 2-3G) ran next to a training job.
* **Derived stores.** Each derived store keeps only `timestamp`, OHLC and the recipe's 121 columns. The derived recipe
  is the fold recipe with the scaler constants re-fitted on the derived store's fit block, using the loader's own
  rules (`tools/expert_guided_wf_recipe.fold_constants`). For the cheat and noise stores only `act.p_hit` changed
  constants, so all other columns are byte-identical in effect. NaN patterns are kept, so the eligible rows and the C2
  outcome-table stamp match the real store.
* **Two diagnostic hooks** (`tools/pipeline_controls.py run-e3m`), both optional and recorded in `E3M.json` under
  `diagnostic_control`:
  1. For the synthetic market, the admitted-bar pin is replaced by the synthetic bars' digest. Every verification
     still runs, against that identity.
  2. A label shuffle that wraps `build_sets` (v1.3) and `c2.sets` / `outcomes.exit_sets` (C2). Labels, regression
     targets, masks and weights are permuted together; inputs and evaluation are untouched.

### Controls

| control | data | what is planted / removed |
|---|---|---|
| **cheat** | real bars, store copy | `act.p_hit` is replaced by the realized first touch of close[b] +/- 1 causal sigma (the sigma of the fill bar), from bar b+1 to the forced flat: +1 = up first (long 1s/1s TP before SL), -1 = down first, 0 = neither or both inside one bar. Rows: 10296 / 10328 / 10492 for +1 / 0 / -1. |
| **cheat + noise s** | as above | the leak + N(0, s^2), with s = 0.5, 1, 2 |
| **synthetic** | synthetic H1 bars on the real admitted timestamps (calendar, closures and forced flats unchanged) | GARCH(1,1) hourly volatility with the real hour-of-week profile, 12 sub-steps per bar for HIGH/LOW, and a planted AR(1) factor f (N(0,1), half-life 12 h) stored in `act.p_hit`. E[r_{b+1} \| f_b] = 0.10 x sigma_h x f_b, so the expected drift is 0.52 / 0.89 / 1.34 hourly sigmas per unit f over the next 6 / 12 / 24 h. The 13 price-difference columns are recomputed on the synthetic prices. The other ~107 columns keep their real values: realistic distractors that carry no information about the synthetic path. |
| **synthetic clean** | synthetic, distractors removed | as synthetic, but every column that is not recomputed, not planted and not a clock column (groups `calendar`, `ev`, `px.hours_since_session_open`) is set to 0 |
| **neg (a) within-day shuffle** | cheat store | the labels of every set are permuted within UTC decision days, as specified |
| **neg (a') global shuffle** | cheat store | the labels are permuted across the whole set (one period per call). I added this after (a) failed; see F4. |
| **neg (b) noise inputs** | real bars | every recipe column is replaced by i.i.d. N(0,1), with the NaN pattern kept |

### References (the theoretical side)

* **Oracle replay** (`pipeline_controls.py oracle`). This policy reads the planted or leaked column directly, with no
  network, through the same engine and the same sac replay contract: open sign(x) when |x| >= q, at a fixed (TP, SL)
  cell, with no early close. q and the cell are chosen on INNER_A by Sharpe among valid points; INNER_B is reported.
* **Feature-level ceilings.** The rank AUC of the raw column against each head's own labels: the C2 sets saved by
  the run, and for v1.3 the ZigZag labels.
* **Stage locators.**
  * `probe-heads`: rank correlation of every trained head with the planted column.
  * `learner-check`: the same C2 labels fitted on TRAIN by L2 logistic regression, gradient boosting and Ridge on the
    same scaled tensor.
  * `ablate-strategy`: the run's own heads and Platt maps (rebuilt from the checkpoints, and checked to reproduce the
    run exactly) replayed with every early close disabled.

## Result table: control x contract, expected vs observed (INNER_B = the 2022 validation year, scored after the freeze)

The oracle is the same-engine ceiling for that control. Sharpe is daily, sqrt(365). One seed.

| control | contract | expected | observed INNER_B (Sharpe, trades/wk, PF, net USD) | oracle INNER_B Sharpe | verdict |
|---|---|---|---|---|---|
| cheat | v1.3 | head AUC > 0.9, large profit, both sides, agent closes > 0 | 8.33, 3.87, 4.32, +85k; 42/58 % buy/sell; 68 agent closes; head AUC 0.80 / 0.88 | 14.0 | **PASS on trading; head AUC < 0.9 (F1)** |
| cheat | C2 v1.3 | same | 9.67, 2.77, 11.6, +247k; 46/54 %; 86 agent closes; head AUC 0.84 / 0.84 | 14.0 | **PASS on trading; head AUC < 0.9 (F1)** |
| cheat + N(0, 0.5^2) | v1.3 | graded | 9.53, 3.83, 7.21 | 12.0 | PASS |
| cheat + N(0, 0.5^2) | C2 | graded | 9.02, 3.08, 8.56 | 12.0 | PASS |
| cheat + N(0, 1) | v1.3 | graded | 7.50, 3.60, 4.50 | 10.5 | PASS |
| cheat + N(0, 1) | C2 | graded | 7.40, 2.81, 5.43 | 10.5 | PASS |
| cheat + N(0, 2^2) | v1.3 | graded | 4.07, 3.17, 2.41 | 6.7 | PASS (61 % of the oracle) |
| cheat + N(0, 2^2) | C2 | graded | 2.47, 2.75, 1.78 | 6.7 | weak (37 % of the oracle) |
| synthetic | v1.3 | recover the pattern; profit near the theoretical | 0.48, 2.70, 1.08, +1.0k | 2.58 | **FAIL (F2, F3)** |
| synthetic | C2 v1.3 | same | -0.85, 2.89, 0.81, -1.7k | 2.58 | **FAIL (F3)** |
| synthetic clean (no distractors) | C2 v1.3 | same | 0.49, 3.09, 1.10, +0.8k | 2.58 | **FAIL (F3)** |
| neg (a) within-day shuffle | v1.3 | ~ -costs | **5.33**, 3.34, 2.85, +33k | n/a | **FAIL (F4)** |
| neg (a) within-day shuffle | C2 | ~ -costs | **7.30**, 2.38, 4.60, +128k | n/a | **FAIL (F4)** |
| neg (a') global shuffle | C2 | ~ -costs | -0.01, 2.79, 0.97, -0.2k; head AUC 0.48-0.53 | n/a | PASS |
| neg (a') global shuffle | v1.3 | ~ -costs | -0.29, 3.85, 0.91, -1.0k; head AUC 0.48-0.50 | n/a | PASS (INNER_A selection = 3.39, F5) |
| neg (b) noise inputs | v1.3 | ~ -costs | 0.62, 3.53, 1.09, +1.6k; head AUC 0.49-0.55 | n/a | PASS (INNER_A selection = 4.14, see F5) |
| neg (b) noise inputs | C2 | ~ -costs | -0.49, 3.47, 0.86, -0.9k; head AUC 0.50-0.54 | n/a | PASS |

For scale: the Lo standard error of a one-year daily Sharpe near 0 is about 1.0 (about 1.5 at Sharpe 2.6). The
synthetic shortfall (2.58 -> 0.48 / -0.85) is therefore about 1.5-2.5 SE. The cheat and noise results are far outside
noise.

Graded dose-response (the minimum signal strength the pipeline can exploit). The noisy column's own AUC against the C2
open labels (INNER_B) is 0.83 / 0.80 / 0.73 / 0.64 for s = 0 / 0.5 / 1 / 2. The pipeline keeps 60-75 % of the oracle
Sharpe down to s = 1 under both contracts. At s = 2 (column AUC ~0.64, oracle Sharpe 6.7) it keeps 61 % (v1.3) and
37 % (C2). The synthetic planted factor has a column AUC of 0.63 against the C2 labels too, but an oracle Sharpe of only
2.58, and there the pipeline keeps nothing. **Measured floor:** the pipeline exploits a signal whose same-engine oracle
Sharpe is about 6.7, and fails at about 2.6. Every real-feature arm so far has been tested against an unknown signal
that is very likely below this floor, so their null results do not show that the inputs carry no signal.

## Per-control detail (generated by `pipeline_controls.py table` from the SUMMARY records; one seed)

Trading detail per side comes from `tools/expert_guided_trading_report.py`, whose replay reproduced every run. Net USD compounds because the sizing rule scales with the balance. Exits are counted by the engine's reason.

#### Heads: calibrated AUC INNER_A / INNER_B (Brier INNER_A / INNER_B)

| unit | open_l | open_s | close_l | close_s | nop |
|---|---|---|---|---|---|
| cheat_v13 | 0.81 / 0.80 (0.046 / 0.105) | 0.80 / 0.88 (0.045 / 0.102) | 0.86 / 0.70 (0.110 / 0.235) | 0.84 / 0.74 (0.095 / 0.204) | 0.61 / 0.62 (0.197 / 0.211) |
| cheat_c2 | 0.82 / 0.84 (0.116 / 0.135) | 0.90 / 0.84 (0.094 / 0.124) | 0.68 / 0.65 (0.152 / 0.223) | 0.75 / 0.71 (0.112 / 0.215) | 0.62 / 0.61 (0.176 / 0.179) |
| cheat_noise0.5_v13 | 0.80 / 0.83 (0.043 / 0.104) | 0.77 / 0.86 (0.046 / 0.104) | 0.82 / 0.70 (0.125 / 0.235) | 0.84 / 0.74 (0.108 / 0.198) | 0.62 / 0.61 (0.196 / 0.212) |
| cheat_noise0.5_c2 | 0.80 / 0.75 (0.125 / 0.177) | 0.90 / 0.84 (0.095 / 0.130) | 0.69 / 0.63 (0.149 / 0.227) | 0.81 / 0.67 (0.107 / 0.228) | 0.62 / 0.61 (0.176 / 0.179) |
| cheat_noise1_v13 | 0.81 / 0.85 (0.044 / 0.103) | 0.80 / 0.88 (0.045 / 0.101) | 0.85 / 0.68 (0.113 / 0.243) | 0.82 / 0.68 (0.121 / 0.220) | 0.61 / 0.61 (0.195 / 0.213) |
| cheat_noise1_c2 | 0.78 / 0.75 (0.133 / 0.178) | 0.88 / 0.81 (0.108 / 0.144) | 0.69 / 0.55 (0.146 / 0.242) | 0.77 / 0.61 (0.140 / 0.239) | 0.62 / 0.60 (0.176 / 0.179) |
| cheat_noise2_v13 | 0.76 / 0.81 (0.046 / 0.109) | 0.75 / 0.87 (0.046 / 0.106) | 0.68 / 0.59 (0.177 / 0.255) | 0.62 / 0.53 (0.179 / 0.242) | 0.63 / 0.63 (0.194 / 0.211) |
| cheat_noise2_c2 | 0.73 / 0.68 (0.141 / 0.188) | 0.81 / 0.70 (0.136 / 0.194) | 0.54 / 0.50 (0.206 / 0.240) | 0.74 / 0.58 (0.167 / 0.239) | 0.62 / 0.60 (0.176 / 0.179) |
| syn_v13 | 0.68 / 0.75 (0.046 / 0.115) | 0.63 / 0.72 (0.051 / 0.115) | 0.56 / 0.55 (0.243 / 0.251) | 0.55 / 0.58 (0.216 / 0.204) | 0.59 / 0.56 (0.198 / 0.213) |
| syn_c2 | 0.53 / 0.55 (0.171 / 0.197) | 0.59 / 0.59 (0.187 / 0.193) | 0.50 / 0.43 (0.250 / 0.250) | 0.54 / 0.41 (0.240 / 0.220) | 0.58 / 0.56 (0.179 / 0.177) |
| syn_clean_c2 | 0.61 / 0.65 (0.168 / 0.189) | 0.66 / 0.73 (0.178 / 0.174) | 0.52 / 0.50 (0.250 / 0.250) | 0.50 / 0.61 (0.240 / 0.217) | 0.59 / 0.57 (0.178 / 0.177) |
| shuf_v13 | 0.53 / 0.50 (0.045 / 0.119) | 0.51 / 0.50 (0.045 / 0.119) | 0.80 / 0.66 (0.149 / 0.224) | 0.82 / 0.67 (0.150 / 0.226) | 0.50 / 0.48 (0.199 / 0.220) |
| shuf_c2 | 0.71 / 0.76 (0.144 / 0.175) | 0.84 / 0.79 (0.116 / 0.142) | 0.69 / 0.60 (0.161 / 0.228) | 0.67 / 0.69 (0.131 / 0.226) | 0.52 / 0.49 (0.182 / 0.184) |
| gshuf_v13 | 0.54 / 0.49 (0.048 / 0.119) | 0.52 / 0.49 (0.058 / 0.117) | 0.52 / 0.48 (0.198 / 0.240) | 0.50 / 0.50 (0.229 / 0.225) | 0.50 / 0.49 (0.206 / 0.218) |
| gshuf_c2 | 0.52 / 0.48 (0.161 / 0.208) | 0.51 / 0.50 (0.177 / 0.184) | 0.53 / 0.51 (0.237 / 0.231) | 0.51 / 0.50 (0.247 / 0.236) | 0.52 / 0.48 (0.182 / 0.184) |
| noiseinp_v13 | 0.50 / 0.51 (0.048 / 0.119) | 0.50 / 0.51 (0.057 / 0.117) | 0.51 / 0.53 (0.211 / 0.233) | 0.52 / 0.55 (0.240 / 0.227) | 0.52 / 0.49 (0.205 / 0.219) |
| noiseinp_c2 | 0.51 / 0.50 (0.161 / 0.207) | 0.54 / 0.54 (0.176 / 0.183) | 0.52 / 0.51 (0.220 / 0.231) | 0.51 / 0.53 (0.237 / 0.226) | 0.50 / 0.50 (0.182 / 0.183) |

#### Trading (engine, sac replay contract)

| unit | period | trades/wk | % buy / sell | win % buy / sell | PF | expectancy (sigma) | Sharpe | max DD % | net USD | exits TP/SL/TIME/AGENT/REV |
|---|---|---|---|---|---|---|---|---|---|---|
| cheat_v13 | INNER_A | 4.30 | 34 / 66 | 75 / 75 | 7.68 | 0.379 | 12.22 | 2.3 | 9619 | 60/0/8/41/7 |
| cheat_v13 | INNER_B | 3.87 | 42 / 58 | 76 / 69 | 4.32 | 0.374 | 8.33 | 6.4 | 85054 | 107/0/17/68/13 |
| cheat_c2 | INNER_A | 2.96 | 42 / 57 | 88 / 96 | 44.44 | 0.927 | 12.53 | 2.5 | 21365 | 22/0/8/50/0 |
| cheat_c2 | INNER_B | 2.77 | 46 / 54 | 74 / 85 | 11.56 | 0.760 | 9.67 | 12.3 | 247458 | 34/0/27/86/0 |
| cheat_noise0.5_v13 | INNER_A | 3.70 | 58 / 42 | 76 / 83 | 16.14 | 0.451 | 12.64 | 1.6 | 10125 | 52/0/4/44/0 |
| cheat_noise0.5_v13 | INNER_B | 3.83 | 55 / 45 | 68 / 70 | 7.21 | 0.367 | 9.53 | 6.8 | 83070 | 86/0/12/105/0 |
| cheat_noise0.5_c2 | INNER_A | 2.93 | 52 / 48 | 71 / 92 | 29.41 | 0.772 | 11.15 | 2.5 | 15649 | 21/0/11/37/10 |
| cheat_noise0.5_c2 | INNER_B | 3.08 | 45 / 55 | 76 / 87 | 8.56 | 0.696 | 9.02 | 16.5 | 244648 | 31/2/25/78/27 |
| cheat_noise1_v13 | INNER_A | 3.85 | 46 / 54 | 73 / 75 | 9.33 | 0.401 | 11.86 | 2.8 | 9155 | 57/0/11/31/5 |
| cheat_noise1_v13 | INNER_B | 3.60 | 43 / 57 | 75 / 67 | 4.50 | 0.368 | 7.50 | 12.6 | 69050 | 100/2/18/61/10 |
| cheat_noise1_c2 | INNER_A | 3.07 | 51 / 49 | 67 / 88 | 28.57 | 0.694 | 10.14 | 3.5 | 14468 | 18/0/7/47/11 |
| cheat_noise1_c2 | INNER_B | 2.81 | 50 / 50 | 68 / 85 | 5.43 | 0.561 | 7.40 | 13.8 | 101811 | 27/2/18/90/12 |
| cheat_noise2_v13 | INNER_A | 3.59 | 60 / 40 | 66 / 85 | 5.05 | 0.368 | 8.48 | 2.9 | 7377 | 49/0/8/35/5 |
| cheat_noise2_v13 | INNER_B | 3.17 | 45 / 55 | 63 / 72 | 2.41 | 0.230 | 4.07 | 12.1 | 22359 | 74/6/25/60/3 |
| cheat_noise2_c2 | INNER_A | 2.67 | 38 / 62 | 70 / 69 | 11.12 | 0.577 | 7.38 | 3.9 | 9039 | 14/1/17/40/0 |
| cheat_noise2_c2 | INNER_B | 2.75 | 45 / 55 | 54 / 60 | 1.78 | 0.203 | 2.47 | 18.8 | 12578 | 16/7/23/100/0 |
| syn_v13 | INNER_A | 2.89 | 64 / 36 | 60 / 68 | 2.14 | 0.166 | 3.31 | 7.4 | 4000 | 19/1/9/49/0 |
| syn_v13 | INNER_B | 2.70 | 53 / 47 | 55 / 64 | 1.08 | 0.036 | 0.48 | 19.7 | 974 | 35/13/21/70/4 |
| syn_c2 | INNER_A | 2.78 | 79 / 21 | 58 / 62 | 2.23 | 0.114 | 2.83 | 6.0 | 2319 | 2/0/0/73/0 |
| syn_c2 | INNER_B | 2.89 | 58 / 42 | 36 / 49 | 0.81 | -0.044 | -0.85 | 30.9 | -1731 | 5/5/2/138/3 |
| syn_clean_c2 | INNER_A | 3.15 | 46 / 54 | 49 / 57 | 2.07 | 0.143 | 3.39 | 8.6 | 3527 | 9/0/0/76/0 |
| syn_clean_c2 | INNER_B | 3.09 | 60 / 40 | 37 / 48 | 1.10 | 0.018 | 0.49 | 16.1 | 771 | 10/0/0/154/0 |
| shuf_v13 | INNER_A | 3.78 | 55 / 45 | 64 / 78 | 9.00 | 0.383 | 10.04 | 2.5 | 8480 | 51/0/13/38/0 |
| shuf_v13 | INNER_B | 3.34 | 37 / 63 | 68 / 66 | 2.85 | 0.284 | 5.33 | 10.5 | 32719 | 82/5/17/73/0 |
| shuf_c2 | INNER_A | 2.41 | 52 / 48 | 82 / 97 | 22.46 | 0.993 | 11.26 | 2.2 | 16757 | 22/0/15/28/0 |
| shuf_c2 | INNER_B | 2.38 | 53 / 47 | 67 / 86 | 4.60 | 0.734 | 7.30 | 18.2 | 128425 | 34/3/34/55/0 |
| gshuf_v13 | INNER_A | 3.81 | 13 / 87 | 85 / 52 | 1.73 | 0.123 | 3.39 | 4.9 | 1963 | 21/1/13/64/4 |
| gshuf_v13 | INNER_B | 3.85 | 17 / 83 | 50 / 42 | 0.91 | -0.011 | -0.29 | 30.4 | -1010 | 31/2/20/137/14 |
| gshuf_c2 | INNER_A | 2.33 | 21 / 79 | 46 / 46 | 1.29 | 0.019 | 0.85 | 4.1 | 233 | 0/0/0/62/1 |
| gshuf_c2 | INNER_B | 2.79 | 23 / 77 | 59 / 37 | 0.97 | -0.013 | -0.01 | 25.5 | -194 | 3/1/4/139/1 |
| noiseinp_v13 | INNER_A | 2.74 | 19 / 81 | 93 / 75 | 2.87 | 0.255 | 4.14 | 4.6 | 3666 | 49/1/14/10/0 |
| noiseinp_v13 | INNER_B | 3.53 | 29 / 71 | 65 / 65 | 1.09 | 0.035 | 0.62 | 31.9 | 1600 | 107/15/32/33/0 |
| noiseinp_c2 | INNER_A | 3.48 | 18 / 82 | 47 / 39 | 1.13 | 0.003 | 0.54 | 5.8 | 180 | 1/0/3/88/2 |
| noiseinp_c2 | INNER_B | 3.47 | 22 / 78 | 41 / 38 | 0.86 | -0.010 | -0.49 | 17.2 | -864 | 2/2/3/170/7 |

#### Frozen strategy genes (chosen on INNER_A)

| unit | genes | strategy |
|---|---|---|
| cheat_v13 | `{"open_rate_l": 0.134672, "open_rate_s": 0.168685, "gain_quantile_l": 0.897345, "gain_quantile_s": 0.285337, "close_rate": 0.322329, "allow_reverse": 0.935148, "nop_veto_rate": 0.009373}` | `{"open_threshold_l": 0.123, "open_threshold_s": 0.114, "gain_margin_l": -1.76, "gain_margin_s": -2.114, "close_threshold": 0.501, "allow_reverse": true, "nop_veto": 0.872}` |
| cheat_c2 | `{"open_rate_l": 0.170785, "open_rate_s": 0.265542, "close_rem_rate": 0.424697, "close_p_rate": 0.100523, "allow_reverse": 0.243551, "nop_veto_rate": 0.260387}` | `{"open_margin_l": 0.827, "open_margin_s": 0.796, "close_margin": 0.151, "close_p": 0.472, "allow_reverse": false, "nop_veto": 0.82}` |
| cheat_noise0.5_v13 | `{"open_rate_l": 0.088547, "open_rate_s": 0.052065, "gain_quantile_l": 0.380955, "gain_quantile_s": 0.904376, "close_rate": 0.549695, "allow_reverse": 0.303514, "nop_veto_rate": 0.3195}` | `{"open_threshold_l": 0.134, "open_threshold_s": 0.153, "gain_margin_l": -2.106, "gain_margin_s": -1.784, "close_threshold": 0.307, "allow_reverse": false, "nop_veto": 0.765}` |
| cheat_noise0.5_c2 | `{"open_rate_l": 0.259582, "open_rate_s": 0.272998, "close_rem_rate": 0.354981, "close_p_rate": 0.131709, "allow_reverse": 0.548521, "nop_veto_rate": 0.252029}` | `{"open_margin_l": 0.668, "open_margin_s": 0.678, "close_margin": 0.188, "close_p": 0.489, "allow_reverse": true, "nop_veto": 0.822}` |
| cheat_noise1_v13 | `{"open_rate_l": 0.071537, "open_rate_s": 0.105684, "gain_quantile_l": 0.159896, "gain_quantile_s": 0.666003, "close_rate": 0.259928, "allow_reverse": 0.544642, "nop_veto_rate": 0.11552}` | `{"open_threshold_l": 0.156, "open_threshold_s": 0.146, "gain_margin_l": -2.141, "gain_margin_s": -1.968, "close_threshold": 0.509, "allow_reverse": true, "nop_veto": 0.813}` |
| cheat_noise1_c2 | `{"open_rate_l": 0.182074, "open_rate_s": 0.267492, "close_rem_rate": 0.374451, "close_p_rate": 0.059591, "allow_reverse": 0.53158, "nop_veto_rate": 0.288481}` | `{"open_margin_l": 0.86, "open_margin_s": 0.645, "close_margin": 0.207, "close_p": 0.528, "allow_reverse": true, "nop_veto": 0.816}` |
| cheat_noise2_v13 | `{"open_rate_l": 0.081513, "open_rate_s": 0.079392, "gain_quantile_l": 0.226548, "gain_quantile_s": 0.646767, "close_rate": 0.135271, "allow_reverse": 0.92176}` | `{"open_threshold_l": 0.123, "open_threshold_s": 0.131, "gain_margin_l": -2.119, "gain_margin_s": -1.986, "close_threshold": 0.383, "allow_reverse": true, "nop_veto": null}` |
| cheat_noise2_c2 | `{"open_rate_l": 0.11908, "open_rate_s": 0.271989, "close_rem_rate": 0.144223, "close_p_rate": 0.110564, "allow_reverse": 0.264109, "nop_veto_rate": 0.260275}` | `{"open_margin_l": 0.893, "open_margin_s": 0.498, "close_margin": 0.165, "close_p": 0.511, "allow_reverse": false, "nop_veto": 0.82}` |
| syn_v13 | `{"open_rate_l": 0.063786, "open_rate_s": 0.032087, "gain_quantile_l": 0.268397, "gain_quantile_s": 0.757997, "close_rate": 0.321741, "allow_reverse": 0.626644, "nop_veto_rate": 0.139789}` | `{"open_threshold_l": 0.097, "open_threshold_s": 0.103, "gain_margin_l": -2.059, "gain_margin_s": -1.934, "close_threshold": 0.423, "allow_reverse": true, "nop_veto": 0.783}` |
| syn_c2 | `{"open_rate_l": 0.244154, "open_rate_s": 0.056435, "close_rem_rate": 0.283636, "close_p_rate": 0.045084, "allow_reverse": 0.616952, "nop_veto_rate": 0.460901}` | `{"open_margin_l": 0.28, "open_margin_s": 1.14, "close_margin": -0.233, "close_p": 0.484, "allow_reverse": true, "nop_veto": 0.776}` |
| syn_clean_c2 | `{"open_rate_l": 0.24309, "open_rate_s": 0.286823, "close_rem_rate": 0.578373, "close_p_rate": 0.069801, "allow_reverse": 0.011304}` | `{"open_margin_l": 0.386, "open_margin_s": 0.323, "close_margin": -0.125, "close_p": 0.501, "allow_reverse": false, "nop_veto": null}` |
| shuf_v13 | `{"open_rate_l": 0.131619, "open_rate_s": 0.277804, "gain_quantile_l": 0.249306, "gain_quantile_s": 0.854479, "close_rate": 0.387239, "allow_reverse": 0.34776, "nop_veto_rate": 0.220824}` | `{"open_threshold_l": 0.052, "open_threshold_s": 0.049, "gain_margin_l": -2.033, "gain_margin_s": -2.008, "close_threshold": 0.47, "allow_reverse": false, "nop_veto": 0.726}` |
| shuf_c2 | `{"open_rate_l": 0.235943, "open_rate_s": 0.230497, "close_rem_rate": 0.212275, "close_p_rate": 0.091506, "allow_reverse": 0.009062, "nop_veto_rate": 0.408947}` | `{"open_margin_l": 0.538, "open_margin_s": 0.779, "close_margin": 0.349, "close_p": 0.4, "allow_reverse": false, "nop_veto": 0.767}` |
| gshuf_v13 | `{"open_rate_l": 0.042577, "open_rate_s": 0.258775, "gain_quantile_l": 0.277139, "gain_quantile_s": 0.636272, "close_rate": 0.121914, "allow_reverse": 0.943898, "nop_veto_rate": 0.0278}` | `{"open_threshold_l": 0.061, "open_threshold_s": 0.065, "gain_margin_l": -2.017, "gain_margin_s": -1.972, "close_threshold": 0.357, "allow_reverse": true, "nop_veto": 0.711}` |
| gshuf_c2 | `{"open_rate_l": 0.010251, "open_rate_s": 0.043059, "close_rem_rate": 0.435486, "close_p_rate": 0.175298, "allow_reverse": 0.642795, "nop_veto_rate": 0.114792}` | `{"open_margin_l": 0.62, "open_margin_s": 0.655, "close_margin": -0.164, "close_p": 0.447, "allow_reverse": true, "nop_veto": 0.779}` |
| noiseinp_v13 | `{"open_rate_l": 0.171408, "open_rate_s": 0.263749, "gain_quantile_l": 0.690805, "gain_quantile_s": 0.582138, "close_rate": 0.009291, "allow_reverse": 0.279809}` | `{"open_threshold_l": 0.052, "open_threshold_s": 0.062, "gain_margin_l": -2.008, "gain_margin_s": -1.987, "close_threshold": 0.42, "allow_reverse": false, "nop_veto": null}` |
| noiseinp_c2 | `{"open_rate_l": 0.031467, "open_rate_s": 0.161194, "close_rem_rate": 0.286991, "close_p_rate": 0.111988, "allow_reverse": 0.702552, "nop_veto_rate": 0.271465}` | `{"open_margin_l": 0.835, "open_margin_s": 0.672, "close_margin": -0.149, "close_p": 0.386, "allow_reverse": true, "nop_veto": 0.762}` |


## Failures, each located at the stage that loses the signal

### F1. v1.3 head targets are not the tradable outcome (stage: labels -> heads); explains "AUC < 0.9" on the cheat

* The cheat is a perfect predictor of the 1 sigma / 1 sigma first touch (oracle INNER_B: 100 % wins, Sharpe 14.0).
  Against v1.3's round-0 labels, however, its own AUC is only **0.60 (long) / 0.57 (short)** on INNER_A and
  0.60 / 0.56 on INNER_B. Those labels are "the expert opens exactly AT a hindsight ZigZag pivot bar" (`oracle.pivot_labels`
  puts the class only on the pivot bar; 23 % of the rows). Against the DAgger labels (the LegOracle open rule) the
  cheat reaches 0.85 / 0.81.
* The heads still reach 0.80-0.88 because their calibration sets mix the round-0 and DAgger rows. On the synthetic
  market the planted factor's AUC against the round-0 pivot labels is **0.495 / 0.492**: the v1.3 imitation target
  carries none of a planted 6-24 h drift. Against the DAgger rule it is only 0.56 / 0.55.
* The v1.3 SL head is constant: its target is SL = 2.0 for every open row (`sl_target_std` = 0 in
  `tp_sl_calibration_block`), so it learns nothing. The TP head targets the remaining ZigZag leg (mean 0.58 sigma).
* For C2, the "AUC < 0.9" is not a loss either. Its target is TP before SL at (1.5, 1.5), where TIME counts as not TP,
  and the 1 sigma / 1 sigma leak predicts that target at AUC 0.78 / 0.86 (INNER_A) and 0.83 / 0.83 (INNER_B). The C2
  heads reach **0.82 / 0.90** and **0.84 / 0.84**, at or above the column's own ceiling, and L2 logistic regression on
  the same tensor gets 0.83-0.86. With a strong single-column signal on real bars, the heads lose nothing.

### F2. v1.3 cannot learn a planted drift: its teacher does not encode it (stage: labels)

Synthetic v1.3: the opener logits correlate with the planted factor at only Spearman -0.04 / +0.04 (INNER_B).
`probe-heads` shows that only `close_l` picks it up (-0.21). INNER_A selection Sharpe is 3.31 (inside the noise floor,
F5) and INNER_B is 0.48. The labels (ZigZag pivot timing) give no gradient toward f, so the pipeline cannot be blamed
downstream. **Any input whose value is a slow drift / trend (hours to a day) is invisible to the v1.3 imitation target.**

### F3. C2 loses a moderate signal in the value heads, and the strategy layer then closes almost every trade (stages: heads -> strategy)

Evidence on synthetic (with distractors) and synthetic clean (no distractors), same labels and same rows:

| measure (INNER_B) | planted column alone | L2 logistic / Ridge on the same tensor | E3-M network, synthetic | E3-M network, synthetic clean |
|---|---|---|---|---|
| AUC, open_l TP-first label | 0.629 | 0.727 (logreg, last slot) | 0.554 | 0.650 |
| AUC, open_s TP-first label | 0.638 | 0.740 | 0.594 | 0.733 |
| Spearman, predicted vs realized net (mean over 12 cells), open_l | 0.198 | 0.189 (Ridge) | 0.075 | 0.056 |
| Spearman, same, open_s | 0.186 | 0.157 | 0.051 | 0.103 |
| head output vs planted f (Spearman), open_l / open_s | n/a | n/a | 0.044 / -0.081 | 0.388 / -0.240 |

* **Heads, binary head.** With 100+ uninformative distractors, the binary heads keep only a small part of the signal
  (0.55-0.59 against 0.73-0.74 for a linear model on the same tensor). TRAIN-to-INNER_B decay (open_l vs f: 0.178 TRAIN,
  0.148 INNER_A, 0.044 INNER_B) shows that the network fits the distractors. Without the distractors the binary heads
  recover (0.65 / 0.73, about the logistic level).
* **Heads, value head.** Even without the distractors, the **value heads**, which are what the C2 open rule actually
  reads, stay at Spearman 0.06-0.10 against a ceiling of 0.16-0.20. The open rule therefore ranks entries on a head
  that has lost about two thirds of the signal, while the better-informed binary head is only reported.
* **Strategy.** The frozen strategies close nearly every trade early: AGENT_CLOSE was 138 of 153 trades (synthetic
  C2) and 154 of 164 (synthetic clean). TP exits were only 5 and 10, against an oracle that holds to TP / SL / forced
  flat. The C2 closers are anti-predictive out of sample: close AUC on INNER_B is 0.43 / 0.41 on synthetic, against
  0.50 / 0.54 on INNER_A. The genes were picked on six months of INNER_A, where heavy closing happened to help
  (close_rem_rate 0.28-0.58). Disabling early close does not rescue the run (INNER_B 0.37 / 0.81, but only 1.45 / 1.53
  trades a week, which is invalid), because the entries carry no edge (value-head loss above). The cheat shows the
  mirror image: early close is worth +3 to +4 Sharpe there (C2 9.67 -> 6.81 and v1.3 8.33 -> 4.13 without it), because
  the closers can read the leak.
* **Verdict.** The C2 pipeline fails a moderate, realistic signal at the heads (distractor overfitting in the binary
  heads; information loss in the value heads that drive the open rule). The INNER_A-selected exit genes then add
  further damage.

### F4. The within-day label shuffle is not a valid negative control for these labels (stage: control design, not the pipeline)

The cheat with labels permuted within days still made INNER_B Sharpe 5.33 (v1.3) and 7.30 (C2), with head AUCs of
0.66-0.85. The reason is in the labels, not in the code: every label here has a horizon of hours to a day, so the
labels of the same day are nearly identical. The leak's AUC against the **within-day-shuffled** C2 labels is still
0.68-0.76 (INNER_A) and 0.69-0.74 (INNER_B). For the closers it is unchanged (0.70 / 0.71): a position's
"SL first from here" label is constant across its own rows. In v1.3 the shuffle did kill the openers (AUC 0.50 / 0.50,
because pivot labels are not autocorrelated), but the closers kept AUC 0.66-0.67 and still produced profit from random
entries plus perfect-information exits. The **global** shuffle (labels permuted across the whole period) passes for C2
(heads 0.48-0.53, INNER_B Sharpe -0.01, net -194 USD): no leakage path from inputs to profit exists outside the labels.
v1.3 under the global shuffle also passes (heads 0.48-0.50 on INNER_B, INNER_B Sharpe -0.29, net -1010 USD).

### F5. The INNER_A strategy search produces large in-sample Sharpes from pure noise (stage: strategy selection)

With i.i.d. noise inputs (heads at AUC 0.49-0.55) the frozen v1.3 strategy shows **INNER_A Sharpe 4.14** (PF 2.87,
78 % wins), while TRAIN is -2.86 and INNER_B 0.62. The globally shuffled v1.3 control shows INNER_A 3.39 (INNER_B -0.29).
C2 noise reaches INNER_A 0.54, and the C2 global shuffle 0.85. Synthetic v1.3 (3.31) and synthetic
C2 (2.83) sit inside this noise band. INNER_A Sharpes and visitation-round "chosen" Sharpes are therefore not evidence
of anything below roughly 4; only post-freeze blocks are.

### F6. Win rate is not evidence under v1.3's level geometry (stage: reporting)

The noise-input v1.3 control won 65 % of its trades on INNER_B with PF 1.09 and expectancy 0.035 sigma. The v1.3 TP head
sits around 0.6-1.25 sigma while SL is fixed at 2 sigma, so the break-even win rate is about 70-75 %. Reports must give
expectancy in sigma (or PF) next to win rate.

### Checks that passed

* **Engine correctness.** `tools/expert_guided_trading_report.py` replayed every unit, reproduced each run exactly
  (`reproduce: true`) and found **0 failed correctness checks** (SL/TP on every order, sizing, daily cap, fills,
  forced flat, cost accounting).
* **No leakage outside the labels.** Noise inputs and the global shuffle do not profit out of sample.
* **Data/window stage.** With a strong leak on real bars, the heads match the column's own ceiling and logistic
  regression (F1), so the window, scaler and recipe binding deliver the column intact.
* **Both sides traded and agent closes > 0** in every positive control. Side shares were 42-55 % buy.

## Recommended fixes

1. **Make a positive control a gate.** No future "no signal" verdict without, in the same run configuration, (i) the
   cheat control passing and (ii) a synthetic planted-signal control of the size being claimed absent. Wire
   `pipeline_controls.py` into the chain the way the sanity gate is.
2. **v1.3 / imitation labels.** Stop using pivot-bar labels (`pivot_labels`) as the open target for any claim about
   input information. Train the openers on the action outcome (C2's entry table) or at least on the LegOracle direction
   with remaining move >= 1 sigma. Either fit the SL head on a varying target or drop it (it is constant now).
3. **C2 value heads.** The open rule should read an expected value built from the calibrated binary head (p x TP -
   (1 - p) x SL - cost at the reference cell) or a separately validated value model. Before trusting the value head,
   check its INNER_A Spearman against realized net next to the binary AUC; today it loses about 2/3 of the signal.
   Consider a lower weight decay or a separate early-stopping score for the regression heads.
4. **Distractor dimensionality.** With 121 columns, a planted factor of realistic size is drowned (binary AUC 0.55 vs
   0.73 for logistic on the same tensor). Use group-level selection or regularization inside the model (the owner's
   "in-model selection"), or a learner-check baseline (logistic / GBM on the same tensor) reported beside every head as
   the floor the network must reach.
5. **Exit genes.** Require the early-close genes to beat "no early close" on a held-out part of the calibration block,
   or constrain `close_rem_rate` / `close_p_rate` to a smaller range. With a 12-24 h drift, the frozen strategies closed
   90 %+ of trades before any TP.
6. **Negative controls.** Replace "shuffle within days" with a global (or week-block) permutation for labels whose
   horizon is about a day. Keep the noise-inputs control.
7. **Selection noise floor.** Report the INNER_A Sharpe of the noise-inputs control (4.1 here) as the floor every
   INNER_A or visitation Sharpe must exceed. Decide only on post-freeze blocks, as the contracts already state.
8. **Seeds.** Repeat the failing controls (synthetic, noise 2) with the five-seed ensemble before acting on effect
   sizes. These are single-seed runs.

## Reproduce

```
python tools/pipeline_controls.py build cheat        --recipe <fold recipe wf2022> --store <full_pool store> --admitted <admitted> --out <dir>/DIAGNOSTIC_cheat
python tools/pipeline_controls.py build cheat --noise 1 ...                                                         --out <dir>/DIAGNOSTIC_cheat_noise1
python tools/pipeline_controls.py build noise_inputs ...                                                            --out <dir>/DIAGNOSTIC_noise_inputs
python tools/pipeline_controls.py build synthetic    ...                                                            --out <dir>/DIAGNOSTIC_synthetic
python tools/pipeline_controls.py build synthetic_clean --recipe <synthetic recipe> --store <synthetic store> ...   --out <dir>/DIAGNOSTIC_synthetic_clean
python tools/pipeline_controls.py run-e3m [--pin-admitted <synthetic sha>] [--shuffle-labels-within-day 7 | --shuffle-labels-global 7] -- \
    run <the WF_V13 / WF_C2_13 unit arguments, fold wf2022, --seeds 20261019, out dir containing DIAGNOSTIC>
python tools/pipeline_controls.py oracle | probe-heads | learner-check | ablate-strategy | summarize | table ...
```

Every heavy step runs under `crispdm-run -q -m 3G` with `CUDA_VISIBLE_DEVICES=""`. The compact evidence (summaries,
oracle replays, head probes, learner checks, strategy ablations, build notes) is in
`evidence/pipeline_controls_DIAGNOSTIC/`. The full run directories (checkpoints, ledgers, per-trade CSVs) stay outside
git in the controls work directory.
