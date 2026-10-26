# Trading detail — compact summary (2026-10-26)

Seed ensemble (primary fitness), decision year INNER_B (frozen, walk-forward validation year). Replays verified to reproduce every run exactly (V13 4/4, C2/C2G 4/4 units). **All 15 correctness checks PASS on every unit, policy and period** (SL+TP on every order and on the correct side, sizing-rule units, rel_volume 0.10–0.20, leverage ≤ 1:50, integer units, ≤ 1 entry/day, no overlap, next-open fills, exit price per reason, operational window, closed before the forced flat, positive costs, engine protection re-derived bar by bar, net = gross − costs). Full detail: `TRADING_DETAIL_V13.md`, `TRADING_DETAIL_C2_C2G_13.md`; per-trade CSVs in `trades/`.

| unit | policy | period | orders | orders/wk | % buy | % buy winning | % sell winning | win rate | profit factor | expectancy σ | net USD | Sharpe | max DD % | exits % TP / SL / TIME / AGENT / REV | early-close value USD [95 % CI] | checks PASS / FAIL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| WF_V13_wf2022_both_hold_v13 | ensemble/sharpe_lcb | INNER_B | 189 | 3.57 | 46.0 % | 41.4 % | 41.2 % | 41.3 % | 0.90 | -0.017 | -1155 | -0.47 | 32.1 | 15 / 15 / 3 / 67 / 0 | -4067 [-8965, 829] | 15 / 0 |
| WF_V13_wf2022_zz_hold_v13 | ensemble/sharpe_lcb | INNER_B | 135 | 2.55 | 28.9 % | 61.5 % | 61.5 % | 61.5 % | 0.73 | -0.123 | -4051 | -1.11 | 45.5 | 47 / 17 / 31 / 1 / 3 | -401 [-1080, 780] | 15 / 0 |
| WF_V13_wf2023_both_hold_v13 | ensemble/sharpe_lcb | INNER_B | 147 | 2.83 | 38.1 % | 37.5 % | 47.3 % | 43.5 % | 0.81 | -0.081 | -2478 | -1.08 | 37.8 | 38 / 51 / 11 / 0 / 0 | 0 [—, —] | 15 / 0 |
| WF_V13_wf2023_zz_hold_v13 | ensemble/sharpe_lcb | INNER_B | 165 | 3.17 | 86.7 % | 30.1 % | 4.5 % | 26.7 % | 0.49 | -0.137 | -4008 | -3.12 | 40.9 | 16 / 1 / 4 / 79 / 0 | 184 [-3951, 4313] | 15 / 0 |
| WF_C2G_13_wf2022_zz_feat_c2g_13 | ensemble/sharpe_lcb | INNER_B | 161 | 3.04 | 54.7 % | 27.3 % | 30.1 % | 28.6 % | 0.73 | -0.076 | -3172 | -1.10 | 39.3 | 9 / 0 / 7 / 80 / 4 | -3617 [-11733, 5097] | 15 / 0 |
| WF_C2G_13_wf2023_zz_feat_c2g_13 | ensemble/sharpe_lcb | INNER_B | 132 | 2.54 | 30.3 % | 42.5 % | 42.4 % | 42.4 % | 0.53 | -0.163 | -4226 | -2.17 | 46.0 | 4 / 2 / 3 / 91 / 1 | -631 [-4162, 3182] | 15 / 0 |
| WF_C2_13_wf2022_zz_feat_c2_13 | ensemble/sharpe_lcb | INNER_B | 133 | 2.51 | 54.1 % | 40.3 % | 49.2 % | 44.4 % | 0.89 | -0.014 | -1110 | -0.25 | 26.9 | 8 / 3 / 5 / 84 / 1 | 1751 [-6607, 9731] | 15 / 0 |
| WF_C2_13_wf2023_zz_feat_c2_13 | ensemble/sharpe_lcb | INNER_B | 156 | 3.00 | 41.7 % | 35.4 % | 28.6 % | 31.4 % | 0.52 | -0.095 | -2945 | -1.98 | 29.7 | 4 / 1 / 1 / 88 / 5 | 919 [-5021, 7298] | 15 / 0 |

## Findings (strategy, not engine)

- **Inverted reward/risk in the ZigZag-label arms**: `zz_hold_v13` median TP 0.72–0.81 σ against SL 2.0 σ (reward/risk 0.36–0.40). The SL head learns the constant 2 σ target. Win rate 61.5 % (2022) still gives profit factor 0.73 (avg win 136 vs avg loss −281 USD on buys).
- **C2 / C2G**: reward/risk 0.6–0.8 (TP 1.5–2.0 σ, SL 2.37–2.44 σ); agent closes are 80–91 % of exits; win rates 29–44 %, profit factor 0.52–0.89, all four negative.
- **Costs exceed the gross edge**: e.g. `both_hold_v13` 2022 gross +191 USD, trading costs 1 351 USD, net −1 155 USD; in every unit gross P&L is ≤ ~0 before costs, or costs flip it negative.
- **The 1 entry/day cap binds**: 42–1 053 entries refused by DAILY_CAP per unit-year (ensemble).
- **Direction skew**: `zz_hold_v13` 2023 is 86.7 % buys with sells winning 4.5 %; individual seeds reach 84–90 % buys.
- **Activity**: 2.5–3.6 orders/wk; 4–14 of 52–53 weeks have fewer than 2 orders (period average rule met).
- **Early close does not pay out of sample** in 5 of 8 ensembles; every 95 % interval includes 0.
