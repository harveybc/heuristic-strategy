# Causal event-window strategy — contract, estimands, results and verdict

**DEVELOPMENT · inputs FIXTURE_RESEARCH_ONLY.** Lead design 2026-10-19 (owner gave creative freedom).
Oracle → noise study plus the real surprise-sign rule, on the two feature-eng test fixtures, with the
existing simulator, business rules and figure style. Decision period 2011–2017, confirmation
2018–2020 (the hourly fixture ends 2020-04-29). TEST (2025+) is irrelevant here; both readers keep
the TEST guard.

Tables: `results/causal_event/TABLAS_EVENTO.md`. Figures:
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/causal_event/evento_oraculo_decision.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/causal_event/evento_oraculo_confirmacion.png

## 1. Data and its limits (nothing acquired, nothing copied)

| Object | Status |
|---|---|
| Economic calendar fixture (Investing.com export, 2011-01 → 2021-04, 121 658 rows) | **RESEARCH-ONLY**: its ToS prohibits use, storage and distribution. Never published, never fed to anything live. The repo holds hashes and aggregates only; per-event rows stay in the run directory outside the repo. |
| Its clock | **Not a legal DST rule.** Data-lane measurement on 1 571 anchor rows: fixed UTC−5 from 2011-01 to 2018-01, then exporter windows UTC−4 / UTC−5 switching 2018-02-28, 2018-09-12, 2019-02-01, 2019-11-01, 2020-02-03, 2020-09-01, 2021-02-03. Converted with that per-period table (`EXPORTER_OFFSET_SWITCHES`). Check: ≥ 95 % of NFP rows land at 08:30 New York after conversion; NFP 2011-01-07 reads actual 103 / forecast 150. |
| `actual_text` | Encodes good/bad, agrees with sign(actual − forecast) only 90 % → only the numeric surprise is used. |
| `previous` | Embeds `<Revised From x>` with its first bytes eaten by quoting; not used. **Retrospective fixture**: consensus and actual may carry revisions, a stated contamination risk — schedule assumed known in advance, consensus assumed frozen pre-release. |
| Hourly EUR/USD fixture (2005-05 → 2020-04-29, 93 084 bars) | Clock unverified. Detected from the NFP jump: the bar with the largest mean \|close − open\| on 47 winter NFP days is label hour 13 (vol ratio 3,9× other hours) → offset 0 h, **CLOCK_DETECTED_NOT_VERIFIED**. |
| Universe | USD and EUR (incl. German) **High** impact with numeric actual and forecast: ~265–320 USD and ~100–115 EUR rows per year; 3 575 rows with at least two contiguous post-release bars. Frequency ≥ 3/week is met with High only, so the High+Moderate fallback was not needed. |

## 2. Contract (every order carries units, SL and TP from the entry)

- **Universe:** one order per release instant (the largest \|surprise\| wins when several families share it). Flat before the release.
- **Entry:** market at the next hourly bar open after the release (NEXT_BAR). LATENCY_0 (entry at the release-bar open) is an **oracle bound**, not achievable.
- **Direction:** sign(standardized surprise) × sign_map[family]. Surprise = (actual − forecast)/scale, scale = 1,4826·MAD of actual − forecast over **prior** releases of that family only (expanding, ≥ 6). sign_map = sign of the slope of the post-release drift on the surprise, fitted **on 2011–2017 only**; a family is traded only when n ≥ 20 and p < 0,05 (the causal take/skip gate). Abstain when \|surprise\| is below a quantile q ∈ {0; 0,25; 0,5} chosen on 2011–2017.
- **TP:** \|slope_family × surprise\| − 1 pip inset (the fitted drift for that family and magnitude), ≥ broker minimum and within 1,5 × the causal event sigma (the order rules of `STRATEGY_CONTRACT.md` §14).
- **SL:** entry ∓ m·σ̂_event, σ̂_event = expanding mean of the family's realized volatility over its window (prior events only); m fixed a priori at 2 with a {1, 2, 3} sensitivity on 2011–2017.
- **Time stop:** k × the family's half-life of the volatility impulse (first post-release hour whose \|return\| ≤ ½ the release-bar \|return\|, averaged over prior events), k ∈ {1, 2, 3}; never beyond the forced flat (1 h before each session close).
- **Sizing:** units = floor(percentual_order_size × 0,10 × balance × leverage / price), percentual_order_size = clip(σ_ref/σ̂_event, 0,25, 1) × clip(0,5 + 0,25\|surprise\|, 0,5, 1); leverage 1:100 default, 1:10 / 1:50 in the appendix.
- **Costs:** realistic profile (hour-dependent spread, slippage, commission, swap) **plus** an event-widened spread on the release bar and the next one: 5 pips default (ASSUMED), with 0 (normal schedule) and 10 pips as a sensitivity axis.
- **Second arm (STRADDLE):** OCO stop legs at reference ± ½ the family's expected release-bar jump, placed on the release bar, TP = the family's mean \|drift\|, SL = m·σ̂_event.
- **Frequency rule:** ≥ 3 trades/week and ≤ 4 orders/day; a configuration failing it is not reported as a result (invalid, hollow in figures).
- **Diagnostic only:** NEXT_BAR_UNGATED switches the causal gate off (every family traded with the sign of its 2011–2017 slope, significant or not). It is not a result.

## 3. Oracle → noise (5 seeds, means; 1:100, 5-pip release spread)

Ideal = per-event knowledge of the realized post-release drift (sign and magnitude), realized volatility and realized dissipation. Each axis degrades one input and leaves the others ideal.

| Axis | Decision 2011–2017 | Confirmation 2018–2020 |
|---|---|---|
| Ideal (all oracle), NEXT_BAR | Sharpe 7,59 · 4,1 trades/wk · 94 % TP exits | Sharpe 7,24 · 4,4/wk · 95 % TP |
| Ideal at LATENCY_0 (release-bar open, oracle bound) | Sharpe 9,29 | Sharpe 9,44 — the hourly latency forfeits the jump but not the drift |
| Drift MAE (naive = drift 0: 26,9 / 19,9 pips) | Sharpe 2,20 at 30 pips, 0,86 at 40; no break-even inside the grid | **Break-even ≈ 31 pips** (above the naive MAE: a noisy magnitude still carries a 66 % sign) |
| Direction accuracy of the sign rule | **Break-even 0,66** (0,60 → −0,72; 0,70 → +1,05) | **Break-even 0,72** (0,70 → +0,14) |
| Volatility MAE × MEAN20 | Sharpe 5,28 at 0,86×, 4,49 at 1,20×; frequency holds | Sharpe 4,44 at 0,80×, 3,77 at 1,11× |
| Half-life MAE (bars) | 7,59 → 7,20 over 0,7 → 2,6 bars | 7,24 → 6,61 |

Reading: with the sign and magnitude known, the event window is very profitable even after an hour of latency and a 5-pip spread; the strategy is **governed by direction accuracy** (needs ≥ 0,66–0,72) and the drift magnitude, and is insensitive to the vol and half-life forecasts. Net USD on the decision panel compounds the 10 % sizing over seven years and is shown on a symmetric log scale; Sharpe is the comparable figure.

## 4. The real rule

| Arm | Period | Trades | Sharpe | Net USD | Exit mix TP/SL/time | Valid |
|---|---|---:|---:|---:|---|---|
| NEXT_BAR (gated) | 2011–2017 and 2018–2020 | **0** | — | 0 | — | no |
| LATENCY_0 (gated) | both | 0 | — | 0 | — | no |
| STRADDLE (chosen k=2, m=2, q=0,5) | 2011–2017 | 1 173 | −3,88 | −10 000 (account exhausted) | 0,38 / 0,12 / 0,51 | sí |
| STRADDLE | 2018–2020 | 438 | −4,19 | −9 795 | 0,30 / 0,10 / 0,60 | sí |
| STRADDLE, normal spread | 2018–2020 | 438 | −3,07 | −9 401 | | sí |
| NEXT_BAR_UNGATED (diagnostic) | 2011–2017 | 133 (0,36/wk) | −1,57 | −3 430 | 0,91 / 0,00 / 0,09 | **no** |
| NEXT_BAR_UNGATED (diagnostic) | 2018–2020 | 37 (0,30/wk) | −1,11 | −1 824 | 0,86 / 0,05 / 0,08 | **no** |

Leverage changes only the net scale (appendix in the tables). The 10-pip spread turns the straddle from −4,2 to −5,3.

**Why the gated rule never trades.** On 2011–2017 **no family of the 26 shows a significant effect of the surprise on the post-release drift** (smallest p-values: Core Durable Goods 0,051, PPI 0,054, CPI 0,16, Retail Sales 0,16; NFP 0,55, ECB decision 0,63). The fitted slopes are 1–8 pips per unit of standardized surprise, so even an ungated TP sits at 0–7 pips: **below the round-trip cost with a 5-pip release spread**, which is why the ungated diagnostic exits 86–91 % at TP and still loses. This matches the NFP POC (vol ratio of the release bar 1,36 here vs 1,07 there with its own normalisation; drift n.s. in both). Direction accuracy of the ungated sign rule: 0,55 on 2011–2017, 0,49 on 2018–2020 — far below the 0,66–0,72 break-even.

**Straddle.** The jump is not capturable at hourly resolution: the stop legs fill through the widened spread and more than half the trades time out; every family loses on 2011–2017 except Philadelphia Fed (+33 USD).

## 5. Verdict

1. The oracle shows a large prize (Sharpe ≈ 7 after latency and widened spread) that depends almost entirely on **direction accuracy ≥ 0,66–0,72** and a drift magnitude with MAE ≲ 30 pips; vol and half-life forecasts are secondary.
2. The causal surprise-sign rule **does not reach that prize**: at hourly resolution and with the Investing.com consensus, the surprise has no significant directional effect on the 1–12 h drift in any USD/EUR high-impact family, the estimated drifts are smaller than transaction costs, and the achieved sign accuracy is ~0,5.
3. The straddle loses under every spread assumption. Neither arm is a candidate for the integrated pipeline; the event layer, if kept, is a **risk filter** (widen stops / abstain around releases), consistent with the calendar diagnostic in `CALENDAR_RANGE_EXPERIMENT.md`.
4. What would change the verdict: minute data around the release (the LATENCY_0 bound is 2 Sharpe higher), a governed point-in-time consensus, and a per-family vintage-safe calendar for 2020–2024.

## Limits
- Fixture-only evidence: retrospective calendar (consensus/actual revisions possible), unverified bar clock, one market path.
- The 2011–2017 numbers of the real rule are in-sample for the family fits; only 2018–2020 is out-of-sample.
- Costs, the 5/10-pip release spread, swap and the broker minimum are ASSUMED_NOT_BROKER_VERIFIED.
- The oracle's net USD compounds the sizing rule; compare Sharpe, not net, across periods.
- Compute: 13 bounded runs (`crispdm-run -m 2G -t 120s`), ≈ 180 CPU-seconds.
