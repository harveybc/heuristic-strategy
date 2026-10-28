# Calendar / implied-vol range experiment — verdict

**DEVELOPMENT.** Owner-approved bounded experiment, 2026-10-19.

**Question:** can a business-day range forecast get from HAR's level down to
≤ ~0,85× MEAN20 using scheduled macro events and implied volatility, and does
that make the direction-free strategy pay?

**Protocol:**
- Inclusion and hyperparameters are decided on **2023**: fit on 2020–2022,
  weekly refits, one-block purge.
- **2024 is used once**, for confirmation, through the sealed adapter.
- TEST closed.
- Rules: SL/TP, units sizing (rel_volume 0,10, 1:100 with 1:10 / 1:50 in the
  appendix) and the frequency rule all apply.
- Tables: `results/calendar_range/TABLAS_CALENDARIO_RANGO.md`.

## 1. Data — the calendar and implied-vol arms could not run (missing objects, nothing fetched)

| Object | Status |
|---|---|
| financial-data `economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet` | Forward-only: 2026-02-05 → 2027-07-14 (acquired 2026-05-01). **No 2020–2024 rows.** |
| `economic_calendar/release_actuals/fxmacrodata/announcements.parquet` | Starts 2024-12-12; 13 rows in 2020–2024 (CHF only). **Not usable.** |
| `economic_calendar/scheduled_events/fred_release_date_proxy` | Dates are FRED *reference periods* (e.g. NFP 2016-01-01), **not release instants.** Not usable as a schedule. |
| EUR/USD implied volatility (EVZ) | **MISSING_OBJECT.** `derivatives/implied_volatility_surfaces` is empty; FRED holds only VIXCLS (S&P 500). |
| feature-eng `tests/data/economic_calendar_2011_2021.csv` | A *test fixture* outside the lake. Date+time, country, "High/Moderate/Low Volatility Expected", event name; ends **2021-04-26**. Time zone undocumented; the importance label may be retrospective. SCHEDULE_ASSUMED_KNOWN_IN_ADVANCE. It cannot cover the 2023 decision or 2024 — used only for a descriptive diagnostic. |

Coverage of high-impact events in the fixture, 2020-01 … 2021-04: about 30–40 USD
and 13–26 EUR per month. The most frequent families are crude inventories,
jobless claims, ECB speeches, CPI, PMI, NFP and retail sales
(`fixture_high_impact_by_month.csv`, `fixture_high_impact_by_family_usd_eur.csv`).

**Fetch commands — recorded, NOT run** (details in `INVENTORY.json`):
- **EVZ:** add `EVZCLS` to the FRED stress list of financial-data
  `_scripts/workers/stage13 FRED follow-up worker (stage13_*_followup_worker.py)` and run it with the
  governed `FRED_API_KEY`. A single-series call is preferable, because the worker
  also refreshes other sources.
- **Historical schedule:**
  - `stage15_fxmacrodata_worker.py --currencies USD EUR`; the provider history
    observed so far starts 2024-12;
  - FRED `release/dates` (realtime vintages) for Employment Situation, CPI, GDP
    and Retail Sales needs a small worker extension;
  - FOMC/ECB meeting calendars have no lake adapter yet.

**Descriptive diagnostic** (fixture period, TRAIN data, no decision taken from it):

| B | Blocks | n | Mean range | HAR range MAE × MEAN20 |
|---|---|---:|---:|---:|
| 24 h | with a high-impact USD/EUR event inside | 104 | 66,4 pips | 0,937 |
| 24 h | without | 34 | 51,5 pips | 0,855 |
| 6 h | with | 288 | 47,9 pips | 0,922 |
| 6 h | without | 569 | 32,3 pips | 0,765 |

Event blocks are 30–50 % wider, and HAR is clearly worse on them. That is exactly
the information a scheduled-event term should add, **so acquiring the historical
schedule is justified.**

## 2. What ran: richer causal volatility features and a small nonlinear model

- **Features** (`app/short_horizon/vol_features.py`, all from bars before t0):
  - HAR excursion terms (1 day / 1 week / 1 month) and calendar position;
  - Parkinson volatility over 6 / 24 / 120 / 480 bars, normalized absolute
    returns over 1 day and 1 week, efficiency ratio, previous excursions.
- **Predictors:**
  - **HAR** (baseline);
  - **HAR-X** (least squares on all features);
  - **GBM**: median gradient boosting, capacity {depth 2/3} × {100/300
    iterations} chosen on 2023 by mean |Û err| + |D̂ err| (the quantities the bracket consumes) → depth 2, 100 iterations for both B (B24: 43,61 pips vs 44,00–44,58; B6: 22,85 vs 22,87–23,07). Other capacities had a slightly better *range* MAE on 2023 but worse side excursions; all are in the tables.
  - All refit weekly on past blocks only.
- **Direction arms:** coin flip (direction-free) and the causal logistic order
  model.
- **Bookkeeping:** every prediction file was sealed before scoring
  (`SEALS.json`), and the GBM choice was recorded with its digests before any
  2024 run (`CHOICE_B24.json`, `CHOICE_B6.json`).

## 3. Results (RF-M, 1:100, costs ×1; all configurations valid at ≥ 3 trades/week)

| B | Year | Predictor | Range MAE ×MEAN20 (×pers.) | Û / D̂ ×MEAN20 | Sharpe, coin direction | Sharpe, logistic direction |
|---|---|---|---:|---:|---:|---:|
| 24 h | 2023 (decide) | HAR | 0,855 (0,626) | 0,940 / 0,949 | −1,03 | −3,57 |
| 24 h | 2023 | HAR-X | 0,855 (0,626) | 0,942 / 0,957 | −0,82 | −3,27 |
| 24 h | 2023 | GBM d2/n100 | 0,878 (0,643) | 0,918 / 0,927 | −1,85 | −3,37 |
| 24 h | 2024 (confirm) | HAR | 0,833 (0,710) | 0,981 / 0,907 | −1,12 | −0,20 |
| 24 h | 2024 | HAR-X | 0,844 (0,719) | 0,970 / 0,927 | −1,21 | −0,13 |
| 24 h | 2024 | GBM d2/n100 | 0,843 (0,718) | 0,911 / 0,898 | −0,96 | −0,31 |
| 6 h | 2023 (decide) | HAR | 0,783 (0,593) | 0,949 / 0,893 | −4,49 | −3,15 |
| 6 h | 2023 | GBM d2/n100 | **0,765** (0,579) | 0,920 / 0,871 | −4,66 | −3,69 |
| 6 h | 2024 (confirm) | HAR | 0,803 (0,617) | 0,941 / 0,898 | −4,76 | −4,37 |
| 6 h | 2024 | GBM d2/n100 | **0,774** (0,595) | 0,900 / 0,862 | −6,16 | −5,65 |

Leverage 1:10 / 1:50 / 1:100 changes Sharpe by ≤ 0,04 and only scales profit and
loss (appendix in the tables).

**Figures** (PNG + PDF; Sharpe vs range MAE with the A1r/A3r break-even curves on
2023 and 2024, and markers for each real predictor):
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/calendar_range/calendario_rango_B24.png
- https://github.com/harveybc/heuristic-strategy/blob/satoshi/short-horizon-strategy-20261018/docs/short_horizon/results/calendar_range/calendario_rango_B6.png

## 4. Verdict (decided on 2023, confirmed on 2024)

1. **Calendar and implied vol:** not testable now. The data needed (2020–2024
   scheduled release instants; EVZ) is missing from the lake, and nothing was
   fetched. The descriptive diagnostic says the event term targets the largest
   remaining error, so acquire the historical schedule first.
2. **Range MAE ≤ 0,85×:** already reached, by HAR at 24 h (0,855× / 0,833×) and
   by every model at 6 h.
   - The 2023-chosen GBM improves the 6 h range forecast by 2–4 % (0,765× vs
     0,783× on 2023; **confirmed on 2024: 0,774× vs 0,803×**). It is included at 6 h.
   - At 24 h nothing beats HAR on 2023, so HAR is kept.
3. **The strategy still loses** with every real predictor:
   - 24 h: Sharpe −0,8 to −1,9 on 2023 and −0,1 to −1,2 on 2024;
   - 6 h: Sharpe −3 to −6.
4. **Why the range target was not sufficient:**
   - The strategy's TP and SL use the **side excursions** Û and D̂. Those remain
     at 0,90–0,98× MEAN20; in the range, U and D errors partly cancel.
   - Real errors are not the blend's scaled copy of the naive error. At 24 h on
     2024 the coin-direction markers sit about 1,5–1,8 Sharpe below the A1r curve
     at the same range MAE (A1r ≈ +0,5 at 0,83–0,84×). On 2023 they sit near the
     A1r curve, but its break-even is stricter that year (≈ 0,67×). At 6 h the
     A1r break-even is ≈ 0,6× (2023) and ≈ 0,53× (2024), far from every real
     predictor (read from the figures).
   - **The quotable requirement must be stated per side** (Û, D̂ MAE) **and
     checked by backtesting real predictions**, as done here; range MAE alone
     over-promises.
5. **Next step:** acquire the 2020–2024 scheduled calendar (and EVZ) under the
   data-gov contract, then rerun this harness unchanged with the calendar terms.
   Judge the result by U/D MAE, the 2023 Sharpe of the real predictions, and a
   single 2024 confirmation.

## Limits

- **Fixture:** the event diagnostic uses an outside-lake test fixture with an
  undocumented time zone and a possibly retrospective importance label. It is
  descriptive only.
- **Model scope:** small GBM capacity grid; no quantile regression beyond the
  median objective; no event or implied-vol features (missing).
- **Data:** one market path per year; 2024 is DEVELOPMENT.
- **Assumptions:** costs, swap and the broker minimum are ASSUMED_NOT_BROKER_VERIFIED.
- **Clock:** the vendor bar clock is still under repair.
- **Compute:** about 12 bounded runs (`crispdm-run -m 2G -t 120s`), roughly 420
  CPU-seconds including the inner-year reference curves.
