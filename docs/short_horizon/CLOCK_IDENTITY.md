# Clock identity of the EURUSD H1 lake and its witness against the admitted root (plan 14c/16)

Status: DEVELOPMENT. Measured 2026-10-19, window 2020-01-01 to 2025-01-01 UTC (exclusive). No row at or
after 2025-01-01 UTC was read from either source. Generated tables: `results/clock_identity/WITNESS_TABLE.md`
(from `WITNESS.json`). Everything below is aggregates.

## 1. What was assumed, and what the measurement says

The working hypothesis was "the lake label clock is New York local time with DST" (+4 h in US summer,
+5 h in US winter). Outside the transition weeks that holds. **In the transition weeks it does not.** The
lake's offset follows the **EU** DST calendar, not the US one:

| rule (tz-database, `zoneinfo`) | verdict | joined | OHLC exact | clock residuals | US/EU gap-week rows exact |
|---|---|---:|---:|---:|---:|
| `EU_DST_UTC_MINUS_5`: label = Europe/London wall time − 5 h | **EXACT_CLOCK_MAPPING** | 30,251 | 30,243 (0.99974) | 0 | 1,968 / 1,968 |
| `AMERICA_NEW_YORK_WALL`: label = America/New_York wall time | CLOCK_RESIDUALS_PRESENT | 30,236 | 28,275 (0.93514) | 1,953 | 0 / 1,953 |
| `FIXED_UTC_MINUS_5`: UTC−5 all year (vendor's documented clock) | CLOCK_RESIDUALS_PRESENT | 30,107 | 12,549 (0.41681) | 17,550 | 1,968 / 1,968 |

"Gap weeks" are the hours where the US and EU offsets differ: from the second Sunday of March to the last
Sunday of March, and from the last Sunday of October to the first Sunday of November (~4 weeks a year).
In those weeks every NY-mapped bar sits exactly one hour off an admitted bar with the same OHLC (class
CLOCK), and every EU-mapped bar is exact. The weekly open confirms it: the lake's Sunday open label is 17:00
in most weeks and 16:00 in the gap weeks (the market opens 17:00 New York, which is 21:00 UTC then, and
21:00 − 5 h = 16:00).

So the clock is UTC−5 outside EU summer time and UTC−4 inside it. A plausible origin, not verified, is a
European-DST server clock shifted by a fixed amount. The rule is stated as a tz-database rule
(Europe/London, label = wall − 5 h). No hour shift is applied by hand anywhere.

## 2. The declared identity (`results/clock_identity/CLOCK_IDENTITY.json`)

- identity sha256 `cb565a65f8507bfaf85b671847731bf866e7d07f14f7fbb77ae27910326d036b`
- rule `EU_DST_UTC_MINUS_5` (tz `Europe/London`, `label_minus_wall_hours = -5`), tzdata `2024b` via `zoneinfo`.
  The 14 transition instants for 2019–2025 are pinned in code (`PINNED_TRANSITIONS`). They fall at 01:00 UTC
  on the last Sundays of March and October. A test checks that the installed tz database reproduces them.
- source: lake `market_data/forex/g10/eurusd/1h.parquet` sha256 `c6628375…4e2e`, naive label = bar start
  (the file's UTC type tag is not the clock).
- out: `UTC_BAR_START`, availability bar start + 1 h.
- transition policy: nonexistent → raise, ambiguous → raise. Every transition of both candidate rules falls
  inside the weekend closure (tested), so a bar at such a label would be a defect. None occurs (0 / 0).
- mapping window 2019-12-01 to 2025-01-01 UTC: 30,854 rows (13,157 at −5 h, 17,697 at −4 h), digest
  `b3f617eff7ccffd135a6ec300393f420d0b7252653127c7cbafc7519cfefd4fa`. The encoding is declared in the record.
  `verify_identity` recomputes it from the lake bytes.
- witness sha256 `db3f595b2e464cbb2a550de15097c48e5b8540f10db46380c403c689a6bea6d5`.

## 3. Witness (EU rule; per year)

| year | lake mapped | admitted | joined | OHLC exact | exact rate | clock | source (partial/all) | lake-only | admitted-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 6,230 | 6,232 | 6,212 | 6,212 | 1.000000 | 0 | 0/0 | 18 | 20 |
| 2021 | 6,219 | 6,235 | 6,214 | 6,214 | 1.000000 | 0 | 0/0 | 5 | 21 |
| 2022 | 6,239 | 6,231 | 6,230 | 6,229 | 0.999839 | 0 | 1/0 | 9 | 1 |
| 2023 | 5,409 | 6,197 | 5,381 | 5,377 | 0.999257 | 0 | 3/1 | 28 | 816 |
| 2024 | 6,243 | 6,221 | 6,214 | 6,211 | 0.999517 | 0 | 3/0 | 29 | 7 |

- **Residual mismatches: 8, all SOURCE, 0 CLOCK.** Three are the admitted root's three FXCM fallback rows,
  which come from a different vendor. Five are `independent-hf` rows: four with a partial difference (some OHLC
  fields equal) and one with every field different.
- **Lake-only rows: 89.** All 89 fall inside the admitted root's declared closure intervals: holiday and
  weekend-edge bars that the admitted calendar excludes. Switching to `admitted` drops them.
- **Admitted-only rows: 865**, of which 816 are in 2023. These are the lake's holes.
- **2023H1 coverage: admitted 3,111 vs lake 2,445**, a net gain of 666 rows (+27.2 %). Of those, 675 are
  admitted-only (February–July 2023, about 150 a month) and 9 are lake-only holiday bars. Of the 2,436 joined
  2023H1 rows, all 2,436 are bitwise equal.
- Matches are **bitwise** (30,243 bitwise = 30,243 within 1e-9). The admitted `independent-hf` series and
  the lake therefore share an upstream on > 99.9 % of bars. That supports the clock mapping well, but it
  does not verify the prices independently.

**Verdict: the mapping is exact.** `EU_DST_UTC_MINUS_5` maps every lake label to the admitted UTC bar start,
with 0 clock residuals and 0 transition-hour labels. The NY-with-DST hypothesis is refuted for the US/EU gap
weeks (1,953 rows, 0 exact), and the vendor's documented fixed UTC−5 is refuted for EU summer time.

## 4. Joined view and adapter

- Key `(asset, utc_bar_start, bar_convention, source)`, with source `admitted:<sha>` or
  `lake_mapped:<rule>:<sha>` (`clock_identity.joined_view`). The wide comparison
  (`comparison_view`) carries `ohlc_equal` as supporting evidence only (F04).
- `config/short_horizon/eurusd_h1_bar_sources.json` declares three sources. `bar_source.load_ohlc(path, s, e)`
  with no `source` is `data.load_ohlc` (byte-identical, tested on the real file: 30,854 rows, equal hashes).
  `source="admitted"` and `source="lake_mapped"` return the same OPEN/HIGH/LOW/CLOSE frame, indexed by the
  UTC bar start. `load_ohlc_from_config` checks the byte pins and the pinned clock identity.
- The legacy `lake_label` TEST boundary sits on the label clock, which runs 4–5 h behind UTC. On this file it
  happens to read no 2025-UTC bar: the last label before 2025 maps to 2024-12-31 21:00 UTC, and the legacy and
  mapped windows hold the same 30,854 rows. That is luck, not a guarantee. The mapped loaders bound the
  read by the label of the UTC end.
- Not done here: a business calendar for the admitted source (its `CLOSURE_INTERVALS.json` is the candidate).
  `cal.*` / hour-of-day features also change meaning under UTC.

Reproduce (CPU, under 1.5 GB):
`FINANCIAL_DATA_ROOT=<financial-data> ADMITTED_ROOT=<admitted copy> crispdm-run -q -m 1500M -t 600s -n clock-id -- python -m app.short_horizon.clock_identity --config config/short_horizon/eurusd_h1_bar_sources.json --out docs/short_horizon/results/clock_identity`
