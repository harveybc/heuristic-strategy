# Prefiltro estricto antes de la selección flotante (2026-10-26)

Orden del propietario: endurecer el filtro **existente** por métricas e inferencia causal para que la selección flotante
(SBFS, que reentrena el modelo por subconjunto) arranque desde aproximadamente la mitad de los 121 candidatos de la
receta v2, sin ajustar ningún modelo nuevo ni repetir la selección. Herramienta: `tools/expert_guided_prefilter.py`;
evidencia: `evidence/prefilter/CANDIDATES_STRICT.json`; pruebas: `tests/test_expert_guided_prefilter.py` (7).

## 1. Reglas, declaradas antes de aplicarlas (umbrales en `RULES`)

| regla | definición | umbral declarado |
| --- | --- | --- |
| R1 | desviación estándar en TRAIN (2020-01-01 → 2023-07-01) igual a 0 o no finita → fuera | — |
| R2 | R² ≥ umbral entre dos columnas (filas completas de TRAIN) forma un clúster; se conserva **una** representante por clúster: la de la familia con mayor ganancia media `opener3` (RANKING.csv, hgb); empate → la primera en el orden de la receta; los bloques coherentes (pares seno/coseno, `rg.gmm.p0/p1/p2`) se mueven juntos | 0.99 |
| R3 | AUC adversaria de una sola característica, TRAIN 2020–2022 contra 2023-S1 (ambos **dentro** de TRAIN; 2024 no se usa) ≥ umbral → fuera, salvo las codificaciones de calendario declaradas (`hod_*`, `dow_*`, `cal.*`), que se conservan por diseño | 0.75 |
| R4 | familias cuya ganancia media `delta_loss` (hgb, RANKING.csv) es ≤ 0 en **todas** las tareas → fuera todas sus columnas; las familias `RESCUED_INTERACTION` se conservan | — |
| R5 | etapa registrada de contrastes padre-contra-componentes (`CONTRASTS_PARENT_COMPONENTS.csv`): un componente cuya ganancia pareada sobre la línea base `none` de su familia (misma tarea/familia/contexto/fold, hgb) es ≤ 0 en **ambas** tareas informativas (`opener3`, `hit`) → fuera; un bloque coherente sale sólo si todos sus miembros están ≤ 0 en ambas | — |
| R6 | dentro de cada familia superviviente con k > 4 columnas se conservan tantas columnas como componentes principales explican la varianza declarada en TRAIN, elegidas por carga cuadrática ponderada por varianza (bloques como unidad); nunca menos de 2 | 0.99 |

**Sobre R5 (lo que existe y lo que no).** La selección v5 **no** registra ningún veredicto causal ni de dependencia
retardada **por columna**: sus decisiones por familia usan tres clases de evidencia (`block_permutation`,
`interaction`, `retrained_contrast`), todas a nivel de familia. La única evidencia de etapa a nivel de columna que
existe es el contraste padre-contra-componentes para las familias de descomposición (wavelet, hilbert, SSA, Kalman,
GMM, fuerza EUR/USD); R5 usa exactamente esa tabla, con el emparejamiento por clave `tarea|familia|contexto|fold`,
brazo `C:<componente>` contra brazo `none`. No se inventó ningún veredicto causal.

## 2. Resultado con la regla declarada

Filas TRAIN 21 809 (completas para R2/R6: ver JSON). Recuento tras cada regla:

| inicio | R1 | R2 | R3 | R4 | R5 | R6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 120 | 110 | 110 | 109 | 92 | **82** |

**La regla declarada deja 82 columnas, no ≈ 60.** No se retocó ningún umbral mirando qué columna caía; la sección 4
da los conjuntos de umbrales que llegarían a ≈ 60 y lo que cada uno implica.

### 2.1 Columnas eliminadas, con su razón

| regla | n | columnas |
| --- | ---: | --- |
| R1 constante | 1 | `px.er_1` (std 0: |r₁|/|r₁|) |
| R2 duplicado | 10 | `RAW_CLOSE_LOG_RETURN`, `px.logret_1`, `kf.lc.level_minus_close` (clúster con `CLOSE@diff`, R² ≥ 0.999); `cdl.body_pips` (≡ `RAW_BAR_LOG_RETURN`); `px.range_1@log`, `cdl.range_pips@asinh` (≡ `RAW_LOG_RANGE@log`); `ev.tokyo_fix.hours_to`, `ev.london_fix.hours_to`, `ev.ny_cut.hours_to` (= 24 − `hours_since`); `wv.r1.A4` (≡ `wv.lc.A4@diff`) |
| R3 proxy de fecha | 0 | ninguna ≥ 0.75; la mayor es `cot.eur.net_share` 0.715 (la AUC 0.84 de la auditoría A1-5 era contra 2024, que aquí no se usa) |
| R4 familia sin evidencia | 1 | `wv.usd.A4` (familia `strength/fx.usd_strength.ret_1`: ≤ 0 en las 7 tareas) |
| R5 componente sin ganancia | 17 | `wv.lc.D2`, `wv.lc.D3`, `wv.lc.D4`, `wv.lc.energy_D1@diff`, `wv.lc.energy_D2@log`, `wv.lc.energy_D3@log`, `wv.lc.energy_D4@log`; `wv.r1.D1`, `wv.r1.D2`, `wv.r1.D3`, `wv.r1.D4`, `wv.r1.energy_D1@diff`, `wv.r1.energy_D3@log`, `wv.r1.energy_D4@log`; `ssa.trend_minus_close_pips`, `ssa.c2_pips`; `hb.eur.inst_freq` |
| R6 redundancia intra-familia | 10 | `inv.raw` (k=6): `RAW_BAR_LOG_RETURN`; `inv.momentum` (k=5): `Stochastic_K`, `Stochastic_D`, `WilliamsR`; `calendar` (k=7): `dow_sin`, `dow_cos`, `cal.utc_session_asia/europe/us`; `px.close_loc` (k=6): `px.close_loc_1`; `px.logret`, `px.range`, `px.er` (k=6 de 6): ninguna |

### 2.2 Candidatos (82, en el orden de la receta)

`OPEN@diff`, `HIGH@diff`, `LOW@diff`, `CLOSE@diff`, `RAW_LOG_RANGE@log`, `RAW_ELAPSED_HOURS`, `ATR`, `BB_Width`,
`ATR_ratio`, `rolling_std_24`, `RSI`, `CCI`, `ROC`, `BB_position`, `price_minus_ema`, `hod_sin`, `hod_cos`,
`cal.utc_hod_sin`, `cal.utc_hod_cos`, `cal.utc_dow_sin`, `cal.utc_dow_cos`, `cal.london_hod_sin`, `cal.london_hod_cos`,
`px.logret_2/4/6/12/24/48`, `px.range_2@log/4@log/6@log/12/24/48`, `px.close_loc_2/4/6/12/24/48`,
`px.er_2/4/6/12/24/48`, `px.hours_since_session_open`, `cdl.upper_wick_pips@asinh`, `cdl.lower_wick_pips@asinh`,
`cdl.close_minus_low_pips@asinh`, `cdl.high_minus_close_pips@asinh`, `dly.us10y.level@diff`, `dly.us10y.chg_1d`,
`dly.us10y.chg_5d`, `cot.eur.net_long`, `cot.eur.net_change`, `cot.eur.net_share`, `cot.eur.net_z156`,
`fac.fx_f1.ret_1/ret_4/ret_24/kalman_slope`, `ev.tokyo_fix.hours_since`, `ev.london_fix.hours_since`,
`ev.ny_cut.hours_since`, `act.p_hit`, `ssa.c3_pips`, `wv.lc.A4@diff`, `wv.lc.D1`, `wv.r1.energy_D2@log`,
`hb.r1.phase_cos`, `hb.r1.phase_sin`, `hb.r1.inst_freq`, `hb.lc_detr.inst_freq`, `hb.eur.phase_cos`,
`hb.eur.phase_sin`, `px.rv_24`, `rg.gmm.p0`, `rg.gmm.p1`, `rg.gmm.p2`, `rg.gmm.entropy`.
Lista exacta y sha en `evidence/prefilter/CANDIDATES_STRICT.json` (`sha256_candidates`).

## 3. Sensibilidad a los dos umbrales ajustables (cuenta final)

| R3 \\ R6 | 0.98 | 0.99 | 0.995 |
| --- | ---: | ---: | ---: |
| 0.70 | 78 | 81 | 83 |
| 0.75 | 79 | **82** | 84 |
| 0.80 | 79 | 82 | 84 |

Los dos umbrales declarados como ajustables sólo mueven la cuenta entre 78 y 84: la masa que queda está en familias
de ≤ 4 columnas (R6 no las toca) y en columnas sin duplicado ni proxy de fecha.

## 4. Qué haría falta para llegar a ≈ 60 (reportado, **no aplicado**)

| palanca alternativa | cuenta | qué implica |
| --- | ---: | --- |
| R6 varianza 0.95 | 73 | se pierde 1–5 % de la varianza intra-familia en las familias grandes |
| R6 varianza 0.90 | 65 | se pierde hasta 10 %; cuestionable para `px.logret`/`px.range` (escalas de horizonte) |
| R2 0.95 + R6 0.90 | 63 | además trata como duplicados pares con R² 0.95–0.99 (15 pares) |
| R5 "≤ 0 en **cualquiera** de las dos tareas" + R6 0.99 | 74 | más agresivo con componentes que ganan en una sola tarea |
| R5 "cualquiera" + R6 0.95 | 65 | combinación de las dos anteriores |

El conjunto más cercano a 60 con palancas interpretables es **R2 0.95 + R6 0.90 → 63** o **R5-cualquiera + R6 0.95 →
65**. Decisión del propietario: aplicar una de ellas (se declara y se re-ejecuta con el mismo script) o aceptar 82.

## 5. Limitaciones (para la auditoría)

- Las ganancias por familia (R4) y por componente (R5) se midieron en la selección v5 sobre `opener3` ("la barra que
  acaba de cerrar es pivote", resoluble con tres barras, auditoría A1-2) y `hit`: son evidencia **débil** sobre valor
  direccional a 12–72 h. Por eso R4 sólo elimina familias ≤ 0 en **todas** las tareas (1 columna) y R5 exige ≤ 0 en
  **ambas** tareas informativas.
- R3 dentro de TRAIN (2020–22 vs 2023-S1) es menos sensible que contra 2024 (A1-5: `cot.eur.net_share` 0.84 allí,
  0.715 aquí); los niveles semanales COT y los niveles de volatilidad en unidades de precio **siguen dentro** de los 82
  y deben codificarse como cambios/ratios (A1-5/A1-6) en la receta derivada, no aquí.
- R6 elige por carga en componentes principales de TRAIN, no por utilidad predictiva: es una regla de redundancia,
  declarada como tal.
- `hod_sin/cos` y `cal.utc_hod_sin/cos` son una rotación exacta (A1-4) pero R2 no las une (el R² es por pares); R6
  las conserva ambas en `calendar` (k = 7). Queda señalado.
- Las decisiones de codificación de `ENCODING.csv` que la receta no aplicó (A1-11: régimen GMM one-hot, relojes de
  evento ventana 2 h) se aplican al derivar la receta, no en este filtro.
