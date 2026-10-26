# Prefiltro antes de la selección flotante, v2 (2026-10-26; las ocho correcciones de Musashi aplicadas)

Orden del propietario: endurecer el filtro **existente** (métricas e inferencia causal de la selección v5) antes de la
selección flotante, sin ajustar modelos ni repetir trabajo. Correcciones del auditor: dos clases de descarte, nada
automático por deriva, ganancias de familia y contrastes padre/componente sólo para **ordenar**, redundancia como
selección de columnas, todo medido en el bloque de ajuste sobre un soporte común fijado, autopruebas del selector,
tabla única de umbrales; **60 es una guía de presupuesto, nunca un objetivo alcanzado ajustando umbrales.**

Herramienta: `tools/expert_guided_prefilter.py` (esquema `expert_guided_prefilter.v2`). Evidencia:
`evidence/prefilter/CANDIDATES_STRICT.json` (niveles A y B con sha, registros por regla, sensibilidad). Pruebas:
`tests/test_expert_guided_prefilter.py` (9, incluidas las autopruebas del auditor).

## 1. Dos clases de descarte

| clase | efecto en el SBFS | reglas |
| --- | --- | --- |
| **DEFINITIVO** (sólo defectos demostrados) | fuera del universo de búsqueda | D1 constantes (std 0 entre valores finitos; se registra conteo finito, cuota de faltantes y constancia por periodo; una columna sin valor finito es **ABSENT**, otra clase); D2 duplicados **exactos** (max \|x−y\| = 0 o identidad afín y = a·x + b con residuo máximo < 1e-9); se conserva una representante por clúster |
| **REVERSIBLE** = nivel B | fuera del conjunto inicial hacia atrás, **dentro del pool de reinclusión flotante** | B1 casi-duplicados (Pearson² ≥ 0.99 en el soporte común; R² por periodo registrado); B2 **alerta** de deriva (AUC 2020–22 vs 2023-S1 ≥ 0.75 **y** la columna es un NIVEL crudo; nunca codificaciones de calendario ni ratios de volatilidad); B3 familias sin ganancia positiva en ninguna tarea (el resto de familias sólo se **ordena**); B5 redundancia intra-familia por selección hacia adelante de columnas (bloques coherentes juntos) hasta explicar el 99 % de la varianza **estandarizada** de la familia en TRAIN |
| nivel A | conjunto inicial del SBFS | todo lo demás |

B4 (contraste padre-contra-componentes, `CONTRASTS_PARENT_COMPONENTS.csv`): **no excluye nada**; sus ganancias
pareadas (brazo `C:<componente>` / `P_*` / `Cgroup` contra `none`, misma tarea|familia|contexto|fold) quedan registradas
como orden dentro de la familia (`b4_component_order_not_applied`). La v5 **no** tiene veredicto por columna de la etapa
de relaciones con supuestos de potencia/retardo registrados, así que no existe descarte por "sin dependencia".

Soporte común fijado: filas de TRAIN (2020-01-01 → 2023-07-01) completas en las 121 columnas: **21 155** de 21 809;
hash `65e998ec58ed3850`. Todas las estadísticas (D2, B1, B2, B5) se calculan sobre esas mismas filas; B2 parte el
soporte dentro de TRAIN (2023-01-01). INNER_A y 2024 no se leen.

## 2. Tabla única de umbrales y su sensibilidad (A = nivel A, B = nivel B)

| umbral | valor declarado | regla | sensibilidad |
| --- | --- | --- | --- |
| residuo afín para duplicado exacto | 1e-9 | D2 | los 6 exactos tienen residuo ≤ 1.8e-13; el siguiente par (R² 0.9993) no es afín: insensible entre 1e-12 y 1e-6 |
| Pearson² casi-duplicado | 0.99 | B1 | 0.95 → A 99 / B 15; **0.99 → A 100 / B 14**; 0.999 → A 102 / B 12 |
| AUC de deriva (alerta, sólo niveles) | 0.75 | B2 | 0.70 → A 99 / B 15; **0.75 → 100 / 14**; 0.80 → 100 / 14 |
| ganancia de familia | ≤ 0 en **todas** las tareas → B | B3 | 1 familia (`strength/fx.usd_strength.ret_1`); el resto ordena |
| varianza explicada intra-familia | 0.99 | B5 | 0.98 → A 98 / B 16; **0.99 → 100 / 14**; 0.995 → A 103 / B 11 |
| mínimo de columnas por familia / tamaño mínimo para B5 | 2 / 5 | B5 | ninguna familia toca el mínimo |

Con la regla declarada: **nivel A = 100, nivel B = 14, definitivos = 7** (universo del SBFS = 114). La guía de ≈ 60
no se alcanza y no se persigue ajustando umbrales: lo que reduce el conjunto inicial de verdad es la propia búsqueda
flotante.

## 3. Recuento por regla

| inicio | ABSENT | D1 constante | D2 exacto | universo | B1 casi-dup | B2 deriva | B3 familia | B5 redundancia | **A** | **B** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 0 | 1 | 6 | 114 | 4 | 0 | 1 | 9 | **100** | **14** |

### 3.1 Definitivos (7)

- D1: `px.er_1` (std 0: |r₁|/|r₁|; 217 faltantes).
- D2: `px.logret_1` ≡ `RAW_CLOSE_LOG_RETURN` (idéntica, diff 0); `px.range_1@log` ≡ `RAW_LOG_RANGE@log` (idéntica);
  `ev.tokyo_fix.hours_to`, `ev.london_fix.hours_to`, `ev.ny_cut.hours_to` = 24 − `hours_since` (afín, residuo ≤ 1.8e-13);
  `wv.r1.A4` = `wv.lc.A4@diff` (afín a = 1, residuo 7.9e-16). Representante = la columna de la familia mejor ordenada
  por B3; empate → primera en el orden de la receta.

### 3.2 Nivel B (14, en el pool de reinclusión)

| regla | columnas | registro |
| --- | --- | --- |
| B1 | `RAW_CLOSE_LOG_RETURN` (R² 0.996 con `CLOSE@diff`), `kf.lc.level_minus_close` (0.9955 con `CLOSE@diff`, 0.9993 con `RAW_CLOSE_LOG_RETURN`), `cdl.body_pips` (0.996 con `RAW_BAR_LOG_RETURN`), `cdl.range_pips@asinh` (0.993 con `RAW_LOG_RANGE@log`) | R² por periodo 2020/2021/2022/2023-S1 entre 0.995 y 1.000: estables |
| B2 | ninguna | mayores AUC: `cot.eur.net_share` 0.711 y `cot.eur.net_long` 0.669 (niveles, bajo 0.75); `wv.lc.energy_D4@log` 0.610 (no nivel). AUC año a año por columna en el JSON. La AUC 0.84 de la auditoría A1-5 era contra 2024, que aquí no se usa |
| B3 | `wv.usd.A4` | familia ≤ 0 en las 7 tareas; evidencia débil (A1-2) |
| B5 | `RAW_BAR_LOG_RETURN` (inv.raw); `Stochastic_D`, `CCI` (inv.momentum); `dow_sin`, `dow_cos`, `cal.utc_hod_sin`, `cal.utc_hod_cos`, `cal.london_hod_sin`, `cal.london_hod_cos` (calendar: `hod_*` y `cal.utc_hod_*` son una rotación exacta, A1-4, por eso una de las dos queda en B) | varianza estandarizada explicada, **no** utilidad de trading; traza de la selección hacia adelante en el JSON |

### 3.3 Nivel A (100, en el orden de la receta)

`OPEN@diff`, `HIGH@diff`, `LOW@diff`, `CLOSE@diff`, `RAW_LOG_RANGE@log`, `RAW_ELAPSED_HOURS`, `ATR`, `BB_Width`, `ATR_ratio`,
`rolling_std_24`, `RSI`, `Stochastic_K`, `WilliamsR`, `ROC`, `BB_position`, `price_minus_ema`, `hod_sin`, `hod_cos`,
`cal.utc_dow_sin`, `cal.utc_dow_cos`, `cal.utc_session_asia/europe/us`, `px.logret_2/4/6/12/24/48`,
`px.range_2@log/4@log/6@log/12/24/48`, `px.close_loc_1/2/4/6/12/24/48`, `px.er_2/4/6/12/24/48`,
`px.hours_since_session_open`, `cdl.upper_wick_pips@asinh`, `cdl.lower_wick_pips@asinh`, `cdl.close_minus_low_pips@asinh`,
`cdl.high_minus_close_pips@asinh`, `dly.us10y.level@diff`, `dly.us10y.chg_1d`, `dly.us10y.chg_5d`, `cot.eur.net_long`,
`cot.eur.net_change`, `cot.eur.net_share`, `cot.eur.net_z156`, `fac.fx_f1.ret_1/ret_4/ret_24/kalman_slope`,
`ev.tokyo_fix.hours_since`, `ev.london_fix.hours_since`, `ev.ny_cut.hours_since`, `act.p_hit`, `ssa.trend_minus_close_pips`,
`ssa.c2_pips`, `ssa.c3_pips`, `wv.lc.A4@diff`, `wv.lc.D1..D4`, `wv.lc.energy_D1@diff/D2@log/D3@log/D4@log`, `wv.r1.D1..D4`,
`wv.r1.energy_D1@diff/D2@log/D3@log/D4@log`, `hb.r1.phase_cos`, `hb.r1.phase_sin`, `hb.r1.inst_freq`, `hb.lc_detr.inst_freq`,
`hb.eur.phase_cos`, `hb.eur.phase_sin`, `hb.eur.inst_freq`, `px.rv_24`, `rg.gmm.p0`, `rg.gmm.p1`, `rg.gmm.p2`, `rg.gmm.entropy`.
Lista exacta y sha: `sha256_tier_A` / `sha256_tier_B` en el JSON.

## 4. Orden (información que no excluye)

- B3 ordena las familias por su mejor ganancia media (hgb): `inv.raw` 0.695 ≫ `px.close_loc` 0.068 > `px.logret` 0.017
  > `wavelet/log_close` 0.011 > `cdl` 0.010 > … (tabla `b3_best_gain`).
- B4 ordena componentes dentro de las familias de descomposición (ganancia pareada C:componente vs `none`), p. ej.
  `wavelet/log_close`: A4 +0.0035, D1 +0.0023 > D2…D4 y energías ≤ 0; `gmm`: p0 +0.0032 > p1 > p2 > entropía
  (`b4_component_order_not_applied`). Estas ganancias se midieron sobre `opener3` ("la barra es pivote", A1-2) y
  `hit`: ordenan, no certifican valor direccional.

## 5. Autopruebas del selector (sintéticas, en la suite)

| prueba | resultado |
| --- | --- |
| señal sólo por interacción (dos columnas inútiles solas, útiles juntas) | sobrevive: su familia va a B por falta de evidencia, nunca definitivo |
| componente útil de un padre débil | sobrevive (B4 ordena, no excluye) |
| duplicados exactos (idéntico y afín) | definitivos; el casi-duplicado no |
| deriva legítima (nivel de volatilidad con tendencia real) | nivel B, no borrado; ratios y calendario nunca marcados |
| constante vs ausente | clases distintas con registro |
| B5 | selección de columnas por bloques, mínimo 2, nota "varianza ≠ utilidad" |

## 6. Limitaciones

- Las ganancias de familia y de componente (B3/B4) provienen de tareas de geometría presente (A1-2): sirven para
  ordenar la eliminación hacia atrás, no como evidencia de valor a 12–72 h.
- B2 dentro de TRAIN es menos sensible que contra 2024; los niveles COT y de volatilidad quedan en A y deben
  codificarse como cambios/ratios en la receta derivada (A1-5/A1-6), igual que las decisiones de `ENCODING.csv` no
  aplicadas (A1-11).
- B5 mide varianza explicada; una columna de poca varianza puede ser la útil. Por eso es reversible.
- El soporte común (21 155 filas) excluye 654 filas de TRAIN con algún faltante; registradas por columna en `d1_record`.
