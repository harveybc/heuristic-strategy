# Anexo T1: perfiles TRAIN de las listas v4 (trabajo de Musashi; consumidos por manifiesto)

Autoría: Musashi, en su worker, 2026-10-22 (predictor `codex/data-coverage-repair-20261018` @4fc98f6d,
`tools/selection_input_profiles.py`). Copia verificada byte a byte: manifiesto `5729d807…`, `PROFILES_core15.csv`
`0ea6a68d…`, `PROFILES_shortlist_strict.csv` `78edb32c…`; entrada = pool
`_s` `32edb424…`, listas core15 `86f8dddf…` y estricta `902622d7…` (iguales a las del spec). Bloque FIT
2020-01-01 → 2023-07-01 por disponibilidad (inicio + 1 h), 21 809 barras, 0 filas de grilla ausentes.

**Alcance (de Musashi, se respeta):** son diagnósticos; no eliminan columnas, no transforman entradas, no eligen
extractores. La significancia es exploratoria (ruido blanco normal, Bonferroni 0.05/48 por feature). Los
estadísticos por fold son descriptivos, no seis desempeños predictivos. Las columnas heredadas con máximos en
INNER/2024 son contexto; ninguna métrica nueva las usa.

## 1. Resumen por grupo de extractor (lista estricta; ids y contraste del manifiesto T3)

ACF y primer rezago no significativo: medianas del grupo, en barras observadas. Redundancia: máximo |Spearman|
dentro de la familia. Estabilidad: máximo del cociente std de medias de fold / std global.

| id T3 | grupo | cols | contraste T3 | worker | ACF₁ | ACF₂₄ | primer rezago no sig. | redund. en familia | estab. fold | faltante máx. |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `inv.raw` | 8 | per_feature | musashi | -0.02 | 0.01 | 2 | 1.00 | 0.32 | 0.00 % |
| 2 | `inv.momentum` | 8 | per_feature | satoshi | 0.91 | 0.01 | 15 | 0.98 | 0.09 | 0.09 % |
| 3 | `calendar` | 13 | pointwise | musashi | 0.97 | 0.99 | 6 | 0.99 | 0.01 | 0.00 % |
| 4 | `px.logret` | 7 | per_feature | satoshi | 0.83 | 0.03 | 13 | 0.77 | 0.10 | 0.22 % |
| 5 | `px.range` | 7 | per_feature | musashi | 0.92 | 0.64 | 10 | 0.90 | 0.50 | 0.22 % |
| 6 | `px.close_loc` | 7 | per_feature | satoshi | 0.67 | 0.02 | 8 | 0.91 | 0.10 | 0.22 % |
| 7 | `px.er` | 7 | per_feature | musashi | 0.66 | -0.01 | 7 | 0.50 | 0.06 | 0.66 % |
| 8 | `px.range_ratio` | 1 | pointwise | satoshi | 0.55 | 0.37 | 7 | — | 0.02 | 0.11 % |
| 9 | `px.hours_since_session_open` | 1 | pointwise | musashi | 0.95 | 0.04 | 25 | — | 0.01 | 0.00 % |
| 10 | `cdl` | 6 | per_feature | satoshi | 0.24 | 0.24 | 9 | 0.63 | 0.26 | 0.00 % |
| 11 | `fx.usdchf` | 4 | per_feature | musashi | 0.85 | 0.05 | 4 | 0.45 | 0.48 | 0.22 % |
| 12 | `dly.gold` | 2 | pointwise | satoshi | 0.98 | 0.40 | 24 | 0.38 | 0.18 | 0.00 % |
| 13 | `cot.eur` | 4 | pointwise | musashi | 1.00 | 0.90 | — | 0.86 | 0.61 | 0.00 % |
| 14 | `fac.fx_f1` | 4 | pointwise | satoshi | 0.68 | 0.01 | 4 | 0.85 | 0.05 | 0.22 % |
| 15 | `ev` | 6 | pointwise | musashi | 0.76 | 0.99 | 5 | 1.00 | 0.00 | 0.00 % |
| 16 | `act` | 1 | pointwise | satoshi | 0.97 | 0.20 | 30 | — | 0.10 | 0.00 % |
| 17 | `wavelet/log_close` | 9 | per_feature | musashi | 0.86 | 0.18 | 5 | 0.86 | 0.63 | 1.17 % |
| 18 | `wavelet/logret_1` | 9 | per_feature | satoshi | 0.79 | 0.15 | 5 | 0.87 | 0.64 | 1.18 % |
| 19 | `hilbert/logret_1` | 3 | per_feature | musashi | -0.04 | 0.02 | 3 | 0.01 | 0.02 | 0.59 % |
| 20 | `strength/fx.usd_strength.ret_1` | 1 | pointwise | satoshi | 0.96 | 0.90 | — | — | 0.65 | 2.35 % |
| 21 | `strength/fx.eur_strength.ret_1` | 1 | pointwise | musashi | 0.55 | 0.01 | 4 | — | 0.01 | 1.17 % |
| 22 | `kalman/log_close` | 1 | pointwise | satoshi | -0.00 | 0.01 | 1 | — | 0.02 | 0.00 % |
| 23 | `px.rv` | 1 | pointwise | musashi | 0.98 | 0.57 | — | — | 0.51 | 0.11 % |
| 24 | `gmm/logret_1+rv_24+log_range` | 4 | per_feature | satoshi | 0.50 | 0.34 | 22 | 0.98 | 0.40 | 0.11 % |
| 25 | `fracdiff/log_close` | 1 | pointwise | musashi | -0.25 | 0.01 | 5 | — | 0.00 | 1.17 % |

`—` en el primer rezago no significativo: ningún rezago ≤ 48 dejó de ser significativo (dependencia larga: COT,
`px.rv`, fuerza del USD). `px.er_1` es **constante** en FIT (cociente de eficiencia de una barra): se conserva y
se marca; con el escalado fijo su std se trata con la regla de columna constante (std 0 → 1).

## 2. Pares con |Spearman| ≥ 0.99 (información muy correlacionada, no veto)

Cada columna del CSV guarda sólo su vecino de máxima correlación: esta tabla es el grafo de esos vecinos (10 aristas
únicas), no una enumeración de todos los pares del conjunto (precisión de Musashi, 2026-10-22).

| columna | grupo | pareja | grupo de la pareja | abs ρ del par | grupo |
| --- | --- | --- | --- | ---: | --- |
| `CLOSE@diff` | `inv.raw` | `RAW_CLOSE_LOG_RETURN` | `inv.raw` | 0.9996 | misma |
| `RAW_BAR_LOG_RETURN` | `inv.raw` | `cdl.body_pips` | `cdl` | 0.9996 | distinta |
| `RAW_CLOSE_LOG_RETURN` | `inv.raw` | `kf.lc.level_minus_close` | `kalman/log_close` | 0.9994 | distinta |
| `RAW_CLOSE_LOG_RETURN` | `inv.raw` | `px.logret_1` | `px.logret` | 1.0000 | distinta |
| `RAW_LOG_RANGE@log` | `inv.raw` | `cdl.range_pips@asinh` | `cdl` | 0.9961 | distinta |
| `RAW_LOG_RANGE@log` | `inv.raw` | `px.range_1@log` | `px.range` | 1.0000 | distinta |
| `ev.london_fix.hours_since` | `ev` | `ev.london_fix.hours_to` | `ev` | 1.0000 | misma |
| `ev.ny_cut.hours_since` | `ev` | `ev.ny_cut.hours_to` | `ev` | 1.0000 | misma |
| `ev.tokyo_fix.hours_since` | `ev` | `ev.tokyo_fix.hours_to` | `ev` | 1.0000 | misma |
| `wv.lc.A4@diff` | `wavelet/log_close` | `wv.r1.A4` | `wavelet/logret_1` | 1.0000 | distinta |

Lectura, sin decidir nada todavía:

- **Seis de estos diez pares cruzan grupos de extractor** (p. ej. `RAW_CLOSE_LOG_RETURN` con `px.logret_1` y con `kf.lc.level_minus_close`,
  `RAW_LOG_RANGE@log` con `px.range_1@log`, `wv.lc.A4@diff` con `wv.r1.A4`; correlación de rangos ≥ 0.999, que no prueba
  igualdad numérica ni de amplitudes). En T3 esto importa: cambiar el tipo de
  extractor de un grupo no elimina esa información del modelo, porque la copia sigue entrando por otro grupo. El
  contraste T3 mide entonces el **tipo de extractor condicionado a las demás entradas presentes**, no una ablación
  de información ni el aporte único del grupo. Así queda declarado en el manifiesto T3 v2.
- Los pares `ev.*.hours_since` / `hours_to` son complementarios dentro del día (ρ = 1 en rango); cada par entra en
  el mismo grupo `ev` y su extractor los ve juntos.
- Ninguna columna se retira por esto (regla del propietario: no perder información útil; regla de Musashi:
  correlación alta no es veto). Si se quiere medir el efecto de la duplicación, el contraste correcto es un brazo
  declarado **sin la copia** con soporte común, como contraste separado tras T3; queda propuesto, no programado.
