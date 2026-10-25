# Ronda limpia guiada por el experto: resultados de las etapas A y A' (Satoshi, 2026-10-25)

Para el propietario y Musashi. Diez brazos, tres semillas cada uno, sobre la receta aceptada `recipe_5a8f2155318ba7c7`
(sha `93ac66dc…`), mismo store, mismo soporte (hash INNER_A `f1c592d53da55019`), contrato de replay (1 entrada/día,
1:50, ≥ 2 órdenes/semana como validez). **Todo es replay de estudiantes por imitación (E3-M) en datos de desarrollo:
TRAIN es dentro de muestra, INNER_A (2023 S2) decide, 2024 es una lectura por brazo que nunca decide, TEST cerrado.
Ninguna cifra es rentabilidad ni una estrategia operable.** Tablas completas, verificación y tabla de cierre generadas
desde los artefactos por `tools/expert_guided_round_results.py` (4 pruebas) en `evidence/a_aprime/RESULTS.{json,md}`;
decisiones del lanzador en `DECISION_A.json`, `DECISION_A_prime.json`; digests de los 40 archivos de resultado en
`RESULT_FILES.sha256`; bitácora de la cola en `LOG` y `LAUNCH_STATUS.md`.

## 1. Tabla (Sharpe diario·√365 del replay; semillas 20261019 / 20 / 21)

| etapa | brazo | etiquetas | tope | TRAIN | INNER_A | INNER_A media ± σ | brecha | DD % INNER_A | órd/sem | 2024 (una lectura) |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| A | **both_hold_shorter** | both + demostradores 2, rev10 | 10G | 2.62 / 2.89 / 3.61 | 1.70 / 3.09 / −0.12 | **1.56 ± 1.31** | 1.48 | 7.0 / 5.1 / 6.3 | 2.8 / 3.4 / 3.0 | −0.01 / −0.90 / −1.31 |
| A | both_hold | both | 6G | 2.78 / 2.95 / 2.33 | 2.86 / 0.02 / 1.55 | 1.48 ± 1.16 | 1.21 | 3.4 / 5.1 / 6.2 | 2.9 / 2.0 / 2.9 | −1.44 / −1.10 / −0.93 |
| A | zz_hold (control) | ZigZag | 6G | 2.71 / 2.84 / 1.80 | 0.42 / 2.51 / 0.48 | 1.13 ± 0.97 | 1.32 | 8.3 / 4.6 / 8.8 | 3.4 / 2.6 / 3.7 | 0.04 / 0.16 / −0.90 |
| A | opp_hold_shorter | opportunity + 2, rev10 | 10G | 3.49 / 3.13 / 3.11 | 1.01 / 0.74 / 1.22 | 0.99 ± 0.20 | 2.26 | 6.6 / 12.1 / 7.6 | 2.8 / 3.3 / 2.3 | −2.90 / −1.67 / −1.83 |
| A | opp_hold | opportunity | 6G | 3.82 / 3.72 / 2.17 | 1.71 / −0.14 / −1.16 | 0.14 ± 1.19 | 3.10 | 5.8 / 8.9 / 13.1 | 3.8 / 2.1 / 3.6 | −1.00 / −1.04 / −2.06 |
| A | opp_optimum | opportunity, cierre optimum | 6G | 1.68 / 1.95 / 1.54 | −0.43 / 0.96 / −0.28 | 0.08 ± 0.62 | 1.64 | 8.5 / 6.0 / 6.1 | 2.6 / 2.6 / 2.0 | 0.66 / −1.08 / −1.37 |
| A | zz_hold_shorter | ZigZag + 2, rev10 | 10G | 2.01 / 1.76 / 1.54 | −1.88 / 1.27 / −1.09 | −0.57 ± 1.34 | 2.34 | 11.1 / 7.4 / 10.3 | 3.0 / 3.7 / 2.6 | −0.94 / −0.72 / −0.75 |
| A' | multires_deap | multires, umbrales DEAP | 7G | 3.25 / 2.16 / 2.57 | −0.84 / −0.28 / −0.91 | −0.68 ± 0.28 | 3.34 | 11.8 / 10.4 / 12.6 | 4.0 / 3.9 / 3.1 | 0.98 / −1.02 / −1.94 |
| A' | multires_default | multires, umbrales por defecto | 7G | 2.51 / 1.86 / 0.96 | −1.67 / −0.60 / −1.47 | −1.25 ± 0.46 | 3.02 | 9.0 / 5.7 / 10.0 | 3.0 / 3.7 / 2.1 | 0.18 / −1.28 / −0.73 |
| A' | multires_deap_weighted | multires DEAP + pesos en la pérdida | 7G | 2.51 / 2.50 / 2.32 | −0.71 / −2.26 / −1.10 | −1.36 ± 0.66 | 3.80 | 7.2 / 14.2 / 7.3 | 3.4 / 4.2 / 3.8 | −1.16 / −1.26 / −1.35 |

Todos los brazos válidos en las tres semillas; 5 460 parámetros cada uno; tasas de acierto INNER_A 26–53 %.

## 2. Decisión (regla congelada antes de leer)

- **Etapa A: `both_hold_shorter`** (1.56), por delante de `both_hold` (1.48) y del control (1.13). Margen sobre el
  segundo **+0.08**, con desviación entre semillas agrupada **0.97**: la regla elige, pero la diferencia entre los dos
  `both` **no es resoluble** con tres semillas; lo que sí separa de la dispersión es que las dos etiquetas `both`
  superan al control y que `opportunity` sola y `zz_hold_shorter` quedan por debajo.
- **Etapa A': ninguna variante multires supera al control** (`zz_hold` 1.13 frente a −0.68 / −1.25 / −1.36). El
  lanzador la decide con el control repetido como referencia: **control retenido**.
- **Conjunta (10 brazos, misma receta y soporte): `both_hold_shorter`.** La etapa B hereda sus etiquetas (`both`,
  hold_through) y sus demostradores extra (2, rev10), con el tope de 10G.

## 3. Lecturas honestas

1. **Dispersión ≈ efecto.** σ entre semillas 0.2–1.3 frente a diferencias entre brazos de 0.1–0.5 en la cabeza de la
   tabla. Tres semillas no bastan para ordenar `both_hold_shorter`, `both_hold` y el control con confianza; el orden es
   el de la regla, no una afirmación científica. Si la etapa B se decide sobre `both_hold_shorter` frente a
   `both_hold`, habrá que ampliar semillas (≥ 5) y declararlo.
2. **2024 (una lectura) es negativo en casi todo**: el ganador −0.01 / −0.90 / −1.31; `both_hold` negativo en las
   tres; el control 0.04 / 0.16 / −0.90. Es la misma lectura que la C v5 (INNER_A mejora, 2024 empeora). 2024 no es
   holdout y no decide, pero avisa de que nada aquí generaliza fuera de 2023 S2.
3. **`opportunity` sola empeora** (brechas 2.3–3.1, el mayor sobreajuste de la tabla) y el cierre `optimum` no ayuda.
4. **Multi-resolución (A') empeoró claramente al estudiante.** Con los umbrales DEAP las etiquetas de apertura caen a
   1 048 L / 1 048 S en TRAIN (contra 1 935 / 1 923 del ZigZag) y NOP sube a 13 367 de 15 485 filas (86 %); el estudiante
   opera menos y peor (acierto 26–51 %, DD 10–14 %). Con los umbrales por defecto (1 854 / 1 878 aperturas) también
   negativo. Los **pesos en la pérdida** empeoran más (−1.36). Lecturas posibles, ninguna probada: (a) el objetivo
   hindsight del DEAP (Sharpe de la **demostración**) no se traslada al estudiante; (b) el NOP graduado/negativo
   quita al abridor los ejemplos de tramos intermedios que el ZigZag sí le daba; (c) el cerrador por aptitud cierra
   demasiado (CLOSE 9 958 frente a HOLD 5 527 en TRAIN con DEAP; 6 552 / 8 933 por defecto). La aptitud **sí** ordena
   las resoluciones como etiquetas, pero **como objetivo de entrenamiento con estos umbrales, no**. Esto cierra la
   pregunta del propietario con dato, no con opinión: la escalera multi-resolución, tal como la convertimos en
   etiquetas, no mejoró a los agentes en este diseño.
5. Nota de artefacto: `labels.sample_weights` en el E3M.json del brazo ponderado conserva el texto de L1 ("la pérdida no
   los consume") mientras `weights_in_loss` es `true`; manda la bandera (código de L2); el texto se corregirá.

## 4. Verificación

Los diez `E3M.json` registran la identidad y el digest de la receta, `agent_state none`, `verify true`, los mismos
hashes de soporte por periodo (TRAIN `fa480f4e…`, INNER_A `f1c592d5…`, 2024 `3364cd26…`) y las mismas filas y marcas de
tiempo en la sonda de cada módulo (8 filas INNER_A, estado cero): el tensor de entrada fue el mismo en todos los brazos.
Sin desviaciones. La cola no produjo recibos `receipted_run`; en su lugar, `RESULT_FILES.sha256` fija los 40 archivos
leídos.

## 5. Incidentes de memoria (declarados)

- Los tres brazos con demostradores extra (`*_shorter`) tocaron el tope de 6 GiB (pico 5.99–6.02 GB) y `crispdm-run`
  los detuvo por presión sostenida. El host tenía 7 GiB disponibles por una VM residente (6.8 GiB); el reintento a 9G
  quedó en cola por falta de margen y el de 7G fue rechazado por el almacén de admisión (tope bajado tras rechazo).
  Error mío: declararlos "no ejecutables" en vez de pedir el reinicio del host.
- El propietario reinició el host (19 GiB disponibles después; la VM volvió a arrancar sola). Se relanzaron en secuencia
  el brazo de A' perdido y los tres `*_shorter` a 10G; todos terminaron. Los brazos a 10G tardaron 2 550–3 400 s frente
  a 1 500–1 900 s los demás.

## 6. Qué hereda la etapa B (no emitida; decide el propietario/Satoshi)

`winner_aux` y `winner_minimal_state` sobre `both_hold_shorter` (etiquetas both, hold_through, demostradores 2 y rev10,
tope 10G). Recomendación antes de emitir: cinco semillas para el ganador y para `both_hold`, porque el margen (+0.08)
es una décima de la dispersión; y el par puente sobre la receta v3 de Musashi (control + ganador) para medir el soporte
recuperado por separado.
