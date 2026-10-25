# Auditoría del cierre temprano (Satoshi, 2026-10-25)

Para el propietario y Musashi. Regla de negocio del propietario: el cierre temprano es **indispensable e inomitible**;
cuando un modelo detecta un patrón aprendido que indica que el SL llegará antes que el TP, debe cerrar. En la línea SAC
se encontró que las políticas nunca cierran voluntariamente. Esta página fija el método y la regla **antes** de leer
ningún número; los resultados se añaden debajo, generados por la herramienta.

## Regla pre-registrada (también en `CLEAN_ROUND_SPEC.json` v7 → `early_close_rule`)

- Una estrategia seleccionable **debe cerrar temprano**: si no cierra ninguna posición por decisión propia en el bloque
  de calibración es `DEGENERATE_NO_EARLY_CLOSE` y **no es seleccionable** (se archiva igual).
- El **valor del cierre temprano** es una métrica obligatoria de todo brazo.
- Un cerrador cuyo intervalo de confianza del 95 % del valor del cierre temprano queda **entero por debajo de 0** es
  `HARMFUL_EARLY_CLOSE`: recorta ganadores. Es un defecto a corregir, **nunca** una razón para quitar el cierre temprano.
- Aplicación: el contrato `ACTION_CONTRACT_v1_1.json` (`strategy.require_early_close`) para la **siguiente** etapa; la
  etapa WF_V1 en curso conserva el contrato v1 y su código instalado sin cambios. Su auditoría se reporta igual.

## Método (`tools/expert_guided_early_close_audit.py`)

1. Se reproduce exactamente la corrida de cada semilla y del ensamble (Sharpe, neto, DD, conteos de salidas por motivo;
   la herramienta se detiene si no coincide).
2. **Por cada cierre del agente** (cierre o reversión): contrafactual con el propio motor (mismas unidades, SL, TP y
   costos de entrada): mantener la posición desde esa barra hasta TP, SL o cierre forzoso. Δ = neto real − neto
   contrafactual. `SAVED` si Δ > 0, `CUT_WINNER` si Δ < 0, `NEUTRAL` si |Δ| < 0.05σ. Valor del cierre temprano = ΣΔ, en
   USD y en σ, con IC 95 % por bootstrap sobre trades; también por lado.
3. **Por cada barra posicionada observable**: etiqueta `SL_FIRST` (mantener termina en el SL, o negativo en el cierre
   forzoso; orden de protección del motor) y `CLOSE_BETTER` (cerrar en la apertura siguiente neta más que mantener);
   AUC de la probabilidad de salida del cerrador frente a ambas; precisión y recall del disparo frente a `SL_FIRST`.
4. **Política sin cierre temprano**: la misma política congelada con todo cierre voluntario sustituido por HOLD,
   reproducida en el motor (ejecutable); y la suma por trade con cada cierre del agente reemplazado por su
   contrafactual (se solapan: no ejecutable, sólo referencia).
5. **Balance de dirección**: entradas largas y cortas, neto y Sharpe por lado, fracción de barras posicionadas por lado.

Bloques: los años de validación del walk-forward (INNER_B, puntuados una vez tras congelar) para las unidades WF; el
bloque interno (INNER_A) para `A_zz_hold` y `A_both_hold` (sesgado por D1, sólo referencia). TEST cerrado.

## Resultados: línea base (generados; tabla completa en `evidence/early_close/EARLY_CLOSE_AUDIT_BASELINE.md`)

Las 8 unidades (los 6 walk-forward en su año de validación y A_zz_hold / A_both_hold en INNER_A) reproducen su corrida
exactamente, por semilla y en el ensamble. Ensamble por unidad:

| unidad | bloque | cierre temprano | SAVED / CUT / NEUTRAL % | valor USD [IC 95 %] | Sharpe real / sin cierre temprano | DD % real / sin cierre | AUC cerrador (SL antes) | precisión / tasa base / recall | largos / cortos |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| WF_wf2022_zz_hold | 2022 | 60 % | 41 / 57 / 1 | −838 [−6290, 4792] | 0.46 / 0.40 | 31.2 / 21.8 | 0.61 | 0.45 / 0.36 / 0.06 | 50 / 76 |
| WF_wf2022_both_hold | 2022 | 52 % | 56 / 44 / 0 | −905 [−5080, 3344] | 0.60 / 0.68 | 29.1 / 31.9 | 0.64 | 0.56 / 0.37 / 0.10 | 75 / 49 |
| WF_wf2022_both_hold_shorter | 2022 | 49 % | 50 / 47 / 3 | −2783 [−7081, 1435] | 0.55 / 0.80 | 13.5 / 25.0 | 0.51 | 0.54 / 0.48 / 0.08 | 89 / 58 |
| WF_wf2023_zz_hold | 2023 | 77 % | 40 / 57 / 2 | −2817 [−6071, 548] | −1.51 / −0.32 | 31.1 / 28.4 | 0.56 | 0.49 / 0.41 / 0.10 | 50 / 54 |
| WF_wf2023_both_hold | 2023 | 56 % | 63 / 37 / 0 | 105 [−2461, 2576] | −0.71 / −0.12 | 23.2 / 26.9 | 0.65 | 0.62 / 0.42 / 0.11 | 73 / 34 |
| WF_wf2023_both_hold_shorter | 2023 | 57 % | 47 / 51 / 3 | −1638 [−4309, 931] | −0.35 / 0.65 | 25.4 / 29.9 | 0.60 | 0.52 / 0.46 / 0.11 | 58 / 74 |
| A_zz_hold (sesgo D1) | INNER_A | 79 % | 41 / 55 / 4 | −370 [−4052, 3402] | 0.96 / −0.17 | 6.9 / 25.5 | 0.70 | 0.44 / 0.19 / 0.37 | 43 / 47 |
| A_both_hold (sesgo D1) | INNER_A | 71 % | 71 / 29 / 0 | 2932 [649, 5176] | 0.19 / −3.00 | 11.8 / 29.8 | 0.69 | 0.73 / 0.53 / 0.24 | 37 / 36 |

Lectura (sin adornos):

- **No es el problema de la línea SAC:** los agentes cierran temprano (49–77 % de las salidas en los años de validación)
  y abren en ambos lados (largos/cortos 50/76 … 89/58). Una semilla (de 5) de `WF_wf2022_both_hold` no cerró ninguna
  posición en 2022: `DEGENERATE_NO_EARLY_CLOSE` en esa semilla; otra de `WF_wf2023_zz_hold` es `HARMFUL_EARLY_CLOSE`.
- **El cerrador detecta algo, pero poco y tarde:** AUC 0.51–0.65 contra "SL antes que TP" en los años de validación, con
  precisión por encima de la tasa base (p. ej. 0.62 frente a 0.42) y **recall de sólo 6–11 %**: casi todas las posiciones
  que terminarían en el SL no se cierran antes.
- **El valor del cierre temprano no es positivo fuera de muestra:** negativo en 5 de 6 ensambles walk-forward y ≈ 0 en
  el sexto, con IC que incluyen el cero en todos (ningún ensamble es `HARMFUL_EARLY_CLOSE`, ninguno es beneficioso).
  Entre 37 % y 57 % de los cierres recortan un ganador. Quitar el cierre temprano sube el Sharpe en 5 de 6 ensambles,
  pero sube el drawdown en 4 de 6: el cierre actual reduce riesgo sin ganar dinero.
- En INNER_A (bloque sesgado por D1, sólo referencia) el cierre sí protege: sin él el DD pasa de 7–12 % a 25–30 %.
- Conclusión para el diseño: el cierre temprano se mantiene (regla de negocio) y se corrige el **cerrador**: su recall
  y su calibración. En v1 el umbral de cierre ya es un parámetro de la estrategia; la etapa siguiente exige además
  `require_early_close` (`ACTION_CONTRACT_v1_1.json`). La auditoría de WF_V1 corre sola al terminar la cadena v6
  (`report/EARLY_CLOSE_AUDIT_V1.md`).
