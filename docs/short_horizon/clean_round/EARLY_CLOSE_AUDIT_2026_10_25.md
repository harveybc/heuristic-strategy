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

## Resultados

(Se añaden desde `report/EARLY_CLOSE_AUDIT_BASELINE.md` y, al terminar la cadena v6, `report/EARLY_CLOSE_AUDIT_V1.md`.)
