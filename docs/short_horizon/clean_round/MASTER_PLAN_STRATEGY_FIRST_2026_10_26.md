# Plan maestro, con la estrategia primero (orden del propietario, 2026-10-26)

Regla del propietario: la estrategia es cimiento; debe estar simulada, probada y optimizada **antes** de configurar
cabezales, entrenar o seleccionar. Ningún paso avanza con un defecto conocido en el anterior. Sin "pendientes": lo que
no se puede, se hace de otra forma o se da como límite medido.

## Fase 0 — Estrategia con salidas simuladas (en curso, carril F9)

| qué | cómo | salida |
| --- | --- | --- |
| verdad del motor | tablas de resultados exactas por tick y acción/celda sobre barras reales 2020–2023 (TEST cerrado) | Q(t,a), etiquetas TP/SL/TIME, neto por celda, SL-primero y restante para cerradores |
| salidas simuladas de los cabezales | funciones ruidosas de la verdad, calibradas, con calidad controlada (escalera AUC 0.50→1.00; MAE y MAE/naive para regresiones), forma realista (desbalance, cuota TIME, asimetría por lado, ruido correlacionado), **rango de salida conocido por cabeza** | tensores de salidas por nivel de calidad y escenario |
| estrategia real + motor real | mismo código de producción (veto NOP → neto esperado por celda → mejor señal del día → orden con TP/SL de la celda, tamaño por riesgo; en posición: P(SL primero) y restante → umbral por lado → reversión; tope diario; corte del viernes) | réplicas con los 15 chequeos de corrección |
| optimización de parámetros | DEAP/NSGA-II de todos los genes en el bloque de selección, por nivel de calidad y escenario; evaluación una vez en bloques de decisión; 3 semillas de ruido | genes óptimos y su estabilidad |
| **elección del fitness** | F0–F6 + el compuesto del propietario (Sharpe + profit/balance + órdenes/día, coeficientes normalizados): se optimiza con cada uno y se mide fuera de bloque cuál da mejor expectativa/Sharpe/profit y con qué estabilidad | **recomendación de fitness con evidencia** |
| barridas y robustez | Sharpe/profit/expectativa/actividad/DD/valor del cierre vs calidad; **AUC de empate** por familia de cabezas; catálogo de escenarios: tendencia/rango/alta/baja volatilidad, cambio de régimen, asimetría por lado, sólo abridores/sólo cerradores/sólo NOP, descalibración, retardo, faltantes 10/30 %, costos ×1–3, picos de spread, tope diario 1/2/4, corte 12/24 h, riesgo 0.5/1/2 %; techo del oráculo y suelo del azar en todo | curvas, tabla de cobertura escenario × calidad × semilla |
| sensibilidad de las salidas | qué pasa si una cabeza satura, es constante o invierte el signo | tabla por cabeza: rango → efecto en la estrategia |
| **puerta** | informe + solicitud de auditoría a Musashi; se corrige todo lo que encuentre | `STRATEGY_IDEAL_OUTPUT_SIMULATION_2026_10_26.md`, `MUSASHI_AUDIT_REQUEST_STRATEGY_2026_10_26.md` |

## Fase 1 — Cabezales configurados con los rangos probados

Los cabezales de C3 se fijan **a partir de la Fase 0**: codificación, rango y calibración de cada salida tal como la
estrategia los consume; TCN por grupo + núcleo TCN para todo; codificación por tipo de dato aplicada. Controles de
tubería con la señal plantada (trampa, sintético, negativos) con **el mismo modelo** — la tubería tiene que superar
el AUC de empate medido en la Fase 0 con la señal moderada, si no, se corrige la tubería, no se avanza.

## Fase 2 — Prefiltro causal (lo que ya funcionaba) y después la selección flotante

- **Prefiltro causal primero**: las etapas existentes de la metodología causal (calidad → perfiles → causal →
  relaciones → candidatos → codificación → extractores por grupo) con sus artefactos de la v5 **ya hechos**; su salida
  son los 121 candidatos. Es el filtro barato que baja el número antes de lo caro. Si el propietario quiere una
  búsqueda aún más barata, el umbral del prefiltro se endurece **con las métricas existentes** y se declara; nunca una
  lista inventada.
- **Selección flotante (SBFS) con el modelo final**: desde los candidatos del prefiltro; **cada subconjunto reentrena
  los extractores y el núcleo desde cero** (nunca se anula una entrada: al quitar una característica cambia todo el
  grupo; al vaciarse un grupo se elimina la rama y se recalcula la fusión); semillas fijas; score = expectativa neta
  por trade con IC en el bloque de selección, con actividad y cierre temprano como restricciones; pasada flotante de
  reinclusión; presupuesto declarado (BUDGET si se agota). Ventana de entrenamiento por ajuste: la de 4 años es la
  preferida; la Fase 0 entrega la tabla de costo (minutos por ajuste y ruido del score) de 4 años vs 1 año para que el
  propietario decida con números.
- Confirmación del subconjunto ganador en bloques de decisión con el protocolo anidado (v11), **una lectura**.

## Fase 3 — Entrenamiento final y evaluación

Con el subconjunto seleccionado y los cabezales de la Fase 1: entrenamiento con el ZigZag multi-resolución como
feedback experto (NOP, aptitud auxiliar, régimen; realidad del motor como objetivo), ensambles por semilla,
protocolo anidado, libro por tick, tabla de cierre; comparación con la línea SAC de Musashi sobre la misma sonda y el
mismo contrato.

## Lo que ya no se hace

Experimentos con listas reducidas inventadas; etiquetas de pivote como entrada; cabeza SL constante; anular entradas
en lugar de reentrenar; elegir nada sobre el bloque de decisión; avanzar con un defecto conocido.
