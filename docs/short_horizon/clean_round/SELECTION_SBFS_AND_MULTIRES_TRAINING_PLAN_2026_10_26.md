# Plan: selección flotante secuencial con el modelo final + entrenamiento con el ZigZag multi-resolución (2026-10-26)

Para el propietario. Corrige mi error: corté a 19 columnas por métricas existentes y **no hice la selección flotante
secuencial (SBFS)** que usted ordenó. Este plan la hace con el **modelo final** y define, sin ambigüedad, cómo se
entrena con el ZigZag multi-resolución. Nada se ejecuta hasta que F1 (contrato C3) regrese; el evaluador se construye
ya (lane F4).

## 1. El modelo final (verificado en el código de Musashi, e18d00d4)

| parte | lo que hay en su código | lo que usaré |
| --- | --- | --- |
| rama por grupo | **una** capa causal (kernel 3 para series, 1×1 para puntuales), unidades del grupo según la receta, SiLU | igual (paridad exacta ya probada, @62f6530); opción declarada `branch_layers=2` si usted la quiere |
| fusión | 1×1 a 8 unidades | igual |
| núcleo | **dos bloques residuales causales**, cada uno con dos convs kernel 3, dilataciones 1 y 2 (RF 13 = la ventana), SiLU | igual |
| lectura | aplanar la ventana | igual (`readout=flatten`) |
| cabezas | categóricas separadas (PPO) | mis cinco especialistas (abrir L/S, NOP, cerrar L/S) + TP/SL por celda, contrato C3 |

Su núcleo NO es "dos capas simples": son dos bloques residuales de dos convs cada uno. Es lo que da el campo receptivo
completo y el gradiente en una red tan pequeña; por eso la adopto tal cual.

## 2. SBFS con el modelo final (lo que usted comandó)

Motor: `feature_selector` de Musashi, metodología `causal_selection` completa (calidad → perfiles → causal →
relaciones → candidatos → codificación → extractores por grupo → **model_wrapper** → confirmación), selector
`floating_backward`. **Lo único distinto es el evaluador**: `expert_guided_model`, registrado en el mismo espacio
de entrada `feature_selector.evaluators`, que entrena **mis especialistas reales** bajo el contrato C3 con el tronco
TCN de §1.

Reglas del recorrido (idénticas a las de Musashi, `rl_subset.py` / `80_NATIVE_SUBSET_EXECUTION.md`):
- Punto de partida: el candidato = unión de las 121 de la receta v2 (todo lo que sobrevivió a las etapas previas),
  **no** las 19; las 19 son una cota de referencia que el SBFS debe superar o confirmar.
- Eliminación hacia atrás: quitar cada característica, reentrenar **desde cero** (semillas fijas, soporte pareado,
  escalador reajustado sólo en TRAIN del fold), puntuar; quitar la de menor pérdida.
- **Grupos y extractores**: la característica se quita de su grupo; cuando sale la última de un grupo, **se elimina
  la rama entera** y se recalcula el ancho de la fusión; los bloques coherentes (seno/coseno, p0/p1/p2) salen juntos;
  la codificación y el extractor de cada grupo quedan congelados dentro de una búsqueda (cambiar arquitectura = otra
  búsqueda declarada).
- **Pasada flotante**: tras cada eliminación, intentar re-incluir las eliminadas; se reincluye sólo si mejora el mejor
  resultado de esa cardinalidad. Estados visitados, empates deterministas, presupuesto duro (128 subconjuntos × 3
  semillas en la primera pasada; si se agota, BUDGET, nunca "selección completa").
- **Puntuación** (lo que NO hacía bien antes): expectativa neta por trade en σ con IC (bootstrap por días) en el
  bloque de **selección interno** del fold (nunca el año de decisión), con validez ≥ 2 trades/semana y cierre
  temprano > 0 como restricciones; se registran parámetros, órdenes, lados, aciertos, cierres propios, costos,
  profit, Sharpe, DD y brecha TRAIN–interno por semilla. Un score indefinido no es 0.
- **Objetivo**: el resultado del motor (C3), nunca "la barra es pivote" (defecto A1-2 que coló las 121).
- Por fold anidado (F2/F7); la lista final = intersección estable entre folds, con la frecuencia de cada columna.
- Confirmación: el subconjunto ganador contra las 19 y contra las 121, en los bloques de decisión, con la puerta de
  control sintético antes.

## 3. Cómo se entrena con el ZigZag multi-resolución (definición operativa)

El ZigZag ya **no dicta la etiqueta de entrada** (la auditoría probó que esa etiqueta enseña a perder). Entra en
tres lugares, cada uno con su métrica:

| papel | cómo entra | qué aporta | métrica |
| --- | --- | --- | --- |
| **NOP especialista** | etiqueta NOP duro (ninguna resolución de la escalera {óptimo, rev20, rev15, rev10, rev8, rev5} encuentra trade) y NOP graduado (f* < θ_f, MAE/tramo > θ_mae, tramo/σ < θ_vol), con peso | el agente aprende **cuándo no operar**, lo que usted pidió como lo más importante | AUC/Brier del NOP; valor del veto en el bloque de selección (se conserva sólo si sube la cota inferior) |
| **aptitud auxiliar** | f_open (ganancia restante hasta el cierre óptimo / tramo óptimo, negativa contra el óptimo) y f_close como cabezas de regresión auxiliares en abridores y cerradores | gradiente extra "cuánto queda", ordenado (regresión), sin cambiar la etiqueta de clase | Spearman frente a la excursión realizada |
| **perfil de régimen** | por barra y escala: tamaño del tramo en σ, excursión adversa, volatilidad del tramo, duración | descriptor de régimen (C1) para la invariancia entre regímenes/años (C3) y para estratificar la evaluación | estabilidad de signo entre años |

Y lo que **sí** dicta la etiqueta de entrada y de cierre: la **tabla de resultados exacta del motor** (TP antes que SL
/ SL antes que TP / TIME, neto con costo, por celda TP/SL) — C2/C3. El multi-ZigZag es la "preferencia" del experto;
la tabla de resultados es la realidad; el modelo aprende la realidad y usa la preferencia como guía (NOP, auxiliar,
régimen). Eso es lo que llamamos RLHF en este proyecto, dicho con precisión.

Umbrales (θ_f, θ_mae, θ_vol, θ_close): los del DEAP ya medidos como punto de partida, re-optimizados en el bloque de
selección interno de cada fold con la regla de actividad con IC; nunca en el año de decisión.

## 4. Orden y puertas

1. F1 (C3, dos variantes: conv1d y TCN) → control sintético debe superar a C2.
2. F4: evaluador `expert_guided_model` + driver SBFS (construido ya; corre cuando C3 exista).
3. F7: stores por fold anidado (corriendo).
4. Puerta: control sintético a 5 semillas con el tronco TCN.
5. SBFS por fold en el 4090 (128 × 3 semillas por fold, primera pasada; BUDGET si se agota).
6. Confirmación en bloques de decisión; selección final y receta nueva (identidad nueva), comunicada a Musashi.
