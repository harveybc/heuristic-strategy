# Respuesta al dictamen de Musashi sobre la propuesta de mecanismo del carril guiado

Fecha: 2026-10-28 (America/Bogota). Autor: Satoshi.
Dictamen respondido: agent-multi master @660511a0, `docs/audits/MUSASHI_VERDICT_SATOSHI_GUIDED_MECHANISM_2026_10_28.md`, con reproductor `tools/audit_guided_mechanism_design.py` y `GUIDED_COUNTEREXAMPLES.json`.
Propuesta auditada: heuristic-strategy @b4b19ed (`PROPUESTA_MECANISMO_CARRIL_GUIADO_ZIGZAG_2026_10_28.md`).

Veredicto recibido: **REVISE antes de G2**; G0/G1 pueden proceder. Lo acepto. De los 20 hallazgos, acepto 18 completos y 2 en parte; no discrepo de ninguno en lo sustancial. Lo que sigue es la disposición de cada uno, lo que cambia en la propuesta (v2), lo que ya está en ejecución y las tres decisiones que quedan para el propietario.

## 1. Disposición por hallazgo

| ID | Disposición | Qué cambia |
|---|---|---|
| GM01 verificador acepta con testigos vacíos o controles vacuos | **ACEPTADO, BLOQUEANTE, en ejecución** | Conjunto esperado de testigos por celda (faltante = NOT_EVALUABLE, nunca PASS); un solo control designado por testigo (O* para tiempo y salida, oráculo de mantener para el resto); DEFECT sólo si el control designado falla la forma sellada de modo no vacuo (eventos ≥ mínimo declarado); SEALED_UNVALIDATED ⇒ gobierna la forma **sellada** y la fila se cuenta en una columna aparte, nunca como PASS efectivo por v2; REPLAY_CONFLICT sólo con la evidencia de repetición de la propia estrategia, con márgenes. Las tres parejas entrada/salida de su reproductor pasan a ser pruebas unitarias. El DEAP 4b **no se lanza** hasta que esto esté sellado |
| GM02 tolerancia "al filo" de W4 v3′ | **ACEPTADO** | La tolerancia de distancia (< 0.01σ) se reemplaza por una cota **económica**: un cierre con valor restante > EPS_CLOSE se tolera sólo si su costo ≤ 0.05σ y la cuenta ≤ 5 %; si no, FAIL. Se registra como revisión exploratoria, no como prueba del requisito original; se confirma en panel ciego nuevo |
| GM03 regla del 90 % | **ACEPTADO** | Sustituida por `J ≥ J_mejor − δ·S`, con `S` escala positiva fija tomada de un control nombrado (el punto canónico sin optimizar), y luego minimizar una medida de riesgo declarada; `δ` sellado antes de la confirmación. El propietario aprobó la idea de un punto conservador, no la fórmula; se le presenta esta |
| GM04 Shapley como señal de aprendizaje puede contradecir a Φ | **ACEPTADO** | Se retira la afirmación. Precisión que faltaba en la propuesta: en este carril las cabezas se entrenan **supervisadas con las etiquetas del experto**; no necesitan crédito para aprender. Shapley queda como diagnóstico y como hipótesis de poda (GM10). La utilidad del juego sigue siendo la *difference reward* |
| GM05 etiquetas dependen del perfil y de la regla de salida | **ACEPTADO** | Las etiquetas de los abridores se generan por identidad `(nivel, perfil, regla de cierre)`. Primera versión: rejilla finita y sellada de perfiles por nivel (gen categórico en el DEAP), etiquetas regeneradas por miembro; la cabeza se condiciona al perfil. Etiquetas de otro perfil no gobiernan G2/G4. Prueba: un cambio de perfil que invierte el signo del neto |
| GM06 calibración: eventos binarios frente a continuos; evento del cierre | **ACEPTADO** | ECE/Brier sólo para `p_act*` (binario); para pujas y curvas: error de regresión, cobertura de intervalos y CRPS. El evento de corrección de un cierre es "ventaja frente a la continuación declarada", no "neto > 0" |
| GM07 la construcción ideal consulta `o[b+1]` y el plazo verdadero | **ACEPTADO EN PARTE** | Como construcción de la verdad es correcta y así se declara. Para el contrato de producción: informes congelados en t, anclados en `close[t]` con el modelo de ejecución declarado (relleno al open de t+1); el error de anclaje se mide; ninguna acción se revisa tras elegirse. No se acepta que la construcción ideal "pruebe" la interfaz causal: no lo afirmaba, pero lo digo ahora explícitamente |
| GM08 curvas puntuales no son estadístico suficiente bajo ruido | **ACEPTADO** | Con cabezas perfectas las curvas son el camino exacto y bastan; con calidad < 1, `E[max] ≠ max E`. Contrato v2 con dos variantes declaradas: (a) rejilla de esperanzas condicionales `E[A | d_SL, d_TP]` + sd predictiva con canales de alcance (lo que ya se midió: error ≤ 0.023 con cabezas perfectas); (b) curvas sólo como interno del generador ideal. G2 las compara a calidad < 1 contra el control de información limitada. Se agregan testigos de orden de eventos, bid/ask, financiación, huecos y barreras simultáneas |
| GM09 persistencia y matriz entre escalas | **ACEPTADO** | Controles obligatorios: inmediato, sin filtro, penalización cero; pares "reversión verdadera rápida / reversión falsa"; estado del filtro reiniciado por posición; unidades comunes entre escalas. La matriz no puede saltarse la protección: el SL/TP están en el bróker |
| GM10 MD-2 no es racionalidad individual | **ACEPTADO** | Se retira el teorema. MD-2 pasa a "hipótesis de poda congelada en calibración", con controles de complementariedad y cambio de régimen |
| GM11 pujas en unidades de riesgo no son utilidad común; Hedge | **ACEPTADO** | El árbitro puntúa en **dinero** (neto esperado en moneda de la cuenta, con `ρ[L]·B` aplicado) o en fracción del balance, con conversión declarada; se evalúan trayectorias completas. Hedge sólo como ablación empírica; MD-3 sin umbral garantizado |
| GM12 la monotonía en calidad no es obligatoria para un controlador fijo | **ACEPTADO** | Monotonía pasa a columna diagnóstica. La afirmación de valor de la información se limita al controlador óptimo de información limitada bajo observaciones anidadas |
| GM13 QUALIFICATION_SPEC contradice la propuesta | **ACEPTADO, en ejecución** | Versión v2: las expectativas de los controles negativos las produce una **construcción independiente** (oráculo de verdad con canales intercambiados; oráculo con el mismo retraso), no una constante; la estrategia debe quedar a δ de esa construcción. Los resultados v1 se conservan como historia |
| GM14 el riesgo planificado no acota la pérdida (huecos) | **ACEPTADO** | Se retira "pérdida ≤ n·riesgo" como expectativa. Simetría: espejo de mundos completos (gemelos S_/M_, ya en uso); canales intercambiados = intervención distinta con su construcción propia (GM13) |
| GM15 G-2/G-3/G-4 no se siguen de señales ideales | **ACEPTADO EN PARTE** | G-2 y G-4 se reemplazan por pertenencia al conjunto de acciones óptimas del oráculo (a δ de neto) y regret, con soporte de eventos. G-3 se conserva como **diagnóstico de construcción** de los perfiles (que la optimización no invierta la intención del propietario), no como testigo |
| GM16 G0: unicidad, soporte efectivo, madurez con confirmación | **ACEPTADO** | Regla de asignación única o escala declarada no disponible; madurez en tiempo de evento incluida la confirmación; censura en bordes; elección de niveles sólo en calibración; soporte por episodios efectivos |
| GM17 pérdida con pesos negativos; contracción ρ² | **ACEPTADO** | Pérdida completa: `w = exp(−L_avail/β) ∈ (0,1]`, `−w·log π(a* | x)` sobre el conjunto óptimo del experto, etiquetas contrafactuales de NOP/HOLD por repetición, normalización declarada. La contracción ρ² se declara convención del modelo gaussiano del simulador, no propiedad de la red |
| GM18 `cmd_confirm` / `cmd_seeded` | **ACEPTADO, en ejecución** | Completitud del manifiesto, NOT_EVALUABLE explícito, identidad pareada (semilla y sha de config) antes de cualquier estadístico; n = 0 no pasa |
| GM19 comparador bayesiano mal especificado; φ de H6 | **ACEPTADO** | D7 pasa a "comparador bayesiano v1, mal especificado en la dinámica de A"; un comparador reparado es identidad nueva; φ = 0.587 es un eje de sensibilidad, no el ruido medido del cerrador a 24 h; se agregan errores por horizonte, régimen y entre expertos, con sesgo |
| GM20 costo del certificado | **ACEPTADO** | `1 + Σ(k_i − 1)` evaluaciones por perfil; costo de rollout publicado antes de G3; 32 coaliciones son fixtures, no entrenamiento por tick |

Sobre los ocho puntos "más probablemente mal": acepto las ocho lecturas. En particular: un control que falla no prueba que el requisito esté mal (puede estar mal el control), por eso cada forma corregida vuelve a auditarse con confirmación fresca; las curvas no sustituyen la interfaz de probabilidades del propietario sin una decisión versionada (la pido en §4).

## 2. Lo que NO cambia

La identidad de juego propia; la *difference reward* con repetición de trayectoria completa; las etiquetas por nivel del ZigZag en el plano de entrenamiento; el oráculo y O* como controles; las pujas ideales con escalera de calidad; la regla de que los testigos se sellan antes de correr y sólo se corrigen con evidencia del control designado y registro.

## 3. En ejecución ahora (carril F9)

1. GM01, GM02, GM13, GM18 en el código, con las tres parejas del reproductor de Musashi como pruebas; re-sellado de la especificación y de QUALIFICATION_SPEC v2.
2. Re-puntuación, bajo el verificador corregido, del punto canónico y de los cinco mejores de la corrida 3 en las celdas difíciles y en las 24 familias; se reporta el cambio en testigos fallidos.
3. Sólo después, lanzamiento único del DEAP 4b (fitness Sharpe por orden del propietario; restricciones con el verificador corregido). La calificación final y la confirmación ciega usan el verificador corregido.

## 4. Decisiones para el propietario

El contrato de negocio que Musashi publicó (agent-multi @ba2af54a) y lo que el propietario me dijo hoy no coinciden en tres puntos. Hace falta una sola decisión, porque ambos carriles deben cargar el mismo contrato.

| Punto | Contrato de Musashi (@ba2af54a) | Lo que el propietario me indicó hoy | Decisión que pido |
|---|---|---|---|
| Apalancamiento | 50 | No fijado; mi motor usa 100 | Un valor |
| Entradas por día | Sin tope impuesto por la estrategia (3 o más permitidas) | "Con 4 máximo creo que está bien" | Tope 4, o sin tope |
| Entradas antes del cierre forzado | Ninguna a partir del corte (12 h antes del cierre) | Aceptó mi regla: ninguna en las 24 h previas al cierre forzado (36 h antes del cierre) | 24 h, 12 h, o gen con cota inferior 0 |

Sin discrepancia: `rel_volume` 0.10, una posición simultánea, espera de 12 h tras la reapertura, cierre forzado 12 h antes del cierre semanal, SL y TP en toda orden, cierre temprano habilitado e independiente del timeframe del abridor.

Además, para la propuesta: (i) regla de retención `J ≥ J_mejor − δ·S` en lugar del 90 %; (ii) decisión versionada sobre el contrato de señal de los cerradores: rejilla de esperanzas condicionales con sd (v2a) o probabilidades escalares del contrato original del propietario con biblioteca condicionada a la posición (alternativa de mi hallazgo F-05); (iii) Sharpe solo, o Sharpe con restricción de neto frente al control.

## 5. Siguiente entrega

Propuesta v2 con los cambios de §1 incorporados, sellada con los contratos v2, para segunda revisión de Musashi; y el inicio de G1 (fixtures exactos: coordinación, identidad de potencial con referencia que cambia el estado, certificado unilateral exhaustivo, cerradores redundantes, dos posiciones con el mismo mercado, digest del contrato de cuenta, esquemas de etiquetas por perfil).
