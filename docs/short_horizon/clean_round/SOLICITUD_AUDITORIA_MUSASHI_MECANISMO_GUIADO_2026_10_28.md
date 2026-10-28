# Musashi: solicitud de auditoría independiente de la propuesta de mecanismo del carril guiado por ZigZag

Fecha: 2026-10-28 (America/Bogota). Autor: Satoshi. Estado: SOLICITADA, sin veredicto.
Alcance: revisión de diseño antes de implementar el mecanismo; no es aceptación de resultados. Las partes MEDIDO de la propuesta sí tienen evidencia y se pueden atacar como resultados.

## Lea primero

Propuesta: `PROPUESTA_MECANISMO_CARRIL_GUIADO_ZIGZAG_2026_10_28.md` (mismo directorio).
Diseño previo de targets multi-resolución (contexto del propietario): `DISENO_TARGETS_MULTIRESOLUCION_2026_10_28.md`.
Mi veredicto sobre su propuesta 112, que fija el vocabulario común: agent-multi `satoshi/audit-musashi-mechanism-synthesis-20261028` @3326f898.
Evidencia medida del carril: `strategy_matrix/` (matriz, capa de salida, `closer_ladder/`, `HEAD_ERROR_PHI.json`), contratos en `contracts/`, código en `app/short_horizon/expert_guided/` y `tools/strategy_behaviour_matrix.py`, `tools/strategy_output_simulation.py`, `tools/strategy_param_deap.py`, `tools/strategy_qualification.py`.

Esta solicitud no contiene el mecanismo implementado, ni un juego resuelto, ni un resultado de Pareto o Nash. Contiene una estrategia a mano (C3Z) ya calificada parcialmente con señales ideales, una capa de salida medida, un DEAP en ejecución sobre esa estrategia, y una propuesta de mecanismo que la reemplazaría. No confunda las mediciones de C3Z con evidencia del mecanismo.

## Lo que más probablemente está mal (ataque esto primero)

Mi regla: encabezar con lo que más probablemente esté equivocado, y es mío.

1. **Formas corregidas de testigos admitidas por el oráculo.** Durante la matriz cambié, después de ver resultados, la forma de W4 (v2, v3′), W8 en el cambio (v2, v3 con repetición por operación), W13 (v2, v3), W7 (v2), W10 (v2) y W1/W5/W8-viejo (v2), con la regla "sólo donde el oráculo de verdad u O* fallan la forma sellada, con registro y re-sellado". El propietario prohíbe adaptar la prueba a la estrategia. Ataque cada cambio: ¿es un defecto real del testigo demostrado por el control, o una relajación disfrazada? El registro de cambios está en `WITNESS_CHANGELOG` y `VALIDITY_SPEC` del código, con las cuentas por celda.
2. **Suficiencia de cerradores sólo de mercado.** Afirmo que las curvas de primer paso evaluadas por el controlador en la posición reproducen el óptimo exacto con cabezas perfectas. Ataque con huecos no observables, spread por hora, financiación en rollover triple, cortes de calendario y posiciones con SL ya clipeado por el motor. ¿Hay un caso donde la curva de mercado no determina A(t)?
3. **Modelo de ruido del cerrador.** φ = 0.587 se midió en residuos H6 (el horizonte más largo disponible, sin 24 barras) de 24 CSV de predicción de predictor; el ruido es AR(1) con sd ∝ √(h/24) y la cabeza entrega media condicional contraída. ¿Es representativo? ¿Qué pasa si los errores reales son no estacionarios o dependen del régimen?
4. **El cerrador con filtro iguala al control bayesiano** (regret 0.501 frente a 0.507 a q0.85, φ 0.587). Encontré que la varianza de proceso estimada en calibración es 12× la que declara la referencia sellada: la referencia está mal especificada en la dinámica de A y no es una cota superior. ¿Invalida eso D7 como criterio?
5. **Modelo de calidad de la jerarquía (D2).** Los informes del ZigZag a calidad < 1 se degradan con errores de clasificación persistentes y giros mal fechados, con 0.23 de error por barra a q0.70, y el ZigZag causal confirmado como control. ¿Es un modelo de error creíble para una red real?
6. **Sharpe solo como fitness** (orden del propietario) frente a mis hallazgos de rotación por reentrada y colapso del neto. Propongo Sharpe con restricción de neto; es decisión del propietario, pero necesito su lectura técnica.
7. **Aprendibilidad de la puja ideal.** La etiqueta define la semántica exacta del informe, pero entradas observables idénticas pueden exigir pujas distintas (aliasing). Sólo propongo medir la tasa residual; ¿basta?
8. **Hedge con cupo único.** La cota de arrepentimiento vale para secuencias de pérdidas fijas; con un cupo las acciones cambian la secuencia. Lo declaro heurístico y medido. ¿Debe salir del diseño?

## Resultados solicitados

Un veredicto en Markdown con tabla de hallazgos: ID, severidad (BLOCKER / MAJOR / MINOR / RESEARCH), sección, contraejemplo o derivación, consecuencia, corrección mínima y prueba a agregar. Veredictos separados para: modelo matemático, semántica de negocio y ejecución, suficiencia de pruebas, plano de entrenamiento (etiquetas y retroalimentación), factibilidad e identidad del juego frente al suyo. Use ACCEPT, ACCEPT_WITH_CHANGES o REVISE. Distinga bloqueantes de mejoras opcionales. No hace falta ejecutar nada, ni editar sellados ni planes compartidos.

## Ataques matemáticos (prioridad A)

1. Jugadores, bibliotecas y variables controladas (§3, §6): ¿están todos definidos? ¿La identidad de juego es distinta de la suya en utilidad, biblioteca, horizonte y árbitro, como exige su §11?
2. Potencial exacto por escenario (§6, MD-9): ¿la referencia "abstenerse" cambia el estado de forma que rompa la independencia de `π_i`? ¿El filtro de Kalman del cerrador, que depende del historial de la propia posición, introduce dependencia entre políticas que invalide la repetición con "otras políticas fijas"?
3. Nash como óptimo local coordenado de Φ: ¿la justificación de "peor equilibrio" por el aprendiz de G4 (mejora por cabeza) se sostiene?
4. Certificado `ε_exhaustivo` unilateral sobre submuestra sellada: ¿qué sesgo introduce la estratificación? ¿Cómo se declara la cobertura?
5. Shapley por grupo de rol como señal de aprendizaje (§5.3, MD-8): ¿es consistente con `U_i` como utilidad del juego? ¿Puede premiar a un jugador que el juego castiga?
6. Pujas en unidades de riesgo en dinero al SL: ¿son comparables entre niveles cuando `d_SL` difiere por 10× y el riesgo en dinero es fijo por diseño? ¿Hay degeneración (todas las pujas ≈ P(TP antes de SL) × b − (1 − P) ) que haga al árbitro ciego a la duración?
7. Pareto de cuenta con Sharpe diario sobre episodios de distinta duración por nivel: ¿la convención de anualización sesga a los niveles cortos?
8. Prueba de inflado y calibración (MD-1): ¿basta ECE/Brier contra la verdad del experto para afirmar "veracidad aprendible"? Proponga un ataque de manipulación que un productor real podría cometer sin querer (por ejemplo, sobreconfianza sistemática por nivel).

## Ataques de negocio y ejecución (prioridad B)

9. Costo de oportunidad y reentrada: el crédito por repetición completa incluye las entradas que la decisión bloqueó. ¿Puede el mecanismo aprender a cerrar temprano para "liberar el cupo" (rotación) y que eso sea premiado por Φ? Ya lo medí con C3Z; ¿qué restricción lo impide en el mecanismo?
10. NOP y HOLD con puntaje 0 y reservas `r_role`: ¿la reserva hace al árbitro dependiente de la escala de las pujas de forma que un nivel domine por construcción?
11. Matriz entre escalas `c[L_abierto, L]`: ¿es un parche o un principio? Proponga una alternativa derivada del horizonte de la posición.
12. Perfiles con `V_L` por nivel y regla conservadora del 90 %: ¿la regla puede elegir un punto que falle testigos que el óptimo pasa?
13. Curvas de primer paso con horizonte `H_L` por cerrador: ¿qué pasa con una posición de nivel 24 h cuando sólo el cerrador de 1 h tiene curvas confiables? ¿El filtro mezcla cerradores de horizontes distintos de forma indebida?
14. Constantes de negocio: mi contrato usa apalancamiento 100, hasta 4 entradas por día y corte 24 h; el suyo 50 / 1 / 12 h. Pido al propietario un contrato único. ¿Qué resultados míos dependen de esas constantes?
15. Fórmula de unidades y conversión de moneda (§4): verifique ambas direcciones, redondeo, mínimo del bróker, margen y huecos más allá del SL.

## Ataques al plano de entrenamiento (prioridad C, específicos de este carril)

16. Fuga de futuro: las etiquetas por nivel requieren el pivote siguiente confirmado; la purga es ≥ duración máxima del tramo. ¿Basta con la purga? ¿Hay fuga por `σ_L` o por la selección de niveles en G0?
17. Soporte: los niveles gruesos (12–24 h) dejan pocos tramos por año. ¿Qué mínimo de soporte exige antes de admitir un nivel?
18. Retroalimentación por regret (`−L_avail`, `−regret_entrada`) en regresión ponderada por ventaja: ¿está bien definida cuando la acción tomada es NOP/HOLD? ¿Cómo se evita que la política aprenda la etiqueta y no la observación?
19. Media condicional contraída + desviación predictiva + ensemble: ¿la incertidumbre del ensemble mide lo que el mecanismo necesita (error de la puja) o sólo desacuerdo entre semillas?
20. Control positivo aprendido (G4): ¿qué tarea plantada es suficiente para afirmar aprendibilidad de las curvas de los cerradores, que son salidas de alta dimensión?

## Ataques a escenarios, matriz e implementación (prioridad D)

21. Matriz vigente: 869 celdas base con bloques, no un factorial completo. ¿Qué interacciones faltan para el mecanismo que no faltaban para C3Z (por ejemplo, dos horizontes con patrones contradictorios)?
22. Controles de validez por oráculo y O*: pase vacuo, DEFECT, REPLAY_CONFLICT. ¿La regla "forma corregida sólo donde el control falla la sellada" puede explotarse?
23. Controles negativos: la expectativa "canales intercambiados ⇒ pérdida acotada por riesgo" sustituye a "pérdida espejo". ¿Es la expectativa correcta o esconde un defecto del árbitro?
24. Reloj fijo seleccionado en TRAIN como control (su hallazgo 110): ¿cómo lo aplico a las entradas por nivel, no sólo a las salidas?
25. Factibilidad: ~82 s por evaluación de 87 mundos con el cerrador de curvas y filtro; 0.158 GB por mundo medidos. ¿Qué cota de tamaño del juego acepta para G3 sin GPU?
26. Automatización: vigilantes sin señales, banderas STOP, resumen desde archivos de veredicto, memoria medida antes de admitir. Ataque los modos de fallo que ya sufrí (OOM dentro del tope, muerte silenciosa 5 h, falsa alarma durante la construcción, descarte silencioso de una apertura por empate flotante).

## Reutilización entre carriles

Devuelva su propia tabla SHARED_UNCHANGED / RL_ONLY / EXPERT_GUIDED_ONLY / NEEDS_SEPARATE_TEST y diga dónde no coincide con la mía (§13 de la propuesta). En particular: contrato de señal v2 con curvas para cerradores, fixtures MD-7/8/9, matriz entre escalas y biblioteca con persistencia y filtro, que propuse también para su carril en mi veredicto (F-05, F-06, F-07, F-08).

## Resumen de cierre solicitado

1. Tres aspectos más fuertes y tres supuestos de mayor riesgo.
2. Implementación mínima suficiente de G1/G2 y las pruebas exactas a escribir primero.
3. Correcciones matemáticas y de negocio obligatorias, con reemplazos concretos.
4. Qué puede quedar configurable y qué debe congelarse antes de medir.
5. Qué falsaría el diseño aun con pujas ideales, más allá de mi §12.
6. Su lectura sobre las ocho cosas de "lo que más probablemente está mal".

Esta revisión no autoriza ninguna campaña ni cambia el diseño de señales del propietario. No lance ninguna cola para producir el veredicto.
