# De imitar al ZigZag a aprender consecuencias: la escalera causal aplicada a nuestro agente (diseño, 2026-10-25)

Para el propietario. Responde a: "¿lo estamos haciendo bien o por intuición? ¿Sirve la escalera de Pearl
(asociación, intervención, contrafactual) obtenida de los datos?". Nada de esto se ha ejecutado; es el diseño de la
etapa que sigue a WF_V1 y a la auditoría de cierre temprano.

## 1. Qué estamos haciendo de verdad, con su nombre

No es RLHF (eso es aprender una recompensa a partir de preferencias humanas). Es **aprendizaje por imitación de un
experto clarividente**: clonamos (con DAgger, Ross et al. 2011) a un ZigZag que conoce el futuro. Es una práctica
conocida, pero con un problema documentado: el **imitation gap** de los expertos con información privilegiada
(Warrington et al., ICML 2021). El experto actúa con información que el estudiante no ve. Por eso el estudiante
aprende "la acción media del experto dado lo que yo veo", que puede ser mala, y se pega a correlaciones espurias
(**causal confusion**, de Haan et al., NeurIPS 2019). El walk-forward de hoy es coherente con eso: Sharpe 4–8 en el
ajuste y ≈ 0 o negativo el año siguiente, sin que ningún brazo sostenga.

## 2. La clave de nuestro problema: nuestras acciones no mueven el precio

Con nuestro tamaño (sin impacto de mercado), **la intervención sobre nuestra acción es identificable exactamente con
datos históricos**. Para cualquier barra, el resultado de abrir largo o corto con un TP/SL dado, mantener o cerrar se
calcula con el motor de replay sobre el camino de precios que ocurrió. Así, los peldaños 2 y 3 de Pearl, **para
nuestras acciones**, no necesitan supuestos ni modelos vivos:

| peldaño | pregunta | en nuestro caso | cómo se obtiene de los datos |
| --- | --- | --- | --- |
| 1. asociación | P(Y \| X) | ¿qué régimen / resultado sigue a esta ventana? | modelos supervisados (lo que hacemos hoy) |
| 2. intervención | P(Y \| do(A)) | ¿qué pasa si **abro / cierro / no hago nada** aquí? | replay exacto de cada acción en cada barra ("tabla de resultados") |
| 3. contrafactual | Y_{A=a'} dado lo que pasó | **este** trade, que cerré: ¿qué habría pasado si lo mantengo? | mismo camino de precios, otra acción; es la auditoría de cierre temprano |

Para **variables de mercado** ("¿y si la sorpresa de la Fed hubiera sido cero?", "¿y si la volatilidad fuera el
doble?") sí hacen falta supuestos causales. Ahí aplican las herramientas de datos observacionales del libro que
viste (*Causal Inference for Data Science*, Ruiz de Villa, Manning 2024; también Facure, *Causal Inference in
Python*, O'Reilly 2023; Hernán y Robins, *What If*): ajuste por la puerta trasera, double ML (Chernozhukov et al.
2018), emparejamiento y control sintético.

**El contrafactual no es "creatividad".** Es tomar algo que **sí pasó**, dejar fijo todo lo demás (el camino del
mercado, el ruido) y cambiar una sola cosa. Para nuestras acciones es un cálculo exacto. Para variables del mercado se
aproxima buscando en los datos **situaciones gemelas** (ventanas casi iguales donde esa variable fue distinta) o con un
modelo generativo condicionado (nuestro `synthetic-datagen`).

## 3. Diseño propuesto (etapas C1–C4)

**C1 — Asociación con objetivos ordenados: el "perfil del régimen" que propusiste.**
- Para cada barra, lado y escala de la escalera ZigZag, calcular el perfil del mejor trade disponible:
  - ganancia del tramo en σ;
  - excursión adversa relativa (drawdown dentro del trade);
  - excursión favorable;
  - volatilidad realizada durante el trade;
  - duración;
  - tiempo hasta TP y tiempo hasta SL.
- Aprenderlo por **regresión multi-salida**, que conserva el orden (como dices). La clasificación queda sólo para la
  decisión final, que es categórica de verdad.
- Es el *triple-barrier labeling* + *meta-labeling* de López de Prado (*Advances in Financial ML*, 2018),
  generalizado a varias escalas.

**C2 — Intervención sobre nuestras acciones: aprender las consecuencias, no imitar al experto.**
- Tabla de resultados por barra: para abrir largo o corto en una rejilla de TP/SL, el resultado neto en σ, si toca
  primero el TP o el SL, y el MAE.
- Para cada posición abierta: el valor de mantener frente a cerrar ahora.
- Los especialistas aprenden Q(x, a) = E[resultado | x, do(a)] por regresión. Para las entradas es un bandido
  contextual con información completa (conocemos el resultado de **todas** las acciones, no sólo de la tomada).
- El cierre temprano pasa a ser un problema de **parada óptima** con objetivo directo: cerrar cuando
  E[lo que queda | x] < 0, o cuando P(SL antes que TP | x) supera el umbral de la capa superior. Es exactamente tu regla
  de negocio.
- El ZigZag deja de ser "el maestro" y queda como descriptor de régimen (C1) y objetivo auxiliar.
- Elimina de raíz el imitation gap: el objetivo no depende de lo que sabía un experto clarividente, sólo de lo que
  ocurrió.

**C3 — Robustez causal de las características (contra la caída TRAIN → año siguiente).**
- Años como "entornos". Predicción invariante (ICP, Peters, Bühlmann y Meinshausen, JRSS-B 2016) / IRM (Arjovsky et
  al. 2019): conservar sólo las relaciones característica → resultado que se mantienen estables en 2020, 2021, 2022
  y 2023.
- Lo que cambia de signo entre años es lo que hoy produce Sharpe 6 en el ajuste y 0 después.
- Para los motores exógenos (noticias, sorpresas macro), efectos con double ML sobre el estudio de eventos que ya
  hicimos.

**C4 — Contrafactuales con datos.**
- **Negativos difíciles**: pares de ventanas casi idénticas en las características seleccionadas, con resultados
  opuestos (vecinos más cercanos; aprendizaje contrastivo). Son los "casos negativos" que propusiste para afinar los
  modelos.
- **Evaluación fuera de política exacta**: cualquier política nueva se evalúa en los datos sin operar en vivo, porque
  sin impacto de mercado el replay es el contrafactual exacto.
- Auditoría por trade: cerrar / mantener / revertir (en curso).

## 4. Supuestos y límites declarados

- Sin impacto de mercado (razonable a nuestro tamaño); con impacto, el peldaño 2 dejaría de ser exacto.
- Los resultados por acción usan el futuro **sólo como etiqueta**: nunca entran en la observación (misma disciplina de
  disponibilidad y censura que hoy).
- Las afirmaciones causales sobre variables de mercado requieren supuestos (sin confusores no medidos, positividad) y
  se reportan como tales.
- El juez sigue siendo el walk-forward con sus reglas pre-registradas; TEST cerrado.

## 5. Orden propuesto

C2 primero, porque ataca directamente la causa más probable del fracaso actual y da el objetivo del cierre temprano;
luego C1 como auxiliares/descriptores; luego C3 sobre la selección; C4 en paralelo como evaluación. Cada etapa se
pre-registra antes de correr y se juzga en el walk-forward.

## 6. Evaluación de una propuesta externa (meta-learners, IPTW, contrafactuales por cVAE), 2026-10-25

Se revisó una propuesta que trata las características como "tratamiento" (do(Xᵢ = x')), el régimen ZigZag actual como
confusor y el régimen futuro como resultado; con Causal Forests / X-Learners, ponderación IPTW de la pérdida y
contrafactuales por abducción en el espacio latente de un cVAE.

**Se rechaza, con medición:**

- **IPTW sobre características: inviable por positividad.** Medido en TRAIN (`evidence/causal/FEATURE_POSITIVITY_TRAIN.json`):
  - la característica mediana se predice desde las otras 119 con R² 0.94 (con un modelo lineal, que es una cota
    inferior);
  - el 61 % tiene R² ≥ 0.9 y el 23 % ≥ 0.99;
  - los pesos IPTW de la característica mediana dejan un tamaño efectivo de muestra de **44 filas de 18 063 (0.24 %)**,
    con un peso máximo 1 514 veces la media.
  
  Entrenar con esos pesos equivale a entrenar con unas decenas de filas.
- **do(característica) no está bien definido.** Una fase de Hilbert o una densidad multitaper es una función
  determinista del camino pasado de precios. No se puede fijar sin cambiar ese camino, y por tanto todas las demás
  características (violación de consistencia / intervención bien definida). Su "efecto causal" sobre el régimen futuro
  no es una cantidad identificable con estos datos.
- **"Contrafactual" por cVAE: no lo es.** Perturbar un 10 % una dimensión latente y decodificar produce entradas
  sintéticas fuera de la distribución observada. El latente de un VAE no está identificado como ruido exógeno de un
  SCM (los resultados de imposibilidad de desenredado lo confirman). Es aumento de datos o entrenamiento adversarial,
  útil quizá como regularizador, pero no un contrafactual de Pearl. La promesa de "convergencia exponencial" no tiene
  respaldo.
- **El régimen actual sacado del ZigZag "hasta T" filtra el futuro** si no se usa la versión confirmada causalmente: el
  último tramo del ZigZag se repinta.
- **Hechos que no coinciden con nuestro sistema**: barras H1, no ticks; 121 columnas en la receta, no 45; y el ZigZag
  no es un experto humano.

**Se adopta, reformulado:**

- **Estratificar por régimen causal-confirmado.** Usar el estado del ZigZag multi-escala **confirmado en T** (sin
  repintado) como entorno, junto con los años, en la selección invariante de C3 (ICP/IRM). La heterogeneidad del
  efecto por régimen es una pregunta legítima.
- **Meta-learners (EconML) donde el tratamiento sí es manipulable: nuestra acción.** Allí ni siquiera hace falta
  propensidad, porque el replay da el resultado de **todas** las acciones (C2). Para motores exógenos (eventos, sorpresas
  macro) se usan DML / causal forest con los refutadores de DoWhy (tratamiento placebo, causa común aleatoria,
  subconjuntos) antes de afirmar un efecto.
- **Sensibilidad de las cabezas a perturbaciones pequeñas dentro de la variedad de datos** (p. ej. entre ventanas
  vecinas reales) como regularizador medido en el walk-forward (C4), declarado como robustez, no como contrafactual.

Librerías: EconML y DoWhy (PyWhy) para lo exógeno; nada de SCM nativo en un cVAE.
