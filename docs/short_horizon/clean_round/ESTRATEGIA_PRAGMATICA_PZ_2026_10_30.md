# Estrategia pragmática PZ para el carril guiado por ZigZag: salidas del modelo y contraste con la estrategia formal

Fecha: 2026-10-30 (America/Bogota). Autor: Satoshi. Pedido del propietario: "haz lo mismo para tu caso individual, para que
tengamos contra qué contrastar la estrategia formal con la pragmática". Estado: PROPUESTA ACEPTADA PARA IMPLEMENTAR COMO CONTRASTE.
No reemplaza la cadena F4 en curso (estrategia formal C3Z `full_f4`): corre a su lado, con el mismo contrato, el mismo motor y
los mismos mundos, y gana sólo si lo demuestra.

## 1. La pregunta y mi respuesta corta

¿Hay una estrategia más simple, como la de predicciones de corto y largo plazo, que tome las salidas ideales de nuestros modelos
y decida comparando ganancia por riesgo (y por tiempo) del tick actual? Sí. Y en mi carril es todavía más natural que en el de
Musashi, porque **las salidas que necesita una estrategia simple son exactamente las cantidades que el experto ZigZag ya etiqueta
en cada tick**: hacia dónde va el tramo de cada horizonte, cuánto recorre a favor hasta el próximo pivote, cuánto en contra antes
de llegar, y cuánto tarda. Eso define a la vez:

1. **las salidas del modelo RLHF** (los objetivos supervisados del experto, sin ambigüedad);
2. **la apertura** (ganancia esperada por unidad de riesgo y de tiempo, el criterio de la estrategia de predicciones);
3. **el cierre temprano** (la misma pregunta recalculada sobre la posición abierta, con el nivel fino como "predicción de corto").

Lo que hemos medido ya apunta en esta dirección: los vetos de la jerarquía no se ganan su lugar en Sharpe (escalera de calidad y
prueba cruzada, evidencias `hierarchy_value_ladder_443b7a1`, `hierarchy_value_cross_6ee6cf7`), y en el carril de Musashi la mejora
fue ajuste de coeficientes en un solo horizonte, no arbitraje entre horizontes (auditoría crossed v2, CX-14). La complejidad debe
demostrar una mejora; PZ es la línea base que la obliga a demostrarlo.

## 2. Salidas del modelo (el contrato que hoy no teníamos)

Horizontes: los cuatro ZigZag MetaTrader de las referencias de Musashi, `h ∈ {3, 6, 12, 24}` horas nominales (depth/backstep
3/1, 4/2, 7/6, 12/11; deviation inerte). Para cada tick `t` (fila observable) y cada `h`:

| salida | definición exacta de la etiqueta (experto, retrospectiva) | tipo | uso |
|---|---|---|---|
| `p_up[h]` | 1 si el próximo pivote confirmado del nivel `h` posterior a `t` es un **pico** (el precio sube primero), 0 si es un valle | probabilidad calibrada (Platt/isotónica en calibración) | dirección |
| `fav[h, s]` cuantiles 25 / 50 / 75 | para el lado `s` ∈ {compra, venta}: excursión favorable desde el precio ejecutable en `t+1` hasta el próximo pivote del nivel `h` en la dirección de `s` (pips) | regresión cuantílica | TP |
| `adv[h, s]` cuantiles 50 / 75 / 90 | excursión adversa máxima antes de ese pivote (pips), incluida la barra del pivote con la regla conservadora de Musashi (MAE desde la entrada) | regresión cuantílica | SL |
| `tau[h]` mediana | horas hasta ese pivote | regresión cuantílica | ganancia por tiempo |

Total: 4 × (1 + 2 × 3 + 2 × 3 + 1) = **56 salidas** por tick. Versión mínima (primer entrenamiento): `p_up`, `fav` q50, `adv` q75 y
`tau` q50 por horizonte y lado = **24 salidas**. Todas son etiquetas del experto por construcción (RLHF = imitación del experto con
la retroalimentación del ZigZag multi-resolución); purga de entrenamiento ≥ duración máxima del tramo de cada nivel; censura
explícita cuando el pivote siguiente cae fuera del bloque (nunca NOP fabricado); entradas sólo causales, sin estado del agente.

Por qué cuantiles y no un solo número: el propietario pidió "incluir la incertidumbre o un rango de tolerancia". El ancho
`fav q75 − q25` es esa tolerancia, medible y calibrable (cobertura de intervalos en calibración, criterio de aceptación 0.8–0.95 a
nominal 0.9, como exige mi auditoría a Musashi, CX-11).

## 3. Apertura

Para cada `(h, s)` en el tick `t` (sólo dentro de la ventana de entrada del contrato):

```text
p      = p_up[h]           si s = compra;   1 − p_up[h]   si s = venta
SL     = a_sl · adv_q75[h, s] + colchón_pips        (redondeado hacia afuera, mínimo del bróker)
TP     = a_tp · fav_q50[h, s]                       (redondeado hacia adentro)
G      = p · TP − (1 − p) · SL − costo_ida_y_vuelta − κ · (fav_q75 − fav_q25)[h, s]
score  = (G / SL) / (tau_q50[h] + τ0)^α             ganancia por unidad de riesgo y por unidad de tiempo
```

Se abre el `(h, s)` de mayor `score` si `score > θ`; si ninguno supera `θ`, NOP. Todas las direcciones se comparan antes de decidir
(nunca prioridad accidental de la compra). Volumen: el del contrato común, `q = c · min(1, ρ · balance / L1)` con la reserva de
financiación hasta el cierre forzado (S3 de mi carril, idéntico al de Musashi). `α = 0` recupera el profit/risk puro de la
estrategia de predicciones; `α = 1` es ganancia por hora.

Genes (todos con cotas declaradas, optimizables por DEAP): `a_sl ∈ [0.5, 2]`, `a_tp ∈ [0.5, 2]`, `colchón ∈ [0, 5]`, `κ ∈ [0, 1]`,
`θ ∈ [−0.2, 1]`, `α ∈ {0, 0.5, 1}`, `τ0 ∈ [0.5, 6]`, `ρ ∈ [0.001, 0.01]`. Diez genes, frente a 36 de C3Z `full_f4`.

## 4. Cierre temprano (la prioridad del propietario)

Una sola pregunta por tick sobre la posición abierta (lado `s`, horizonte de apertura `h_p`, precio actual `P`, niveles SL/TP
fijos en el bróker): ¿mantener sigue pagando su riesgo?

```text
p'      = probabilidad a favor de s en el nivel h_p, recalculada en el tick actual
p'_f    = probabilidad a favor de s en el nivel inmediatamente más fino que h_p (h = 3 usa el propio 3)
HOLD    = p' · (TP − P) − (1 − p') · (P − SL)          en pips, con las distancias que quedan
cerrar  si  HOLD < −costo_cierre  durante k ticks seguidos,
        o   si  p'_f < p_urg   (reversión fina confirmada: salida urgente, sin persistencia)
```

Es la lógica de la estrategia de predicciones (comparar el profit/risk de mantener con el de salir), con el nivel fino haciendo
de predicción de corto plazo. La regla de empate es la del control designado O\* (cerrar si mantener no supera a cerrar por
más de la banda), la misma que restauré en C3Z (decisión C′). Genes: `k ∈ {1, 2, 3}`, `p_urg ∈ [0.05, 0.5]`. No hay curvas de primer
paso, ni filtro de Kalman, ni vetos: si la versión simple pierde contra la formal en cierre temprano, eso queda medido.

## 5. Contraste formal contra pragmático (cómo se decide)

Mismo contrato (`full_f4`), mismo motor, mismos mundos, mismas salidas ideales degradadas por la misma escalera de calidad
(AR(1) φ = 0.587 sobre cada salida, cobertura medida), mismo presupuesto de búsqueda.

| brazo | qué es |
|---|---|
| **C3Z formal** | el campeón de la cadena F4 (menú absoluto, jerarquía, cerrador de curvas con filtro) |
| **PZ pragmática** | este documento |
| **PZ-6h** | PZ restringida a `h = 6` (ablación de un solo horizonte, como el P1-only6 de Musashi) |
| **base trivial** | "abre siempre el lado de `p_up` del nivel 6 h con los perfiles medianos de Musashi; cierra sólo por SL/TP/forzado" |

Instrumentos (los de la calificación final, `QUALIFICATION_SPEC` v2.6):

1. Matriz de comportamiento con testigos sellados (nivel 1, luego nivel 2): ¿la estrategia simple se comporta bien en todas las
   condiciones con salidas perfectas? Mismas reglas: nada se adapta a la estrategia.
2. Escalera de calidad `q ∈ {0.55, 0.62, 0.70, 0.80, 1.00}`: Sharpe, profit, σ por operación, operaciones por semana contra MAE y
   MAE/naive de las salidas (regla de gráficas del propietario), y la calidad de equilibrio.
3. Cierre temprano: contrafactual nativo de cada cierre (cuota perjudicial en banda ±0.02σ, regret contra O\* en unidades de
   riesgo en dinero) y las series E sobre episodios con posición forzada.
4. Bootstrap por mundo (celda, semilla) con columna Holm para cada comparación pareada; tamaño 1 para separar exposición.

Regla de decisión, fijada antes de correr: **si PZ no es peor que C3Z dentro del intervalo pareado en Sharpe y en regret de
cierre a q 0.70, y pasa la matriz, la estrategia del carril pasa a ser PZ** (más simple, menos genes, salidas definidas). C3Z sólo
se conserva si su ventaja es significativa después de Holm. En cualquier caso, el contrato de salidas de §2 queda fijado para la
selección de características fase 2.

## 6. Plan de ejecución (sin tocar la cadena F4)

| paso | contenido | recursos |
|---|---|---|
| PZ1 | etiquetas del experto de §2 sobre los mundos sintéticos (verdad exacta del motor) y la cabeza ideal + escalera de calidad por salida | local, ≤ 3 GiB |
| PZ2 | `strategy_pz.py` (apertura §3, cierre §4) sobre el motor `full_f4`, pruebas primero, perfiles viejos byte-idénticos | local |
| PZ3 | nivel 1 de la matriz con salidas perfectas; diagnóstico de la primera decisión divergente de cada falla | host si hay margen, si no local |
| PZ4 | escalera de calidad y contraste contra C3Z (con el campeón de la cadena cuando exista; antes, con el punto canónico) | host después de S2, o local |
| PZ5 | DEAP de los 10 genes de PZ (Sharpe solo, testigos como restricciones, puertas de lanzamiento) | host cuando la cadena F4 esté en Q o termine |
| PZ6 | informe del contraste, incluido en el informe y la solicitud de auditoría de la estrategia | — |

Lo que PZ necesita de los datos reales (para la fase 2, no ahora): soporte por horizonte de las etiquetas de §2 (los conteos de
orígenes completos y censurados de Musashi, plan 126, ya lo dan: 24 h tiene 7 340 censurados de 19 879), tasas base de `p_up` por
horizonte y la cobertura de los cuantiles en calibración.

## 7. Registro de cambios

### v1.1 (2026-10-30): corrección de diseño DECLARADA, por fallo del coordinador tras PZ3

**Evidencia de v1.0.** El nivel 1 de la matriz con salidas perfectas, en el punto canónico y con full_f4, dio **0 de 135 registros
PASS** (`evidence/pz_20261030/tier1_e6091ae/`, código e6091ae; pre-registro en `PZ_PREREGISTRO_2026_10_30.md`). Sobre 32 231
operaciones (39 celdas base, semilla 0) las salidas fueron:

| EARLY | SL | TIME | TP |
|---|---|---|---|
| 0.860 | 0.104 | 0.036 | **0.000** |

Fallaron en todos los registros W2, W3, W7, W15 y E2/E3/E4. Además, el 25 % de los cierres tempranos tenía todavía valor de mantener
mayor que 0.02 σ, y 5 985 de esos los disparó solo la regla urgente.

**Mecanismo: dos defectos de DISEÑO de este documento, no de parámetros**

1. **TP en el extremo predicho.** Con salidas perfectas, el TP de §3 queda en el extremo del próximo pivote. El motor llena un TP límite
   solo cuando el precio lo atraviesa por `limit_through` (1 pip), y un pivote retrospectivo solo lo toca. Por construcción el TP no es
   ejecutable: la orden mantenida termina en SL o en TIME, y la verdad de su propio menú es negativa.
2. **Regla urgente sin umbral de alcance.** La regla urgente de §4 cerraba en cada contra-pivote fino, aunque el tramo propio de la
   posición siguiera su curso.

**Corrección v1.1**

- **Construcción de la orden (§3).** Distancia del TP = a_tp · fav_q50 − (limit_through + spread del lado de salida a la hora de la
  decisión), redondeada hacia adentro. Una distancia ≤ 0 deja al horizonte sin orden. Lado de salida: un largo sale al BID (las barras
  son BID, sin spread); un corto sale al ASK (el spread de la hora cuando el motor dispara los cortos en el ask, `short_trigger_on_ask`).
- **Cierre urgente (§4).** Solo se cierra de inmediato cuando el nivel fino se vuelve en contra (p'_f < p_urg) **y** se predice que esa
  reversión alcanza el SL: el fav_q50 del lado opuesto en el nivel fino ≥ la distancia que queda hasta el SL.
- **Rango de p_urg.** Pasa a [0, 0.5], donde 0 = cierre urgente apagado.

**Qué no cambia**

- Los valores del punto canónico siguen como se pre-registraron (a_tp 1, p_urg 0.25, el resto igual; sha de genes `36e3e235…`).
- La regla de decisión de §5 no cambia.
- Ningún testigo cambia.
- El resultado de nivel 1 de v1.0 se conserva como historia. En el informe final del contraste figura como "PZ v1.0 falló por una razón
  de geometría de ejecución". El código lo reproduce byte a byte con `strategy_pz.DESIGN_VERSION = "1.0"` (`--pz-design 1.0`).

**Pruebas y plan**

- Prueba unitaria nueva: una orden con salidas perfectas en un pivote es ejecutable, y su TP se llena cuando el precio lo atraviesa;
  el TP de v1.0 en la misma barra nunca se llena.
- Secuencia: nivel 1 de nuevo con v1.1 (informe b′), luego PZ4 y PZ5 sobre v1.1.
- La corrida de PZ4 sobre v1.0 (180 mundos en el host, 36 locales, detenida con su bandera STOP) se rotula como historia v1.0.

### v1.2 (2026-10-30): segunda corrección de diseño DECLARADA (fallo del coordinador tras el nivel 1 de v1.1)

**Evidencia de v1.1.** Nivel 1 con 0 de 135 PASS (`evidence/pz_20261030/tier1_v11_a217215/`). Las correcciones de v1.1 funcionaron: las
salidas por TP pasaron de 0.000 a 0.659 y el neto por operación de 0.50 σ a 0.92 σ. Quedan dos mecanismos, y ambos son errores de
diseño de este documento.

1. **Evento equivocado.** La p de §3 y §4 era la dirección del PRÓXIMO pivote, no P(TP antes que SL) de la orden de cada lado. Una compra
   apunta a su propio pico y su SL cubre la MAE del valle intermedio; con salidas perfectas llega con certeza, pero recibía p = 0. Eso
   explica W2 lado equivocado (mediana 0.214), W7 (0.59) y W1. La regla HOLD, que usaba la misma p', cerraba posiciones cuyo pivote propio
   todavía llegaba. Eso explica W4 y E4.
2. **SL de cortos sin spread.** El stop de un corto se dispara en el ASK y adv se mide sobre los máximos BID. Resultado: 0 SL de compra y
   11.6 % de SL de venta. Con salidas perfectas ningún episodio de compra está destinado a SL (EP_counts de compra 0 de 30).

**Corrección v1.2 (`strategy_pz.V12_RULES`, `DESIGN_VERSION = "1.2"`)**

1. **El evento propio de la orden.** p_s = P(el objetivo propio de ESTA orden antes que SU stop).
   - El SL está en el cuantil declarado q_SL de la excursión adversa antes del pivote propio del lado. Con cuantiles calibrados,
     P(SL primero) ≈ 1 − q_SL, así que p_s = q_SL. El canónico es q_SL = 0.75, el adv_q75 de antes.
   - Con salidas perfectas los cuantiles colapsan y p_s = 1.
   - En PZ5, q_SL es un gen categórico en {0.5, 0.75, 0.9}.
   - p_up queda como etiqueta de diagnóstico y como desempate de un empate exacto entre lados: se usa el p_up del slot óptimo más fino
     (> 0.5 compra, < 0.5 venta, = 0.5 NOP). El score ya no lo usa.
   - La cobertura por (h, lado) en cada peldaño (`head_report` → adv `below_by_h_side`) es la que valida este mapeo.
2. **HOLD con la trayectoria propia de la orden.** En cada tick se recalculan las salidas de (h_p, s) desde el precio actual P:
   - **SL primero.** Si adv_q{q_SL}(ahora) ≥ la distancia que queda al SL, se predice SL primero. Se cierra cuando cerrar le gana a
     mantener dentro de la banda de O\* (A = dist_SL − costo_cierre ≥ −banda). Esta es la regla urgente y no lleva persistencia.
   - **TP inalcanzable.** Si no, y fav_q50(ahora) < la distancia que queda al TP − tolerancia (el offset de salida de la orden en la fila
     actual), se predice TP inalcanzable. Entonces HOLD = p_s · min(fav_q50, dist_TP) − (1 − p_s) · dist_SL, y se cierra si
     HOLD < −costo_cierre durante k ticks.
   - **En otro caso, se mantiene.**
   - p_urg queda **inerte**: sigue en el documento y fuera del genoma.
3. **SL de cortos.** Distancia del SL = a_sl · adv_q{q_SL} + colchón + spread del lado de salida (el spread de la hora para un corto
   cuando el motor dispara los cortos en el ask; 0 para un largo). Es el gemelo del lado SL de la corrección de TP de v1.1.

**Qué no cambia.** Los demás valores canónicos (se agrega q_sl = 0.75, el valor que v1.0 y v1.1 ya usaban), la regla de decisión y los
testigos. Las historias v1.0 y v1.1 se conservan y se reproducen con `--pz-design 1.0` y `--pz-design 1.1`.

**Pruebas nuevas**

- Con salidas perfectas, una venta cuya excursión adversa queda bajo el SL nunca toca el stop (la prueba del motor en el ask) y su TP se
  llena.
- Una compra y una venta son espejo exacto con spread cero.
- p_s vale 1 con salidas perfectas y q_SL por debajo.
- La regla de cierre sobre la trayectoria propia (SL primero, TP inalcanzable con persistencia, mantener).

**Si v1.2 tampoco pasa la matriz**, se informan los mecanismos y el coordinador decide si PZ continúa. No se itera el diseño
indefinidamente.
