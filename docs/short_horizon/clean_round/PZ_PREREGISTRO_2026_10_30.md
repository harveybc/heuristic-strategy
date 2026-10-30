# PZ: pre-registro de la medición (fijado antes de PZ3)

Fecha: 2026-10-30 (UTC). Autor: Satoshi (carril PZ). Diseño: `ESTRATEGIA_PRAGMATICA_PZ_2026_10_30.md` (heuristic-strategy @1b87f5c,
sha256 del documento 43fc3739…6267f38). Este archivo **no cambia la regla de decisión** del §5: solo la vuelve exacta y fija los
instrumentos, las entradas y el punto canónico **antes** de cualquier corrida de PZ3 / PZ4 / PZ5. Ninguna de estas definiciones se toca
después de ver resultados. Si hace falta corregir algo, se agrega una entrada fechada al registro de cambios del final y la corrida
afectada se repite.

Corrida previa a este commit, declarada: una sola corrida de integración (`B_vol_mid`, semilla 0, salidas perfectas), para comprobar que
el corredor ejecuta PZ de punta a punta. Esa celda **falla** (W2 0.38, W7 0.42, W3, W4 justificado 0.71, serie E2/E3/E4). No modificó nada
de lo que sigue.

## 1. Punto canónico de PZ (antes de cualquier corrida)

`contracts/PZ_PARAMS.json`, sha256 de los genes `36e3e235…a3d5` (`strategy_pz.genes_sha`):

| gen | valor | por qué |
|---|---|---|
| a_sl, a_tp, buffer_pips, rho | 1.0, 1.0, 1.0, 0.005 | línea base del contrato compartido (`horizon_levels.GENE_BASELINE`) |
| kappa, theta, alpha, tau0 | 0.0, 0.0, 0.0, 1.0 | apertura de punto de equilibrio y profit/risk puro: la estrategia de predicciones (con alpha 0, tau0 es inerte) |
| k_persist, p_urg | 1, 0.25 | sin persistencia; urgencia cuando el nivel fino da menos de 1/4 a favor |

Las elecciones que el diseño deja abiertas están en `strategy_pz.DECLARED`:

- **Etiquetas.** Pivotes finales con confirmación conocible. Excursiones medidas desde OPEN[b+1].
- **Órdenes.** Se construyen con `horizon_levels.levels`: SL redondeado hacia afuera y TP hacia adentro, con referencia en el cierre de la
  barra de decisión.
- **Costos.** El costo de ida y vuelta es el de la fila (`row_costs`). El costo de cierre es la mitad.
- **Banda de O\*.** Se cierra si HOLD ≤ 0.02 σ_trade − costo de cierre.
- **Horizonte de apertura.** Se recuerda por orden.
- **Sin reversa.** Igual que el C3Z canónico (allow_reverse 0).
- **Empates.** Un empate exacto entre lados es NOP. Dentro de un lado gana el horizonte más fino.
- **Censura.** Un horizonte con etiqueta censurada no es candidato.
- **Escalera de calidad.**
  - Cada una de las 24 salidas pasa por su propio AR(1) con φ = 0.587, con ρ = sin(π(q − ½)).
  - `p_up` se calibra con Platt sobre las filas CAL.
  - Los cuantiles de fav, adv y tau son la predicción más los cuantiles empíricos de los residuos en CAL.

## 2. La verdad bajo PZ (principio del fallo del menú, 2026-10-29)

Bajo `full_f4` la estrategia valora exactamente las órdenes que coloca. La verdad del motor para PZ es la repetición exacta de **sus**
órdenes (h, lado), construidas con las etiquetas perfectas (`strategy_pz.truth_orders_fn`, gancho `strategy_hierarchy.MENU_ORDERS_FN`).
La máscara del menú son las órdenes ejecutables, porque PZ no exige confirmación MT.

- **Mismos controles y testigos.** El oráculo de verdad, O\* y el oráculo de mantener actúan sobre esa verdad. Los testigos sellados
  leen `truth.menu` sin ningún cambio de forma.
- **Comprobación previa.** Con calidad 1, la corrida exige que las órdenes de la política sean byte a byte las de la verdad. Si no lo
  son, sale con `PZ_MENU_MISMATCH`.
- **C3Z no cambia.** Con `--strategy c3z`, que es el valor por defecto, el camino es el de siempre (pruebas `test_hooks_default_to_c3z`
  y la suite completa).

## 3. PZ3: nivel 1 de la matriz

**Entradas**

- Celdas: `evidence/pz_20261030/inputs/TIER1_CELLS.json`, sha256 `de51d833…c6353d3`. Es la misma cola de nivel 1 de la cadena formal:
  39 celdas base, 9 S_ y 9 M_, que dan 135 registros con semillas 0, 1 y 2.
- Parámetros: `contracts/STRATEGY_PARAMS.json` v2.3, sha256 `4e03fb1a…0b269`, perfil full_f4. Se pasa `--strategy pz --pz-params
  contracts/PZ_PARAMS.json`. Las salidas son perfectas.

**Veredicto por registro.** Se toma el veredicto gobernante del corredor (`PASS_effective`, VALIDITY_SPEC v2). Los testigos se reportan
como PASS / FAIL / NOT_EVALUABLE / CONTROL_CONFLICT, cada uno con la primera decisión divergente frente al oráculo.

**"Pasa la matriz" significa cero registros FAIL**, es decir, cero registros cuyo veredicto gobernante sea FAIL por un testigo
VALIDATED o SEALED_UNVALIDATED. Los registros NOT_EVALUABLE y CONTROL_CONFLICT se cuentan aparte: no son FAIL ni PASS. Como referencia
se informa el nivel 1 de C3Z sobre el mismo código, pero no forma parte de la regla.

## 4. PZ4: escalera de calidad y contraste

**Entradas**

- Mundos: los 24 mundos de nivel del subconjunto DEAP (`evidence/pz_20261030/inputs/LEVEL_WORLDS_24.json`, sha256 `e54d173e…bc8fc10`),
  con semillas 0, 1 y 2.
- Calidades: q ∈ {0.55, 0.62, 0.70, 0.80, 1.00}. La calidad de las cabezas de jerarquía es la del mundo.

**Brazos.** Todos corren sobre el **mismo mundo**, construido una sola vez por (celda, semilla, q).

| brazo | qué es |
|---|---|
| C3Z | punto canónico full_f4 (STRATEGY_PARAMS v2.3) mientras no exista el campeón S2; con el campeón se repite todo |
| PZ | punto canónico |
| PZ-6h | PZ con solo h = 6 |
| base trivial | lado de `p_up[6]` en cada tick plano, orden h = 6 del menú C3Z con referencias medianas y genes base, sin cierre temprano |

**Métricas por mundo y brazo**

- Sharpe diario (`sharpe_daily_sqrt365`), profit **mediano** por operación (nunca el USD compuesto), σ por operación y operaciones por
  semana.
- **Regret de cierre contra O\*.** Para cada operación del brazo: (P\* − neto realizado) / riesgo en dinero inicial, donde P\* es el
  mejor cierre ejecutable de su trayectoria o la salida natural (`closer.path`). Se toma la media sobre **todas** las operaciones del
  mundo, no solo sobre los cierres tempranos.
- El contrafactual nativo de cada cierre temprano (cuota destructiva en la banda ±0.02 σ).
- Para PZ: la cobertura de cuantiles en cada peldaño (fav [q05, q95] nominal 0.90, adv ≤ q75 nominal 0.75, tau [q05, q95] nominal 0.90),
  el MAE / naive de cada familia de salidas y el Brier / naive de `p_up`.
- Un mundo sin operaciones en un brazo es FAILED_SANITY para ese brazo. Queda fuera de las comparaciones de ese brazo y se cuenta.

**Contraste pareado**

- Para cada brazo X ∈ {PZ, PZ-6h, base trivial} contra C3Z, a q 0.70, se toma la diferencia por mundo (X − C3Z) en Sharpe y en regret de
  cierre.
- Bootstrap de conglomerados por mundo (celda, semilla): 10 000 remuestreos con semilla 20261028. Intervalo percentil al 95 %. Valor p
  bilateral = min(1, 2 · min(P(Δ ≤ 0), P(Δ ≥ 0))).
- **Familia Holm.** Son estas 6 pruebas: 3 brazos × {Sharpe, regret}, con α = 0.05. Los demás peldaños de q se informan con la misma
  columna Holm, como descripción.

**Regla de decisión (§5, exacta)**

- **PZ pasa a ser la estrategia del carril** si se cumplen las tres condiciones:
  1. PZ pasa la matriz (§3).
  2. C3Z **no** es significativamente mejor en Sharpe: no se da a la vez Δ_Sharpe(PZ − C3Z) < 0 y p Holm < 0.05.
  3. C3Z **no** es significativamente mejor en regret de cierre: no se da a la vez Δ_regret(PZ − C3Z) > 0 y p Holm < 0.05.
- **C3Z se conserva** solo si su ventaja es significativa después de Holm (2 o 3 fallan). Si PZ no pasa la matriz, el veredicto es
  "PZ no adoptada (matriz)", con el contraste informado igual.
- **Campeón S2.** El contraste se repite contra el campeón de la cadena cuando exista. Mientras tanto, el veredicto se rotula
  "contra el punto canónico".
- Gráficas (regla del propietario): Sharpe y profit contra el MAE y el MAE / naive de las salidas.

## 5. PZ5 y la calificación final (orden del propietario, 2026-10-30)

**DEAP local de los 10 genes**

- Subconjunto: el mismo de la DEAP formal (`evidence/pz_20261030/inputs/DEAP_SUBSET_F4.json`, sha256 `e15d008f…22d43e1`).
- Fitness: solo Sharpe, con restricciones Deb (testigos como restricciones, igual que la DEAP formal). Población ≥ 15 y generaciones ≥ 30.
- Recursos: 2 trabajadores bajo crispdm-run, dentro de 3 GiB.
- Reanudable con una bandera STOP.
- Puertas previas al lanzamiento: la prueba de propiedad espejo sobre el sha de lanzamiento, y después la puerta de degeneración de la
  generación 0.

**Calificación final de PZ.** Usa los mismos instrumentos de QUALIFICATION_SPEC v2.6 D9:

- bootstrap de conglomerados por sorteo con columna Holm;
- re-corridas de igualdad de traza con tamaño 1;
- ablaciones de un solo horizonte;
- contrafactual nativo de cada cierre temprano a q < 1.

## Registro de cambios

- 2026-10-30: versión inicial, antes de PZ3.
