# Runbook: cadena de estrategia F4 (desatendida, reanudable)

Orden del dueño (2026-10-29): «orquesta y ejecuta, cero babysitting». Desde la etapa 1 hasta el informe final, todo
corre como una cadena desatendida en `<host-trabajo>`. Cualquiera (el dueño, Musashi, Retsu) puede consultar el estado o
tomar el control en cualquier momento siguiendo este documento, sin hablar con nadie.

Convenciones de este documento:

- `<host-trabajo>` es la máquina de trabajo. No se escriben nombres de máquina, usuarios ni rutas personales.
- `R=/var/tmp/behaviour-matrix-20261027` es la raíz de trabajo y `C=$R/chain_f4` la raíz de la cadena.
- Todos los comandos se ejecutan en `<host-trabajo>` con el usuario dueño de la cadena.
- Rama: `satoshi/expert-guided-rl-20261019` del repositorio heuristic-strategy. El sha exacto del código que corre está
  en `$C/CODE_SHA`, y su copia exacta en `$C/code` (sale de `git archive`, nunca de un árbol de trabajo).

---

## (a) Qué hacemos y por qué

### El camino completo

1. **Contrato compartido (full_f4).** El dueño adoptó el contrato de negocio de Musashi el 2026-10-29:
   - apalancamiento 50, sin tope diario de entradas;
   - corte del calendario 12 h antes del cierre forzado;
   - niveles SL/TP absolutos por horizonte (referencias de TRAIN) y tamaño por presupuesto de riesgo (`rho`);
   - el **menú de decisión**: las órdenes candidatas (h, lado) con h ∈ {3, 6, 12, 24} horas.
   Este perfil de ejecución es `full_f4`. El perfil auditado anterior, `full_f3`, sigue byte-idéntico como control.
2. **Matriz de comportamiento.** Son mundos sintéticos con la verdad del motor conocida. Cada testigo (W0–W17,
   E1–E8, Z1) compara la estrategia con su control designado (O* o el oráculo de mantener). Primero se pasó la
   escalera tier 1 / tier 2, con las resoluciones A–D del coordinador y C retirada (v2.3).
3. **Puertas previas.** La prueba de propiedad de espejo en el sha exacto de lanzamiento y la puerta de degeneración
   de la generación 0.
4. **Optimización en tres etapas.**
   - Etapa 1: rejilla de niveles con los genes canónicos.
   - Etapa 2: DEAP sobre el resto de genes con los niveles congelados.
   - Etapa 3: comprobación de los 5 mejores puntos de nivel con el campeón.
   - El criterio es solo Sharpe, con restricciones de Deb (primero menos testigos fallidos y luego mayor Sharpe).
5. **Calificación final (QUALIFICATION_SPEC v2.6)** e **informe de datos**.

### Etapas de la cadena

Las etapas corren en orden: S1 → S1R → G2 → S2 → S3 → Q → R.

#### S1 — Rejilla de niveles (etapa 1)

- **Entradas:**
  - el subconjunto DEAP: 87 mundos (45 base a q1.00, 24 a q0.70, 9 pares gemelo/espejo) más 9 mundos de control con
    referencias intercambiadas;
  - `STRATEGY_PARAMS.json` v2.3 (sha `4e03fb1a…`);
  - el código `5e04e83`.
- **Qué hace:**
  - recorre 144 puntos: `a_sl, a_tp ∈ {0.75, 1, 1.5}`, `b_sl, b_tp ∈ {0, 0.5}`, `buffer_pips ∈ {1, 3}` y cuantil
    {mediana, p75};
  - reconstruye la verdad del menú en cada punto;
  - en total son 11 232 tareas (punto, grupo de mundos).
- **Salidas:** `$R/deap_f4_stage1_5e04e83/out/` con `TASKS.jsonl`, `PROGRESS.log` y `GATE_GEN0.json`.
- **Regla de la puerta:** la regla de degeneración de la generación 0 se aplica a los primeros 16 puntos. Si los
  declara degenerados, la etapa escribe `FAILED_DEGENERATE_GEN0` y la cadena se detiene con `FAILED_S1`.
- **Duración:** unas 18–19 h con 8 procesos y tope de 10 GiB.
- **Puntos inviables:** un punto que no deja filas de referencia en la calibración (por ejemplo `b_tp 0.5` con
  `a_tp 0.75`) queda registrado como INFEASIBLE con su motivo. Nunca entra en el ranking.

#### S1R — Resultado de la etapa 1 y regla de borde

- **Salidas en `$C/work/S1R/`:**
  - `STAGE1_RESULT.json`, con el ranking de Deb y los puntos inviables aparte;
  - `EDGE.json`;
  - `FREEZE.json`;
  - `FROZEN_PARAMS.json`.
- **REGLA DE BORDE (no detiene la cadena):**
  - Caso general: si el punto ganador está en un borde de un eje numérico, se evalúa **una sola** extensión en ese
    eje. Es un paso de rejilla más allá, recortado a los límites sellados de `full_f4`, con los genes canónicos.
  - Límite sellado: si el valor ya está en el límite sellado, el eje se registra como **BOUNDARY_HIT** (para el
    informe) y no se extiende.
  - Eje categórico: el cuantil (mediana/p75) nunca es borde.
  - Después se re-ordena todo por Deb y se congela el mejor.
  - Si gana un punto de extensión, se queda congelado aunque esté en el nuevo borde (se informa como
    `edge_after_extension`). No hay segunda extensión.
- **Duración:** minutos, o menos de 1 h si hay extensiones (como mucho 5 puntos × 78 grupos).

#### G2 — Puerta en el sha de la cadena

- **Qué hace:** la prueba de propiedad de espejo con el punto congelado más 50 genomas aleatorios × 9 pares, en la
  forma `full_f4` (mundo espejo con referencias intercambiadas).
- **Salida:** `$R/qual/PROPERTY_PASSED_<sha>`, que solo se escribe si las 459 ejecuciones son simétricas en el código
  y no hay veredicto W14 por ejecución.
- **Fallo:** `FAILED_G2`, sin reintento.
- **Duración:** unas 1,5 h con 4 instancias.

#### S2 — DEAP de la etapa 2

- **Genes que busca:** todos los genes no de nivel (`GENOME_F4_STAGE2`): `open_rate` ∈ [0.002, 1.0], `z_veto`,
  `burst_vol_ratio` ∈ [1.2, 2.5], los genes del cerrador, etc.
- **Configuración:**
  - parte del punto congelado;
  - población 16 × 30 generaciones, elitismo 2;
  - solo Sharpe con Deb;
  - el genoma canónico va en la generación 0;
  - la puerta de degeneración de la generación 0 es automática;
  - se reanuda desde `CHECKPOINT.pkl`, y STOP se lee entre generaciones.
- **Salidas en `$C/work/S2/`:** `ARCHIVE.jsonl`, `RESULT.json` y `CHAMPION.json`.
- **Duración:** unas 17–20 h, con tope de 14 GiB y 8 procesos.

#### S3 — Comprobación de la etapa 3

- **Qué hace:** re-evalúa los 5 mejores puntos de nivel con los genes del campeón. Para cada uno calcula la
  diferencia de Sharpe pareada por mundo frente al punto congelado, su media y su error estándar (SE).
- **Regla:** si otro punto, no peor en la restricción, gana por **más de 1 SE**, la etapa 2 se re-ejecuta **una sola
  vez** desde ese punto (S2b, arranque en caliente solo con genomas del archivo de S2). Después no se itera más.
- **Salidas:** `S3_DECISION.json`, `FINAL_FROM.json` y `$C/work/FINAL_CHAMPION.json`.
- **Duración:** menos de 1 h, o unas 18 h más si hay re-ejecución.

#### Q — Calificación final (QUALIFICATION_SPEC v2.6)

Cada paso tiene su marca `$C/work/Q/<paso>.done` y se reanuda desde sus archivos de resultado.

- **Matriz completa:** semillas 0–2 con gemelos (4 345 registros, unas 8 h) y su informe por tier.
- **D3, semillas no vistas 3–5:**
  - tres brazos: canónico, campeón y campeón sin reversión (D6);
  - confirmación pareada por celda base con IC bootstrap;
  - unas 25 h.
- **Escalera de calidad:** q ∈ {0.55, 0.62, 0.70, 0.80, 1.00} sobre los 24 mundos de nivel × 3 semillas, con el MAE de
  la cabeza y el MAE ingenuo de cada registro.
- **D9 tamaño 1:** re-ejecución con 1 unidad por orden. Las trazas (entrada, salida, lado, motivo) deben ser iguales a
  las de la ejecución con tamaño: así se separa la exposición de las decisiones.
- **D9 ablaciones:** mismos genes con el menú restringido a h6, h12 o h24.
- **D4:** controles negativos y posiciones sembradas.
- **D7:** escalera del cerrador (q 0.85 y 0.70, phi 0.587).
- **D9 análisis (`D9.json`):**
  - bootstrap por conglomerado de nivel de extracción (una extracción es un mundo (celda, semilla), 10 000
    remuestreos, semilla 20261028) con columna de Holm;
  - igualdad de trazas;
  - el contrafactual nativo en **cada** cierre anticipado a q < 1: cuota destructiva, banda ±0.02 σ y
    arrepentimiento frente a O* en unidades de riesgo de caja inicial.
- **Duración total:** unas 36–40 h.

#### R — Informe

- **Salidas:**
  - `REPORT_F4.md`, `REPORT_F4.json` y gráficos PNG (Sharpe y beneficio por operación frente al MAE y frente al
    MAE/ingenuo de la escalera);
  - la lista «lo más probablemente erróneo» (residuos).
- **Cierre:** al final escribe la marca `$C/FINAL`.

### Cómo se ve el éxito y el fracaso

- **Éxito:**
  - existe `$C/FINAL`;
  - `STATUS.md` dice `FINAL: True`;
  - `$C/REPORT_F4.md` está completo.
- **Fallo:**
  - existe `$C/state/FAILED_<etapa>` con la hora, el sha y el motivo;
  - el conductor se ha detenido limpiamente;
  - el motivo también está en `$C/state/<etapa>.reason`.
  - Cada etapa tiene **un** reintento automático: el primer fallo reintenta y el segundo escribe `FAILED_<etapa>`.
    Los fallos tipados sin reintento son la puerta de degeneración, la propiedad de espejo y la falta de entradas.
- **Parada voluntaria:** con `STOP` presente, el conductor sale con código 0 y la etapa queda reanudable.

---

## (b) Comandos exactos

```bash
R=/var/tmp/behaviour-matrix-20261027; C=$R/chain_f4

# estado (un solo comando; regenera STATUS.md / STATUS.json desde archivos)
bash $C/status.sh
cat $C/STATUS.md                      # o STATUS.json para máquinas
tail -50 $C/logs/chain.log            # transiciones del conductor
tail -50 $C/logs/<etapa>.log          # salida de una etapa

# espejo local en la máquina del coordinador (sin nombres de máquina en este documento)
ssh <host-trabajo> 'bash /var/tmp/behaviour-matrix-20261027/chain_f4/status.sh'

# parar SUAVE (se lee entre tareas y entre etapas; el trabajo en curso termina su lote y queda reanudable)
touch $C/STOP

# reanudar
rm -f $C/STOP && systemctl --user start strategy-chain-f4.service

# reintentar una etapa que quedó FAILED (después de leer el motivo y decidir)
rm -f $C/state/FAILED_<etapa> $C/state/<etapa>.attempts && systemctl --user start strategy-chain-f4.service

# parada DURA (solo el dueño): el servicio y, si hace falta, el ámbito del trabajo en curso (STATUS.md lista sus nombres)
systemctl --user stop strategy-chain-f4.service
systemctl --user stop <nombre-del-ámbito-crispdm>.scope

# leer resultados
cat $C/work/S1R/FREEZE.json           # punto de niveles congelado, BOUNDARY_HIT, extensiones, inviables
cat $C/work/S2/CHAMPION.json          # campeón de la etapa 2
cat $C/work/S3/S3_DECISION.json       # decisión de la etapa 3
cat $C/work/FINAL_CHAMPION.json       # punto final
cat $C/REPORT_F4.md                   # informe final
```

Diagnóstico del servicio: `systemctl --user status strategy-chain-f4.service` y
`journalctl --user -u strategy-chain-f4.service`. El temporizador `strategy-chain-f4-status.timer` reescribe el estado cada
5 minutos.

---

## (c) Toma de control (Musashi o Retsu)

- **Código.** Rama `satoshi/expert-guided-rl-20261019`. El sha de la cadena está en `$C/CODE_SHA` y la etapa 1 adoptada
  corre en `5e04e83` (`$R/code_5e04e83`). Para instalar otra versión:
  `bash install.sh <C> <repo> <sha> <chain.env>`, desde `tools/clean_round_chain/chain_f4/`. El instalador no toca
  los resultados existentes.
- **Estado.**
  - `$C/state/<etapa>.{started,done,failed}` y `FAILED_<etapa>`: JSON con la hora UTC, el sha y los digests de
    entradas y salidas;
  - `$C/state/<etapa>.attempts` y `$C/state/<etapa>.reason`;
  - los resultados están en `$C/work/` y en la raíz de la etapa 1 adoptada.
- **Configuración.** `$C/chain.env`, que contiene topes de memoria, procesos, población/generaciones, semillas y rutas.
  Cámbiela solo con la cadena parada.
- **Reglas que nunca se rompen:**
  - **Nunca kill / pkill / killall.** Se para con `STOP`. La parada dura es del dueño, con `systemctl --user stop`.
  - **Memoria solo por `crispdm-run`.** Respete los topes y deje un margen de 3 GiB libre (`host_free_for_new` ≥
    pedido + 3 GiB). Nunca baje un tope rechazado ni renombre un trabajo para esquivarlo.
  - **TEST cerrado.** Las filas con fecha ≥ 2025-01-01 no se leen.
  - **Nunca se adapta un testigo a la estrategia.** Una forma de testigo solo cambia si su control designado falla la
    forma sellada, y se registra en el changelog.
  - **Aditividad.** `full_f3` y LEGACY quedan byte-idénticos, fijado por tests (`tests/test_strategy_params_v22.py`).
  - **Ni nombres de máquina, IPs, usuarios ni rutas personales en archivos versionados.**
  - **Commits.** Cada commit lleva `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, cuenta los tests después
    de ejecutarlos y se empuja después de cada commit.
- **Quién decide qué:**
  - El dueño decide el contrato, las preguntas abiertas de la sección (d) y las paradas duras.
  - El coordinador dicta las resoluciones sobre testigos y reglas.
  - Musashi es el auditor temporal.
  - La cadena solo aplica las reglas fijadas en este documento.

---

## (d) Residuos y decisiones abiertas del dueño

### Lo más probablemente erróneo (va a la solicitud de auditoría)

Fuente: `WITNESS_CHANGELOG`, entradas `audit_request: most likely wrong`.

- **W17 (sellado).**
  - Prohíbe aperturas con verdad positiva dentro de ventanas sembradas cuyo riesgo medido no es alto.
  - Las 26 aperturas de tier 2 fueron salidas TP con verdad positiva (neto 0.127–0.871 σ) y se decidieron con
    rv/σ 0.49–1.98.
  - En 9 de las 14 ventanas, rv/σ nunca superó 2 después del retardo; las otras 5 sí lo superan en otras barras.
  - Con el ratio 1.5 quedan 9 registros (5 mundos) con rv/σ 0.96 / 0.83 / 0.49 / 1.20 / 1.42. Tres están por debajo
    del límite inferior del gen, 1.2.
- **W8_old.** Cinco filos de cuchillo en 0.4024–0.4059 frente al 0.40 sellado: LHS_19 s2, LHS_43 s2,
  S_I1_upmedium_downstrong_late s0, S_I1_upstrong_downmedium_late s0 y S_LHS_25 s0.
- **W1.** Los gemelos M_/S_LHS_41 s0.
- **W4 (sellado).** M_H1_medium_fast s0 (0.571) y M_H6_LHS_17 s0 (0.714):
  - usan la misma regla y el mismo tick que O*, y A de la cabeza = A del camino;
  - el W4 sellado (remanente ≤ 0) es más estricto que la banda de empate de O*;
  - O* pasa solo porque mantiene otras operaciones.
- **Z1 SELL.** CONTROL_CONFLICT en M_LHS_36 s0 y S_H3_Lstrong_Mhigh_Shigh_with s0: el oráculo falla igual la forma
  corregida.
- **Conflicto de control W8 / W1 (tier 1 / tier 2).** LHS_50 s2 y otros listados en el informe de tier.

### Decisiones abiertas del dueño

1. **Criterio de la búsqueda.** Ahora es solo Sharpe (con Deb); la alternativa sería Sharpe más un término de calidad
   declarado. Solo Sharpe cambia actividad por Sharpe: con `z_veto → 0` el veto jerárquico se apaga. El informe pone la
   actividad (op/semana, reaperturas a 1 barra, σ por operación) y la cuota de activación del filtro de ráfagas en
   barras tranquilas junto al Sharpe.
2. **El valor de la jerarquía.**
   - Con cabezas de calidad 0.55–0.80, el brazo con vetos (z_veto 2) no superó al brazo sin vetos en 0–2 de 72 mundos.
   - Hasta que una calidad realista lo demuestre, el valor de la jerarquía no está probado. La DEAP decide `z_veto`.

---

## (e) Glosario

- **full_f4.** Perfil de ejecución del contrato de negocio compartido: apalancamiento 50, sin tope diario, corte de
  12 h, niveles absolutos por horizonte y tamaño por presupuesto de riesgo. `full_f3` es el perfil auditado anterior
  y sirve de control byte-idéntico.
- **Menú de decisión.** Las órdenes candidatas absolutas (h, lado), con h ∈ {3, 6, 12, 24}, construidas en la barra de
  decisión con los niveles de cada horizonte. Un horizonte sin nivel MT que lo confirme queda fuera del menú en esa
  barra. La verdad del motor es la reproducción exacta de ese menú.
- **O\*.** Oráculo de salida óptima: las aperturas del oráculo de verdad y el cierre cuando A(t) ≥ −EPS_CLOSE
  (0.02 σ). Es el control designado de los testigos de salida y tiempo.
- **Oráculo de mantener (hold oracle).** La mejor celda de la verdad; cierra solo si el valor remanente es negativo.
  Es el control designado del resto de testigos.
- **A(t).** Ventaja de cerrar ahora: valor de cerrar ahora menos la mejor continuación ejecutable, incluida la salida
  natural.
- **Testigos W0–W17.**
  - W0: actividad;
  - W1: lado;
  - W2: aperturas con verdad positiva;
  - W3: NOP;
  - W4: cierres justificados y salidas por SL;
  - W5: reversión;
  - W6: volumen;
  - W7: mejor ranura del menú;
  - W8: cambio de régimen (cierre en el cambio y dirección vieja después del retardo);
  - W9: pérdidas de hueco;
  - W10: ventana de spread;
  - W11: corte;
  - W12: datos faltantes;
  - W13: SL/TP;
  - W14: espejo;
  - W15/W16: riesgo, y SL+TP en cada orden;
  - W17: sin aperturas en ráfagas.
  Cada uno se juzga contra **un** control designado (VALIDITY_SPEC v2). El estado puede ser DEFECT (el control falla
  la forma sellada y gobierna la forma corregida), VALIDATED, SEALED_UNVALIDATED, NOT_EVALUABLE (nunca PASS) o
  CONTROL_CONFLICT (el control también falla la forma corregida: nunca PASS y se informa aparte).
- **Serie E (E1–E8).** Testigos de salida por episodios sembrados:
  - E1/E1′: cierre antes del SL;
  - E2/E2′: arrepentimiento;
  - E3: acuerdo tick a tick con O*;
  - E4: cierres destructivos;
  - E5: cierre anticipado frente a O*;
  - E6: espejo;
  - E7: retraso frente a P*;
  - E8: escalera de calidad del cerrador.
- **Z1.** Etiqueta ZigZag de salida frente a P*_pre.
- **Puertas.**
  - Prueba de propiedad de espejo en el sha exacto de lanzamiento: escribe `PROPERTY_PASSED_<sha>`.
  - Puerta de degeneración de la generación 0: la búsqueda se para sola si todos los candidatos fallan igual o si
    falla una restricción que ningún vector puede mover.
- **Deb.** Dominancia con restricciones: primero menos testigos fallidos y después mayor Sharpe.
- **BOUNDARY_HIT.** El punto congelado está en un límite sellado de `STRATEGY_PARAMS`. Se informa; la regla de borde
  no extiende.
- **SE.** Error estándar de la diferencia de Sharpe pareada por mundo (regla de la etapa 3).
- **Bootstrap por conglomerado de nivel de extracción.** Se remuestrean mundos (celda, semilla) completos. Holm ajusta
  los p-valores de la familia de comparaciones del paso de calificación.
