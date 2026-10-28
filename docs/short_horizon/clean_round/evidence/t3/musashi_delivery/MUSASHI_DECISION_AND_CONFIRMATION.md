# T3:43 unidades cerradas, decision y confirmacion en curso

Para Satoshi y el propietario,2026-10-23. No duplicar la unidad C:ya esta
entrenando bajo nuestro supervisor en el worker5090. No se ha iniciado SAC.

## Cierre real y regla aplicada

Nuestras21 impares terminaron sin fallos terminales; las22 pares/control de
Satoshi tambien estan disponibles y verificadas. El timer entrego
`READY_FOR_T3_DECISION_AUDIT.json`,43 ids exactos,116 columnas, soportes y semillas
pareados. Los intentos fallidos historicos no se borraron ni se contaron como
mediciones. Los espejos propios y sus replicas se verificaron por hash.

Se reunieron los resultados por referencia, sin editar E3M ni sus recibos,
y se ejecuto `expert_guided_t3_manifest.py decide` del codigo1aaf367 con la regla
congelada. Decision:
`edb331ef24629b5331b524e76f9f213c3464a8eac72b57e1d75f4d34d5d47c0c`.

| Grupo | Tipo elegido | Sharpe INNER_A medio |
| --- | --- | ---: |
| Control:todos los grupos Conv1D | referencia | 0.779954 |
| cot.eur | per_feature | 1.368338 |
| gmm/logret_1+rv_24+log_range | pointwise | 2.731758 |
| fracdiff/log_close | pointwise | 1.973875 |

Los otros22 grupos conservan conv. Son contrastes individuales condicionados a
las demas entradas, no el rendimiento de un sistema que combina las3 mejoras.
Las3 alternativas pasan la condicion de validez, ventaja sobre ruido entre
semillas y brecha de generalizacion de la regla original. No se eligio usando
OBS_2024 ni TEST. Las expectativas T1 acertaron9/17:se reporta, no decide.

## Replay real de checkpoints

Se reconstruyeron las ventanas, se comprobaron las constantes guardadas y se
cargaron los12 checkpoints por unidad sin entrenar. En los mismos8 origenes
retenidos por PROBE, los43 experimentos reprodujeron exactamente logits y
regresiones:516 checkpoints. Evidencia:`NATIVE_REPLAY_ALL43.json`, entregada
en la carpeta compartida y copiada al almacenamiento de experimentos.

Se conserva tambien el diagnostico que no paso. El primer replay en otro
worker uso8 hilos frente al1 del launcher y dio diferencias; al igualar hilos,
las9 unidades entrenadas en ese worker fueron exactas, pero las de otras CPUs
todavia diferian. El maximo absoluto observado fue5.364418029785156e-6.
Al repetir en las CPUs originales con el entorno del launcher, pasaron las12
unidades del worker5090 y las22 del worker4090, junto con las9 del4070.
No se ajustaron tolerancias, pesos, sondas ni metricas para hacerlas pasar.

Esto demuestra recarga/forward en el contexto original; no promete identidad
binaria entre CPUs distintas ni es por si solo un replay de las metricas de
trading. Ademas, el replay de trading completo del control ya termino en su CPU
original, leyendo sus arbitros/umbrales congelados, sin volver a elegirlos.
Sus9 rollouts (3 semillas x TRAIN,INNER_A,OBS_2024) reprodujeron exactamente
todas las metricas nativas, incluido profit,Sharpe,drawdown,ordenes y aciertos,
con el mismo simulador y contrato1/dia,1:50. Evidencia:`CONTROL_FULL_REPLAY.json`,
estado PASS, entregada a Satoshi. No se uso TEST ni se entreno en el replay.

## Confirmacion C ya lanzada

El `combine` original genero el archivo con las3 elecciones:
`a9d878cf8c63a5b0ecf13f3e3a664e020ae70cc9cf2fa2d252f1643b227f76ef`.
El comando de C solo cambia `--groups` y `--out` respecto a la unidad tipada
original. Datos, columnas, ventana13/2, unidades2, profundidad2, cabezal16,
semillas20261019/20/21, dropout0.1,wd1e-4, max_epochs1000, escalado fijo TRAIN,
nonstationary diff, consolidated, etiquetas ZigZag, sin feedback y contrato
de negocio permanecen iguales. No hay cambio de DOIN ni lectura de TEST.

Configuracion operacional C:
`7ffebafb8253936ed6100b6ca4dd9aff53243b7a763189b39320777f7b4257d1`.
Worker5090,5GiB por trabajo, UUID fijado en privado, solo un proceso GPU pesado.
El chequeo real confirmo un proceso CUDA y la primera semilla completada.

El wrapper conserva los intentos y sus reintentos acotados, verifica la receta
combinada contra la decision y rechaza soporte/orden de columnas distintos al
control. La prueba de regresion demuestra que solo cambian las dos opciones
de ruta del comando; una decision o archivo de grupos alterado se rechaza.

## Cadena automatica siguiente

1. Terminar las3 semillas de C, validar12 checkpoints/scaler/sondas y sellar.
2. Colector propio:copiar al volumen de experimentos y verificar replica fisica.
3. Aplicar la regla combinada congelada:validas las3 semillas; media INNER_A de
   C no inferior al control por mas de la dispersion agrupada. Esto comprueba
   interacciones; no impone una regla nueva despues de mirar resultados.
4. Reportar por separado TRAIN,INNER_A,OBS_2024, brecha y las metricas nativas
   de trading de las3 semillas, incluidas ganancias,ordenes,aciertos y drawdown.
5. Replay de sondas y de trading completo de C en su CPU original y con el
   entorno del launcher. Carga pesos/arbitro/umbral guardados:no refit ni tuning.
6. Entregar `COMBINATION_DECISION.json` y `COMBINATION_REPLAY.json` a Satoshi;
   deshabilitar los timers propios al cerrar. Un fallo real se publica como tal.

No se emite RELEASE ni se sustituye la solicitud de auditoria final por estos
recibos. Satoshi puede preparar esa solicitud con este cierre, las replicas,
el replay43 y el resultado C cuando llegue; no necesita repetir nuestros fits.
La linea de expertos sigue siendo suya y SAC sin experto sigue siendo nuestra.

## Pruebas y estado de metodo

Las regresiones nuevas cubren el comando de C, union de grupos desde la decision,
identidades, regla de confirmacion, ausencia de OBS_2024 en su eleccion, espera
sin lanzar trabajo antes del cierre y fallos terminales sin metricas inventadas.
El conjunto focalizado pasa134 pruebas CPU. La comprobacion real de replay
reconstruyo645 archivos sellados y ejecuto516 forwards, sin entrenamiento.
La confirmacion y sus rollouts finales siguen en curso:no se declara cerrada.
