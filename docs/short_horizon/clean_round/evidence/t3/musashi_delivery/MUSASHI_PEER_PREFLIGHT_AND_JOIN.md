# T3: verificaciones adelantadas y entrega conjunta

Para Satoshi y el propietario,2026-10-23. Sin nuevos fits, inferencias ni TEST.
No modifica las colas activas ni la regla congelada de eleccion por grupo.

## Verificacion real de la mitad de Satoshi

`tools/t3_peer_preflight.py` se ejecuto sobre sus22 resultados, sin escribir
dentro del productor. Se comprobaron:

- Manifiesto compartido `t3v2_8875abf84a63` y listado tipado de22 ids pares/control.
- Comandos tipados y grupos contra el manifiesto, no solo marcas STATE/DONE.
- 157 archivos de entrada/codigo/grupos contra los pins de nuestra configuracion
  de referencia; tambien el conjunto exacto de modulos Python, sin extras.
- Las tres semillas20261019/20/21 y12 checkpoints no vacios por unidad.
- E3M, SCALER y PROBE: columnas consumidas en orden, escalas finitas, ventana13/2,
  layout de grupos, fixed_train, none, ZigZag, consolidated y contrato1 entrada/dia.
- Metricas nativas presentes/finitas segun su contrato; ausencia de discrepancias
  de soporte o de orden de columnas entre los22 resultados.
- Segunda lectura de cada resultado y nueva verificacion de entradas, para
  rechazar cambios durante la lectura.

Resultado: `ARTIFACTS_VERIFIED`,22/22. Las18 unidades nuestras ya completadas
tambien comparten exactamente esos soportes y116 columnas, y sus espejos fueron
verificados nuevamente por hash. Quedan3 unidades nuestras al chequeo.

## Limite del chequeo

Esto es una verificacion independiente de artefactos retenidos, no un recibo
historico firmado de ejecucion. No vuelve a entrenar ni ejecuta el forward de
los checkpoints, no recalcula metricas ni certifica causalidad de toda la cadena.
El replay sigue siendo un punto explicito de la auditoria final; no se sustituye
por la existencia de PROBE ni por hashes correctos. No se ha elegido extractor,
iniciado la unidad combinada, emitido RELEASE ni arrancado SAC por este chequeo.

## Continuacion automatica sin vigilancia

`tools/t3_joined_readiness.py` corre en un timer propio de bajo consumo. Mientras
nuestras21 unidades no esten completas, solo lee cobertura y sale. Si hay un
fallo terminal, informa `RECOVERY_REQUIRED`, nunca una entrega satisfactoria.

Al completarse la mitad propia sin fallos, repite el preflight de Satoshi y
verifica los43 ids exactos, sin duplicados:22 pares/control y21 impares.
Autentica los espejos propios, sus recibos, manifiestos, soportes, columnas y
semillas antes de producir `READY_FOR_T3_DECISION_AUDIT.json`. Copia ese informe
a la carpeta de entrega ya compartida y deshabilita su timer al entregar.
El informe no interpreta resultados financieros ni convierte un resultado
negativo en un fallo de ejecucion.

Satoshi recibe en esa misma carpeta el presente documento y el preflight
`PEER_PREFLIGHT.json`. Su siguiente accion sigue siendo aplicar `decide` por la
regla congelada y preparar la combinacion. La confirmacion combinada y el replay
permanecen separados de la comprobacion de disponibilidad de los43 resultados.
No duplicar las impares ni cambiar sus configuraciones al reunir los resultados.

## Pruebas y trazabilidad

128 pruebas focalizadas CPU aprobadas. Adiciones: propietario explicito para
validar pares/control sin ampliar el worker impar por defecto; rechazo de codigo
extra, soporte u orden de columnas diferente; rechazo de ids faltantes,
duplicados, recibos o manifiestos alterados; espera sin ejecutar comandos antes
del cierre; flags que impiden confundir disponibilidad con replay o release.
La nueva prueba de propietario fallo antes del cambio y pasa despues.

El primer tick real devolvio `WAITING_FOR_OWN_SHARD`,18/21; no se presenta la
entrega conjunta futura como ya observada. Las colas GPU siguen intactas.
