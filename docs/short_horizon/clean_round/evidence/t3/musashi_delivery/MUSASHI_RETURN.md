# T3 repartido entre dos workers

## Actualizacion: rebalanceo del tramo final

Posterior a un nuevo reinicio del worker4070, se verificaron aproximadamente
24GiB disponibles, swap sin uso y ausencia de procesos GPU de entrenamiento.
Su primera asignacion ya habia terminado:7/7 contando la unidad01 preservada.
El worker5090 completo8 unidades (15..29 impares); la29 termino en su segundo
intento despues de un fallo CUDA previo al reinicio. El kernel del arranque
anterior registro Xid79 y solicitud de reinicio; no se infiere la causa fisica.
No hubo nuevos Xid en el chequeo posterior. Los intentos previos se conservan.

Para aprovechar la GPU libre sin repetir resultados:

| Cola | Asignacion restante | Estado al desplegar |
| --- | --- | --- |
| Worker5090 original | unidad31 | hijo cientifico preservado y activo |
| Worker5090 sucesora | 33,35,37; tambien31 solo si no se completa | relevo automatico preparado |
| Worker4070 sucesora | 39,41 | unidad39 entrenando en GPU |

Se pauso exclusivamente el despachador original5090, con identidad de proceso
verificada, y se deshabilito su timer. Su hijo cientifico no recibio senales.
El timer de relevo espera la salida del hijo y del launcher, sella la31 con su
configuracion original y solo entonces inicia la sucesora5090. Si se interrumpe,
conserva el intento y recupera31 en la sucesora; nunca repite una31 ya sellada.
La antigua cola no despacha39/41 mientras el worker4070 las ejecuta. No se editan
configuraciones cientificas ni recibos existentes.

La sucesora4070 conserva5GiB y registra configuracion
`52af12883fd757f88289f5e216fd7eebda943f8e7b6848a4ff80c4302ab53dc8`.
La5090 sucesora se configura despues de terminar31: su identidad se publica en
el recibo de relevo, no se predice antes de disponer del snapshot del origen.

Las dos sucesoras tienen colectores y replicas separados. El indice consolidado
incluye las nuevas raices, autentica recibos y rechaza duplicados; sigue sumando
exactamente21 unidades, no26. Las replicas logicas nuevas para Satoshi son
`expert-guided-musashi-t3-tail-slow-replica-20261023` y
`expert-guided-musashi-t3-tail-fast-replica-20261023`. El indice entrega la ruta
exacta por unidad, sin reubicar recibos entre configuraciones. El colector viejo
5090 se retira automaticamente cuando termina la cobertura global; los nuevos
se cierran despues de autenticar sus cierres nativos.

Pruebas:116 focalizadas CPU aprobadas. Las regresiones nuevas verifican el
relevo generico31 completo/interrumpido, destinos canonicos sin duplicados y
la inclusion autenticada de shards adicionales. PRE:3 pruebas nuevas fallaron
antes del cambio. POST:las variantes y todas las pruebas existentes pasan.
Comprobacion operacional:15/21 completas, ambas GPUs con procesos cientificos,
unidad31 preservada y unidad39 iniciada. El cierre futuro del relevo no se
presenta como observado; lo ejecuta su timer. No cambia ningun dato, semilla,
extractor, regularizacion, mascara, escala, split, metrica ni presupuesto.

## Retorno original del reparto

Para Satoshi y el propietario,2026-10-23. Estado actual posterior al reinicio
del worker5090. Sustituye el reparto de21 unidades exclusivamente al worker4070,
sin invalidar ni borrar sus intentos anteriores.

## Reparto y comprobacion real

El reinicio aumento MemAvailable del worker5090 de aproximadamente7.2GiB a12GiB,
con swap sin usar y PSI en cero. Las dos GPUs ya ejecutan procesos cientificos:
la5090 entrena `t3_unit_15` y la4070 ya paso automaticamente a `t3_unit_03`.
La unidad01 termino sus tres semillas, checkpoints/scaler/sondas y recibo sin
repetirse: costo nativo3235.0s. Cobertura al chequeo:1/21 completo,0 fallos en las
colas actuales; los cinco fallos historicos del primer intento siguen retenidos.

| Propietario operacional | Unidades asignadas | Cantidad | Presupuesto |
| --- | --- | ---: | --- |
| Worker5090 | ids15,17,19,...,41 | 14 | 5GiB por trabajo |
| Worker4070 | ids01,03,05,07,09,11,13 | 7 | unidad01 completo con4GiB; seis restantes5GiB |
| Satoshi, worker4090 | Sus ids pares/control originales | 22 | Sin cambios de nuestra parte |

Las primeras dos filas son particiones disjuntas que cubren exactamente nuestras
21 unidades impares. No se cambia el manifiesto cientifico de43 unidades ni la
identidad de ningun comando tipado; cada configuracion operacional guarda su
subconjunto explicito. Todos cargan strict116, por lo que la etiqueta del grupo
no prueba un consumo menor de RAM. La5090 recibe mas unidades y los contrastes
per_feature finales, sin inventar umbrales de memoria por nombre de familia.

## Margen de memoria y politica intacta

Los picos retenidos en los incidentes originales llegaron a4025729024 bytes,
unos3.75GiB. Se redondea hacia arriba1.25 veces ese pico a5GiB. Es un presupuesto
operacional con margen, no un pico completo demostrado: los intentos retenidos
fueron truncados por presion. El admisor sigue reservando3GiB para el escritorio,
con MemoryHigh al90%, MemoryMax, sin swap del trabajo y guardias PSI intactos.
No se deshabilitan servicios del propietario ni se modifica el sistema para
forzar admision. Un solo proceso cientifico pesado por host; la5070Ti no se usa.

La codificacion, datos, ventana13/2, T3 grupos por unidad, codigo1aaf367, modelo
2/2/16, semillas20261019/20/21, regularizacion base0.1/1e-4, split consolidated,
etiquetas ZigZag, ausencia de feedback y contrato de ejecucion permanecen fijos.
Cambian particion operacional, presupuesto y hardware, registrados en recibos.

Configuracion nueva de14 unidades del worker5090:
`83419a98f57a1b701345069845a17cc60f4ea181f8808ca8054a714c3222821f`.
La cola sucesora4070 ya arranco con seis unidades, configuracion:
`8382a3e8b11cc570f178b3fc018cfefbc0d7340c157f225d640680dddd5910a3`.

## Conservar el entrenamiento ya avanzado

El despachador original4070 se pauso con SIGSTOP, despues de verificar
PID/comando/start_ticks y la raiz exacta. No se detuvo ni senalizo su hijo
cientifico, el launcher de admision ni el guardia de memoria: la unidad01 termino
normalmente. Se deshabilito solo su timer de reinicio para que no
vuelva a despachar el lote completo de21 unidades.

El relevo automatico consulta el hijo cientifico y el launcher. Mientras viven,
no sella ni mueve outputs, no los reentrena y no inicia otra tarea4070. Cuando
terminen, valida y sella los artefactos reales con la configuracion original;
despues termina ese despachador y arranca la cola4070 restante. Una interrupcion
sin resultado completo conserva el intento y encola01 de nuevo con5GiB; no lo
convierte en una medicion. Nunca repite una unidad con recibo valido.

El relevo real ya paso: primera unidad sellada y copiada; antiguo despachador
terminado; unidad03 entrenando con5GiB. La transicion deja recibos separados,
copia del ledger previo y configuracion
padre. No modifica pesos ni metricas para adoptar el primer resultado. Un corte
electrico antes del relevo no reinicia el despachador antiguo: el timer de relevo
retoma el procedimiento y conserva o reintenta el primer caso segun sus archivos.

## Automatizacion y custodia

Dos colas secuenciales, locks y recibos por unidad; timer de relevo propio y
cobertura consolidada de21. Se rechazan ids ajenos al reparto, subconjuntos
vacios/duplicados y resultados duplicados entre origenes. El indice comprueba
hashes con el verificador de espejos, no recalcula metricas ni elige extractores.

Los colectores verifican y replican cada cola por separado. El primer resultado
4070 conserva su directorio/recibo originales; su ultima copia se fuerza antes
de detener el colector antiguo. La cola4070 sucesora activa su propio colector
al arrancar. La nueva cola5090 ya tiene su replica comprobada con0 completados.
La cobertura consolidada se copia al worker de Satoshi cuando cambia; los timers
de cada cola y el de cobertura se desactivan al alcanzar sus estados terminales.

POST:112 pruebas focalizadas CPU aprobadas, incluyendo el relevo con resultado
completo/incompleto, no senalizar al hijo activo, particion21=14+7, rechazo de
duplicados y no repetir recibos completados. Ademas de los casos sinteticos, se
verifico el primer relevo real y sus copias. Esto no equivale a haber cerrado
las20 unidades restantes.

## Entrega para Satoshi

No duplicar nuestras21 impares. Sustituir el consumo de la unica raiz vieja por:

1. Primera unidad completada y preservada: replica logica
   `expert-guided-musashi-t3-relocated-replica-20261023`.
2. Cola5090 de14: `expert-guided-musashi-t3-parallel-fast-replica-20261023`.
3. Cola sucesora4070: `expert-guided-musashi-t3-parallel-slow-replica-20261023`.

La cobertura llega a `musashi-t3-parallel-delivery-20261023/COVERAGE.json` en tu
worker. Cada fila liga unidad/origen/recibo/hash/configuracion/output; conserva
las configuraciones de origen al unir los resultados. No copies un recibo de
una configuracion a otra ni reescribas E3M.json para unificar carpetas.

Tu decision por grupo y combinacion sigue usando la regla congelada y los mismos
soportes/semillas. Anexa el hardware por unidad. No hay nuevo SAC, RELEASE global,
cambio DOIN, lectura TEST ni servicio de trading en este reparto.
