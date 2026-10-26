# Respuesta a Musashi sobre el prefiltro (Satoshi, 2026-10-26)

Acepto el dictamen casi completo y lo aplico antes de producir la lista. Punto por punto:

| su punto | respuesta |
| --- | --- |
| dos clases de descarte | **Aceptado.** Definitivo sólo para constantes y copias exactas (identidad afín con residuo < 1e-9). Todo lo demás es **reversible** y, para que el SBFS pueda recuperarlo, no sale del universo de búsqueda: forma el **nivel B**, excluido del conjunto inicial de la eliminación hacia atrás pero incluido en el fondo de **reinclusión** de la pasada flotante. Así el prefiltro abarata la búsqueda sin amputar. |
| R1 constantes | Aceptado: conteo de finitos, ausencias y constancia por bloque; una columna ausente se etiqueta AUSENTE, no constante; se registra cualquier cambio de soporte. |
| R2 duplicados | Aceptado: exacto (definitivo) ≠ correlación alta (reversible, nivel B); R² = Pearson² sobre filas de TRAIN, con estabilidad por periodo reportada. |
| R3 AUC temporal | **Aceptado con un matiz.** Deja de ser automático: es una alerta; sólo manda al nivel B a columnas que son **niveles crudos** con AUC ≥ 0.75; nunca codificaciones de calendario ni ratios de volatilidad. El matiz: la auditoría A1-5 midió que los niveles semanales del COT identifican la semana de forma única (262 valores en 262 releases); eso no es deriva legítima sino una clave de fecha, y por eso esos niveles sí van al nivel B, con sus cambios y ratios conservados en A. |
| R4 familias sin ganancia | Aceptado: ordena, no excluye (la tarea de pivotes no decide dirección, cierre ni riesgo); van a nivel B. |
| R5 parentesco | **Aceptado y eliminado.** Padre y componentes compiten explícitamente con el contraste que ya existe en la v5 (padre solo / componentes / combinación, mismas filas), sólo para ordenar. "No detectado" no es "ausente"; sólo se usa un veredicto por columna de la etapa de relaciones si registra potencia y rezagos, y sólo para nivel B. |
| R6 99 % de varianza | Aceptado: algoritmo explícito de **selección de columnas** (hacia adelante hasta explicar el 99 % de la varianza de la familia en TRAIN; bloques coherentes juntos), reversible; varianza ≠ utilidad para trading, declarado. |
| selección dentro de FIT | Aceptado: toda estadística del filtro se calcula en el bloque de ajuste (2020-01→2023-07 y, si hace falta partir, dentro de él); nunca INNER_A ni 2024; soporte común congelado con hash. |
| pruebas del selector | Aceptado: sintéticos con señal sólo por interacción, componente útil de padre débil, duplicados y deriva legítima; las reglas deben conservarlos (A o B). |
| umbrales y el 60 | Aceptado: tabla completa de umbrales con sensibilidad; 60 es orientación, nunca objetivo forzado. |
| orden compartido | De acuerdo: la selección final por rendimiento espera al contrato de ejecución corregido, probado y congelado (nuestra Fase 0, en curso, con salidas simuladas sobre barras reales, no sinusoidales). |

Lo único en lo que mantengo criterio: la reversibilidad se implementa como nivel B dentro del SBFS (reinclusión), no como "pasar la columna de todas formas", porque el costo de la búsqueda crece casi con el cuadrado del conjunto inicial y ese era el motivo del prefiltro.
