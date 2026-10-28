# Barrido de complejidad del tronco temporal (E1-T y E2-T), DESARROLLO

Medias de 3 semillas (mín–máx entre paréntesis para el Sharpe). Sin lectura de 2024. Variables: las 15 de Musashi. Elección por VAL A; VAL B confirma.

## E1-T BC+DAgger

| u | Parámetros | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Brecha TRAIN − VAL A | Ganancia VAL A | Ganancia VAL B | DD % VAL B | Órdenes/sem VAL B | Válida VAL A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 388 | -1.92 (-2.09…-1.80) | -1.06 (-2.44…0.14) | -3.13 (-4.03…-2.42) | -0.87 | -2 981.2 | -5 214.1 | 55.0 | 13.6 | sí |
| 4 | 1100 | -0.85 (-1.50…-0.49) | -3.55 (-4.40…-2.57) | -5.08 (-7.53…-3.13) | 2.70 | -6 180.6 | -6 794.4 | 72.2 | 15.0 | sí |

Elección por VAL A: u = 2. Marcador de inicio de sobreajuste (TRAIN sube y VAL A baja): u = 4.

## E2-T SAC

| u | Parámetros | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Brecha TRAIN − VAL A | Ganancia VAL A | Ganancia VAL B | DD % VAL B | Órdenes/sem VAL B | Válida VAL A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 740 | -0.67 (-1.04…-0.38) | 0.58 (-0.61…1.87) | -0.25 (-1.16…0.33) | -1.26 | 659.2 | -1 231.2 | 39.4 | 4.0 | sí |

Elección por VAL A: u = 2. Marcador de inicio de sobreajuste (TRAIN sube y VAL A baja): u = None.

## E2-T SAC+demos

| u | Parámetros | Sharpe TRAIN | Sharpe VAL A | Sharpe VAL B | Brecha TRAIN − VAL A | Ganancia VAL A | Ganancia VAL B | DD % VAL B | Órdenes/sem VAL B | Válida VAL A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 740 | -0.44 (-0.62…-0.33) | -0.02 (-1.12…0.72) | -1.14 (-2.49…-0.07) | -0.42 | -865.8 | -3 204.1 | 53.5 | 3.7 | sí |

Elección por VAL A: u = 2. Marcador de inicio de sobreajuste (TRAIN sube y VAL A baja): u = None.

