# Escalera de capacidad por coordenadas (Satoshi), DESARROLLO

Orden núcleo → cabezas → ramas → profundidad; 3 semillas por brazo; elección solo en VAL A; VAL B reportado; 2024 no se lee dentro de la escalera.

| Arquitectura | Parámetros E2 (actor+crítico) | E1 Sharpe TRAIN / VAL A / VAL B | E2 Sharpe TRAIN | E2 Sharpe VAL A | E2 Sharpe VAL B | Brecha TRAIN − VAL A | DD % VAL A | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b2_c2_h0_d1_do0.1_wd0.0001 | — | -1.92 / -1.06 / -3.13 | -0.56 | 0.28 | -0.70 | -0.84 | 39.7 | sí |
| b2_c8_h0_d1_do0.1_wd0.0001 | 2240 | -0.48 / -3.73 / -3.42 | -0.17 | 0.24 | -0.11 | -0.41 | 34.1 | sí |

## Etapas

- **S1_core** (en curso): titular antes b2_c2_h0_d1_do0.1_wd0.0001
  - b2_c8_h0_d1_do0.1_wd0.0001: VAL A 0.24, TRAIN -0.17, VAL B -0.11; criterios {"improve": false, "paired": false, "pairs_better": 2, "gap": true, "risk": true, "valid": true, "eligible": false}
