# Escalera de capacidad por coordenadas (Satoshi), DESARROLLO

Orden núcleo → cabezas → ramas → profundidad; 3 semillas por brazo; elección solo en VAL A; VAL B reportado; 2024 no se lee dentro de la escalera.

| Arquitectura | Parámetros E2 (actor+crítico) | E1 Sharpe TRAIN / VAL A / VAL B | E2 Sharpe TRAIN | E2 Sharpe VAL A | E2 Sharpe VAL B | Brecha TRAIN − VAL A | DD % VAL A | Válida |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b2_c2_h0_d1_do0.1_wd0.0001 | — | -1.92 / -1.06 / -3.13 | -0.56 | 0.28 | -0.70 | -0.84 | 39.7 | sí |
| b2_c8_h0_d1_do0.1_wd0.0001 | 2240 | -0.48 / -3.73 / -3.42 | -0.17 | 0.24 | -0.11 | -0.41 | 34.1 | sí |
| b2_c16_h0_d1_do0.1_wd0.0001 | 4912 | 2.00 / -3.48 / -4.46 | -0.31 | -0.12 | -0.52 | -0.20 | 40.3 | sí |
| b2_c2_h8_d1_do0.1_wd0.0001 | 1040 | -1.34 / -2.37 / -2.52 | -0.19 | 0.62 | -1.06 | -0.80 | 35.9 | sí |
| b2_c2_h16_d1_do0.1_wd0.0001 | 1440 | -1.05 / -3.46 / -2.68 | 0.03 | 0.93 | -0.58 | -0.90 | 35.6 | sí |
| b4_c2_h16_d1_do0.1_wd0.0001 | 2040 | -1.46 / -1.87 / -1.47 | 0.13 | 0.90 | -1.42 | -0.77 | 37.7 | sí |
| b8_c2_h16_d1_do0.1_wd0.0001 | 3240 | -2.34 / -3.91 / -4.98 | -0.27 | 0.71 | -1.79 | -0.98 | 34.3 | sí |
| b2_c2_h16_d2_do0.1_wd0.0001 | 1468 | 0.10 / -2.04 / -1.78 | -0.17 | 0.70 | -0.03 | -0.87 | 36.2 | sí |

## Etapas

- **S1_core**: titular antes b2_c2_h0_d1_do0.1_wd0.0001 → después b2_c2_h0_d1_do0.1_wd0.0001 (promovido: False)
  - b2_c8_h0_d1_do0.1_wd0.0001: VAL A 0.24, TRAIN -0.17, VAL B -0.11; criterios {"improve": false, "paired": false, "pairs_better": 2, "gap": true, "risk": true, "valid": true, "eligible": false}
  - b2_c16_h0_d1_do0.1_wd0.0001: VAL A -0.12, TRAIN -0.31, VAL B -0.52; criterios {"improve": false, "paired": false, "pairs_better": 2, "gap": true, "risk": true, "valid": true, "eligible": false}
- **S2_head**: titular antes b2_c2_h0_d1_do0.1_wd0.0001 → después b2_c2_h16_d1_do0.1_wd0.0001 (promovido: True)
  - b2_c2_h8_d1_do0.1_wd0.0001: VAL A 0.62, TRAIN -0.19, VAL B -1.06; criterios {"improve": true, "paired": false, "pairs_better": 3, "gap": true, "risk": true, "valid": true, "eligible": false}
  - b2_c2_h16_d1_do0.1_wd0.0001: VAL A 0.93, TRAIN 0.03, VAL B -0.58; criterios {"improve": true, "paired": true, "pairs_better": 5, "gap": true, "risk": true, "valid": true, "eligible": true}
- **S3_branch**: titular antes b2_c2_h16_d1_do0.1_wd0.0001 → después b2_c2_h16_d1_do0.1_wd0.0001 (promovido: False)
  - b4_c2_h16_d1_do0.1_wd0.0001: VAL A 0.90, TRAIN 0.13, VAL B -1.42; criterios {"improve": false, "paired": true, "pairs_better": 4, "gap": true, "risk": false, "valid": true, "eligible": false}
  - b8_c2_h16_d1_do0.1_wd0.0001: VAL A 0.71, TRAIN -0.27, VAL B -1.79; criterios {"improve": false, "paired": false, "pairs_better": 3, "gap": true, "risk": true, "valid": true, "eligible": false}
- **S4_depth**: titular antes b2_c2_h16_d1_do0.1_wd0.0001 → después b2_c2_h16_d1_do0.1_wd0.0001 (promovido: False)
  - b2_c2_h16_d2_do0.1_wd0.0001: VAL A 0.70, TRAIN -0.17, VAL B -0.03; criterios {"improve": false, "paired": false, "pairs_better": 3, "gap": true, "risk": true, "valid": true, "eligible": false}

Finalista: **b2_c2_h16_d1_do0.1_wd0.0001**.
