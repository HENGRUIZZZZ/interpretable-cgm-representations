# v18 Baselines vs Ours by Standard Meal

| Meal        | Model             |     SSPG ρ |   SSPG RMSE |       DI ρ |   DI RMSE |
|:------------|:------------------|-----------:|------------:|-----------:|----------:|
| Cornflakes  | Wang(Exp1)        | nan        |     80.1368 | nan        |  0.761466 |
| PB_sandwich | Wang(Exp1)        | nan        |     80.1368 | nan        |  0.761466 |
| Protein_bar | Wang(Exp1)        | nan        |     80.1368 | nan        |  0.761466 |
| Cornflakes  | Ours(Exp8)        |   0.109774 |     78.721  |  -0.5      |  1.36433  |
| PB_sandwich | Ours(Exp8)        |  -0.126316 |     84.0093 |  -0.391176 |  1.35252  |
| Protein_bar | Ours(Exp8)        |  -0.142857 |     81.2083 |  -0.694118 |  1.35956  |
| Cornflakes  | Metwally(Exp2)    |   0.747368 |     69.4347 |   0.588235 |  0.709671 |
| Cornflakes  | SimpleStats(Exp4) |   0.810526 |     66.5261 |   0.75     |  0.805862 |
| Cornflakes  | Healey(Exp3)      |   0.4      |     72.7735 |   0.414706 |  0.836783 |
| PB_sandwich | Metwally(Exp2)    |   0.377444 |     78.4769 |   0.564706 |  0.799811 |
| PB_sandwich | SimpleStats(Exp4) |   0.251128 |     81.1462 |   0.485294 |  0.896319 |
| PB_sandwich | Healey(Exp3)      |   0.505263 |     64.8534 |   0.394118 |  0.881895 |
| Protein_bar | Metwally(Exp2)    |   0.437594 |     76.4699 |   0.376471 |  0.80891  |
| Protein_bar | SimpleStats(Exp4) |   0.56391  |     72.5711 |   0.644118 |  0.956338 |
| Protein_bar | Healey(Exp3)      |   0.46015  |     71.1069 |   0.220588 |  0.897646 |
