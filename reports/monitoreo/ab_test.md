# A/B testing de P1: champion contra challenger

Generado con `mlflow run . -e abtest` (`modeling/ab_test.py`). No editar a mano.

- **Modelo registrado:** `mibici-equipo18-p1-demanda`
- **A (`@champion`):** version 1, run `166442af41ea432bbe181ccd17a571f9`, entrenada con datos hasta 2026-03-31
- **B (`@challenger`):** version 3, run `cc9d2e9a7e79461a9cd1ce35f7fba836`, entrenada con datos hasta 2026-08-31 (reentrenamiento de la version 1)
- **Conjunto de evaluacion (el mismo para las dos):** P1 de 2026-09-01 a 2026-09-30, sin muestreo
- **Version de datos (`data_version`):** `06b0010a4627`
- **Estaciones conocidas:** las que tienen filas en el entrenamiento del champion (2024-10-01 a 2026-03-31); las demas son nuevas.

## Conjunto de evaluacion

| Grupo | Filas | Estaciones |
|---|---|---|
| todas | 336,912 | 468 |
| conocidas | 265,680 | 369 |
| nuevas | 71,232 | 99 |

## Metricas

Cambio = B - A; un cambio negativo en estas metricas favorece a B.

| Grupo | Metrica | A | B | Cambio | Cambio relativo |
|---|---|---|---|---|---|
| todas | mae | 0.7969 | 0.7951 | -0.0018 | -0.2% |
| todas | rmse | 1.5064 | 1.4959 | -0.0105 | -0.7% |
| todas | wape | 0.5708 | 0.5696 | -0.0013 | -0.2% |
| todas | devianza_poisson | 0.9622 | 0.9558 | -0.0064 | -0.7% |
| conocidas | mae | 0.8825 | 0.8808 | -0.0017 | -0.2% |
| conocidas | rmse | 1.6400 | 1.6281 | -0.0120 | -0.7% |
| conocidas | wape | 0.5422 | 0.5411 | -0.0011 | -0.2% |
| conocidas | devianza_poisson | 1.0103 | 1.0031 | -0.0072 | -0.7% |
| nuevas | mae | 0.4776 | 0.4755 | -0.0021 | -0.4% |
| nuevas | rmse | 0.8372 | 0.8357 | -0.0015 | -0.2% |
| nuevas | wape | 0.8980 | 0.8941 | -0.0039 | -0.4% |
| nuevas | devianza_poisson | 0.7829 | 0.7798 | -0.0032 | -0.4% |

## Bootstrap pareado por dia

10,000 remuestreos de los 30 dias del periodo (semilla 42).

| Diferencia MAE(A) - MAE(B) | IC 95% | Remuestreos en que B es mejor |
|---:|---|---:|
| +0.0018 | [+0.0010, +0.0026] | 100.0% |

## Decision

Regla fijada en `config.py` antes de ver los resultados: B gana solo si (1) el IC de MAE(A) -
MAE(B) queda completo por encima de 0 y (2) en las estaciones conocidas el MAE de B no es mas de
1% peor que el de A.

| Condicion | Cumple |
|---|---|
| (1) B mejor en MAE (IC arriba de 0) | si |
| (2) Sin regresion en conocidas (cambio -0.2%) | si |

**B gana: el intervalo de MAE(A) - MAE(B) queda arriba de 0 y no empeora en las conocidas.** Aliases: sin cambios; para promover a B: make abtest PROMOVER=si.

## MAE mensual del champion

Abr-jun es la validacion con que se eligio el champion; jul-sep, meses que no vio.

| Mes | Particion | Filas | MAE |
|---|---|---|---|
| 2026-04 | validacion | 266,400 | 0.7556 |
| 2026-05 | validacion | 275,280 | 0.7055 |
| 2026-06 | validacion | 275,472 | 0.7905 |
| 2026-07 | prueba | 316,224 | 0.7200 |
| 2026-08 | prueba | 338,496 | 0.6782 |
| 2026-09 | prueba | 336,912 | 0.7969 |
