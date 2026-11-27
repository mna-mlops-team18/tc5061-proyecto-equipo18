# Monitoreo de P1: drift con Evidently

Generado con `python -m monitoreo.drift` (ambiente `.venv-monitoreo`). No editar a mano.

- **Version de datos (`data_version`):** `06b0010a4627`
- **Evidently:** 0.7.23. Numericas: `wasserstein` (distancia de Wasserstein normalizada); categoricas: `jensenshannon` (distancia de Jensen-Shannon). Una variable deriva si su distancia es >= 0.1; hay drift del conjunto si deriva al menos el 50% de las variables.
- **Muestra:** hasta 200,000 filas por conjunto (semilla 42). Reportes HTML en `reports/monitoreo/drift_<comparacion>.html` (no se versionan).

## Conjuntos comparados

| Comparacion | Referencia | Actual | Solo conocidas | Filas ref. | Filas actual | Estaciones comparadas (ref. / actual) | Estaciones nuevas en el periodo actual (fraccion de sus filas) | Variables con drift |
|---|---|---|---|---:|---:|---|---|---|
| principal | 2024-10-01 a 2026-03-31 | 2026-07-01 a 2026-09-30 | no | 4,692,984 | 991,632 | 370 / 469 | 99 (17.7%) | 4 de 17 (24%); drift del conjunto: no |
| sin_nuevas | 2024-10-01 a 2026-03-31 | 2026-07-01 a 2026-09-30 | si | 4,692,984 | 816,144 | 370 / 370 | 99 (17.7%) | 1 de 17 (6%); drift del conjunto: no |
| misma_temporada | 2025-07-01 a 2025-09-30 | 2026-07-01 a 2026-09-30 | si | 812,544 | 816,144 | 368 / 370 | 99 (17.7%) | 0 de 17 (0%); drift del conjunto: no |

## Drift por variable

Puntaje de Evidently en cada comparacion (en **negritas** si deriva). La causa probable lee la escalera de comparaciones: si deriva en `principal` pero no en `sin_nuevas`, la explican las estaciones nuevas; si deriva en `sin_nuevas` pero no en `misma_temporada`, es la temporada; si deriva en `misma_temporada`, cambio de un anio a otro la red que ya existia. Es una guia: se confirma con las graficas. El cambio se mide en la comparacion principal.

| Variable | Metodo | principal | sin_nuevas | misma_temporada | Causa probable | Cambio |
|---|---|---:|---:|---:|---|---|
| `mes` | jensenshannon | **0.668** | **0.668** | 0.002 | temporada | mayor cambio de proporcion en 9: 5.8% -> 34.1% |
| `salidas_media_7d` | wasserstein | **0.136** | 0.028 | 0.030 | estaciones nuevas | media 1.36 -> 1.19 (-12%); ceros 0% -> 0% |
| `longitud` | wasserstein | **0.125** | 0.009 | 0.009 | estaciones nuevas | media -103.36 -> -103.36 (-0%); ceros 0% -> 0% |
| `salidas_media_24h_previa` | wasserstein | **0.113** | 0.023 | 0.038 | estaciones nuevas | media 1.36 -> 1.21 (-11%); ceros 1% -> 1% |
| `capacidad` | wasserstein | 0.091 | 0.007 | 0.002 | estable | media 16.92 -> 17.05 (+1%); ceros 0% -> 0% |
| `salidas_lag_336h` | wasserstein | 0.087 | 0.024 | 0.009 | estable | media 1.36 -> 1.14 (-16%); ceros 53% -> 58% |
| `salidas_media_misma_hora_7d` | wasserstein | 0.080 | 0.009 | 0.015 | estable | media 1.35 -> 1.20 (-11%); ceros 26% -> 28% |
| `salidas_lag_168h` | wasserstein | 0.075 | 0.009 | 0.005 | estable | media 1.36 -> 1.18 (-13%); ceros 53% -> 57% |
| `salidas_lag_48h` | wasserstein | 0.061 | 0.011 | 0.018 | estable | media 1.35 -> 1.21 (-11%); ceros 53% -> 56% |
| `salidas_lag_24h` | wasserstein | 0.059 | 0.013 | 0.020 | estable | media 1.36 -> 1.22 (-10%); ceros 53% -> 56% |
| `latitud` | wasserstein | 0.059 | 0.005 | 0.008 | estable | media 20.68 -> 20.68 (-0%); ceros 0% -> 0% |
| `salidas` | wasserstein | 0.057 | 0.015 | 0.024 | estable | media 1.36 -> 1.23 (-10%); ceros 53% -> 56% |
| `llegadas_lag_24h` | wasserstein | 0.052 | 0.018 | 0.022 | estable | media 1.35 -> 1.22 (-10%); ceros 54% -> 57% |
| `es_festivo` | jensenshannon | 0.040 | 0.041 | 0.001 | estable | mayor cambio de proporcion en 0: 97.4% -> 98.9% |
| `dia_semana` | jensenshannon | 0.009 | 0.010 | 0.013 | estable | mayor cambio de proporcion en 2: 14.3% -> 15.1% |
| `hora_dia` | jensenshannon | 0.004 | 0.005 | 0.004 | estable | mayor cambio de proporcion en 18: 4.1% -> 4.2% |
| `es_fin_semana` | jensenshannon | 0.000 | 0.002 | 0.000 | estable | mayor cambio de proporcion en 1: 28.4% -> 28.4% |

## Calidad de los datos (conjunto actual, comparacion principal)

| Revision (DataSummaryPreset) | Valor |
|---|---:|
| Filas con valores repetidos | 239 |
| Columnas duplicadas | 0 |
| Columnas casi constantes | 1 |
| Filas vacias | 0 |
| Columnas vacias | 0 |
| Columnas constantes | 0 |
| Valores faltantes | 0 |

Las filas con valores repetidos no son registros duplicados: la llave `(estacion_id, hora)` es unica (lo revisa la validacion de datos). Son horas distintas con exactamente las mismas variables; lo esperable es que sean horas sin viajes de una misma estacion (rezagos y medias en cero).

Pruebas automaticas de Evidently (sin las de drift) que no pasaron en la comparacion principal: 26. Casi todas comparan un estadistico del conjunto actual contra la referencia con +-10 % de tolerancia, asi que repiten el drift con otras palabras.

| Estado | Prueba | Detalle |
|---|---|---|
| FAIL | Duplicated row count in dataset: Equal 52.000 +- 5.200 | Duplicated row count in dataset: Actual value 239.000 , but expected 52.000 +- 5.200 |
| FAIL | Std value of 'longitud': Equal 0.027 +- 0.003 | Std value of 'longitud': Actual value 0.031 , but expected 0.027 +- 0.003 |
| FAIL | Maximum value of 'salidas_lag_24h': Equal 73.000 +- 7.300 | Maximum value of 'salidas_lag_24h': Actual value 137.000 , but expected 73.000 +- 7.300 |
| FAIL | Mean value of 'salidas_lag_24h': Equal 1.356 +- 0.136 | Mean value of 'salidas_lag_24h': Actual value 1.220 , but expected 1.356 +- 0.136 |
| FAIL | Maximum value of 'salidas_lag_48h': Equal 74.000 +- 7.400 | Maximum value of 'salidas_lag_48h': Actual value 98.000 , but expected 74.000 +- 7.400 |
| FAIL | Mean value of 'salidas_lag_48h': Equal 1.353 +- 0.135 | Mean value of 'salidas_lag_48h': Actual value 1.209 , but expected 1.353 +- 0.135 |
| FAIL | Maximum value of 'salidas_lag_168h': Equal 106.000 +- 10.600 | Maximum value of 'salidas_lag_168h': Actual value 82.000 , but expected 106.000 +- 10.600 |
| FAIL | Mean value of 'salidas_lag_168h': Equal 1.357 +- 0.136 | Mean value of 'salidas_lag_168h': Actual value 1.179 , but expected 1.357 +- 0.136 |
| FAIL | Quantile 0.75 of 'salidas_lag_168h': Equal 2.000 +- 0.200 | Quantile 0.75 of 'salidas_lag_168h': Actual value 1.000 , but expected 2.000 +- 0.200 |
| FAIL | Maximum value of 'salidas_lag_336h': Equal 205.000 +- 20.500 | Maximum value of 'salidas_lag_336h': Actual value 79.000 , but expected 205.000 +- 20.500 |
| FAIL | Mean value of 'salidas_lag_336h': Equal 1.360 +- 0.136 | Mean value of 'salidas_lag_336h': Actual value 1.143 , but expected 1.360 +- 0.136 |
| FAIL | Quantile 0.75 of 'salidas_lag_336h': Equal 2.000 +- 0.200 | Quantile 0.75 of 'salidas_lag_336h': Actual value 1.000 , but expected 2.000 +- 0.200 |
| FAIL | Maximum value of 'llegadas_lag_24h': Equal 114.000 +- 11.400 | Maximum value of 'llegadas_lag_24h': Actual value 136.000 , but expected 114.000 +- 11.400 |
| FAIL | Mean value of 'salidas_media_misma_hora_7d': Equal 1.354 +- 0.135 | Mean value of 'salidas_media_misma_hora_7d': Actual value 1.200 , but expected 1.354 +- 0.135 |
| FAIL | Quantile 0.5 of 'salidas_media_misma_hora_7d': Equal 0.714 +- 0.071 | Quantile 0.5 of 'salidas_media_misma_hora_7d': Actual value 0.571 , but expected 0.714 +- 0.071 |
| FAIL | Quantile 0.75 of 'salidas_media_misma_hora_7d': Equal 1.857 +- 0.186 | Quantile 0.75 of 'salidas_media_misma_hora_7d': Actual value 1.571 , but expected 1.857 +- 0.186 |
| FAIL | Mean value of 'salidas_media_24h_previa': Equal 1.356 +- 0.136 | Mean value of 'salidas_media_24h_previa': Actual value 1.213 , but expected 1.356 +- 0.136 |
| FAIL | Quantile 0.25 of 'salidas_media_24h_previa': Equal 0.458 +- 0.046 | Quantile 0.25 of 'salidas_media_24h_previa': Actual value 0.333 , but expected 0.458 +- 0.046 |
| FAIL | Quantile 0.5 of 'salidas_media_24h_previa': Equal 0.917 +- 0.092 | Quantile 0.5 of 'salidas_media_24h_previa': Actual value 0.792 , but expected 0.917 +- 0.092 |
| FAIL | Quantile 0.75 of 'salidas_media_24h_previa': Equal 1.833 +- 0.183 | Quantile 0.75 of 'salidas_media_24h_previa': Actual value 1.625 , but expected 1.833 +- 0.183 |
| FAIL | Maximum value of 'salidas_media_7d': Equal 11.696 +- 1.170 | Maximum value of 'salidas_media_7d': Actual value 16.054 , but expected 11.696 +- 1.170 |
| FAIL | Mean value of 'salidas_media_7d': Equal 1.356 +- 0.136 | Mean value of 'salidas_media_7d': Actual value 1.194 , but expected 1.356 +- 0.136 |
| FAIL | Quantile 0.25 of 'salidas_media_7d': Equal 0.494 +- 0.049 | Quantile 0.25 of 'salidas_media_7d': Actual value 0.375 , but expected 0.494 +- 0.049 |
| FAIL | Quantile 0.5 of 'salidas_media_7d': Equal 0.982 +- 0.098 | Quantile 0.5 of 'salidas_media_7d': Actual value 0.798 , but expected 0.982 +- 0.098 |
| FAIL | Quantile 0.75 of 'salidas_media_7d': Equal 1.815 +- 0.182 | Quantile 0.75 of 'salidas_media_7d': Actual value 1.589 , but expected 1.815 +- 0.182 |
| FAIL | Maximum value of 'salidas': Equal 117.000 +- 11.700 | Maximum value of 'salidas': Actual value 75.000 , but expected 117.000 +- 11.700 |
