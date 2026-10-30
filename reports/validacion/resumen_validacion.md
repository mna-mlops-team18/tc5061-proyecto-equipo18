# Resumen de la validacion de datos

- Version de datos: `06b0010a4627`
- Fecha: 2026-10-30
- Herramienta: DeepChecks 0.19.1 (ambiente aparte, ver README)
- Muestra: hasta 200,000 filas por particion en P1 y P2 (semilla 42); P3 completo
- Reportes HTML: `reports/validacion/<pregunta>/` (ignorados por Git: llevan filas)

## P1

- entrenamiento: 200,000 filas validadas de 4,692,984
- prueba: 200,000 filas validadas de 991,632

Pasan: 29. Fallas esperadas: 5. Bloqueantes: 0. No aplican: 4.

| Suite | Particion | Comprobacion | Estado | Detalle |
|---|---|---|---|---|
| integridad | entrenamiento | Feature-Feature Correlation | falla_esperada | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('salidas_media_24h_previa', 'salidas_media_7d')] |
| integridad | entrenamiento | Single Value in Column | pasa | Passed for 18 relevant columns |
| integridad | entrenamiento | Special Characters | pasa | Passed for 18 relevant columns |
| integridad | entrenamiento | Mixed Nulls | pasa | Passed for 18 relevant columns |
| integridad | entrenamiento | Mixed Data Types | pasa | 18 columns passed: found 0 columns with negligible types mix, and 18 columns without any types mix |
| integridad | entrenamiento | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | entrenamiento | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | entrenamiento | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | entrenamiento | Feature Label Correlation | pasa | Passed for 16 relevant columns |
| integridad | entrenamiento | Identifier Label Correlation | pasa | Passed for 1 relevant column |
| integridad | entrenamiento | PercentOfNulls | pasa | Passed for 16 relevant columns |
| integridad | entrenamiento | Outlier Sample Detection | informativo | sin condicion |
| integridad | entrenamiento | Conflicting Labels | no_aplica | no aplica a regresion (la etiqueta es continua) |
| integridad | prueba | Feature-Feature Correlation | falla_esperada | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('salidas_media_24h_previa', 'salidas_media_7d')] |
| integridad | prueba | Single Value in Column | pasa | Passed for 18 relevant columns |
| integridad | prueba | Special Characters | pasa | Passed for 18 relevant columns |
| integridad | prueba | Mixed Nulls | pasa | Passed for 18 relevant columns |
| integridad | prueba | Mixed Data Types | pasa | 18 columns passed: found 0 columns with negligible types mix, and 18 columns without any types mix |
| integridad | prueba | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | prueba | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | prueba | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | prueba | Feature Label Correlation | pasa | Passed for 16 relevant columns |
| integridad | prueba | Identifier Label Correlation | pasa | Passed for 1 relevant column |
| integridad | prueba | PercentOfNulls | pasa | Passed for 16 relevant columns |
| integridad | prueba | Outlier Sample Detection | informativo | sin condicion |
| integridad | prueba | Conflicting Labels | no_aplica | no aplica a regresion (la etiqueta es continua) |
| integridad | completo | Llave ('estacion_id', 'hora') sin nulos (archivo completo) | pasa | 0 celdas vacias en la llave |
| integridad | completo | Llave ('estacion_id', 'hora') unica (archivo completo) | pasa | 0 filas con la llave repetida |
| deriva | entrenamiento_vs_prueba | New Category Train Test | falla_esperada | Ratio of samples with a new category is less or equal to 0%: Found 1 features with ratio of new categories above threshold: {'estacion_id': '17.73%'} |
| deriva | entrenamiento_vs_prueba | Feature Drift | falla_esperada | categorical drift score < 0.2 and numerical drift score < 0.2: Failed for 1 out of 17 columns. Found 1 categorical columns with Cramer's V above threshold: {'mes': '0.84'} |
| deriva | entrenamiento_vs_prueba | Multivariate Drift | falla_esperada | Drift value is less than 0.25: Found drift value of: 0.85, corresponding to a domain classifier AUC of: 0.92 |
| deriva | entrenamiento_vs_prueba | Datasets Size Comparison | pasa | Test-Train size ratio is 1 |
| deriva | entrenamiento_vs_prueba | String Mismatch Comparison | pasa | No relevant columns to check were found |
| deriva | entrenamiento_vs_prueba | Date Train-Test Leakage (duplicates) | pasa | No leaked dates found |
| deriva | entrenamiento_vs_prueba | Date Train-Test Leakage (overlap) | pasa | No leaked dates found |
| deriva | entrenamiento_vs_prueba | Train Test Samples Mix | pasa | Percent of test data samples that appear in train data: 0.03% |
| deriva | entrenamiento_vs_prueba | Feature Label Correlation Change | pasa | Passed for 17 relevant columns; Passed for 17 relevant columns |
| deriva | entrenamiento_vs_prueba | Label Drift | pasa | Label's drift score Kolmogorov-Smirnov is 0.03 |
| deriva | entrenamiento_vs_prueba | New Label Train Test | no_aplica | no aplica a regresion (la etiqueta es continua) |
| deriva | entrenamiento_vs_prueba | Index Train Test Leakage | no_aplica | sin columna de indice; la llave se revisa en el archivo completo y la fuga temporal con Date Train-Test Leakage |

Fallas esperadas de P1 y por que:

- **Feature-Feature Correlation** (entrenamiento): Las medias moviles de las 24 h y de los 7 dias previos se traslapan (las 24 h estan dentro de las 168 h; ver construir_dataset_p1), asi que su correlacion de Spearman supera 0.9 por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **Feature-Feature Correlation** (prueba): Las medias moviles de las 24 h y de los 7 dias previos se traslapan (las 24 h estan dentro de las 168 h; ver construir_dataset_p1), asi que su correlacion de Spearman supera 0.9 por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **New Category Train Test** (entrenamiento_vs_prueba): Hay estaciones que aparecen en la prueba y nunca en el entrenamiento (99 de 469 en el archivo completo, conteo con pandas): son el 17.73% de las filas de la prueba. La red crece con el tiempo, asi que se espera. Solo se aceptan en estacion_id; si otra variable categorica trae categorias nuevas, el hallazgo vuelve a ser bloqueante. Cuando se agregue una estacion, el modelo no la vio en el entrenamiento: va a la seccion de limitaciones.
- **Feature Drift** (entrenamiento_vs_prueba): El entrenamiento cubre los 12 meses (oct 2024 a mar 2026) y la prueba solo jul a sep, asi que la distribucion de mes difiere por construccion (Cramer's V 0.84). Es estacionalidad de la particion cronologica, no un defecto de los datos. Es la unica variable que deriva: la siguiente tiene un puntaje menor a 0.1, y la etiqueta no deriva (Label Drift 0.03).
- **Multivariate Drift** (entrenamiento_vs_prueba): El clasificador de dominio separa entrenamiento y prueba (AUC 0.92, puntaje 0.85) y la unica variable que contribuye es mes (importancia 1.0), por la misma razon que Feature Drift. Sin mes, el AUC baja a 0.58 y el puntaje a 0.15, bajo el umbral de 0.25; lo que queda es estacion_id: las estaciones nuevas, ya cubiertas arriba.

## P2

- entrenamiento: 200,000 filas validadas de 1,585,048
- prueba: 200,000 filas validadas de 333,960

Pasan: 32. Fallas esperadas: 5. Bloqueantes: 0. No aplican: 1.

| Suite | Particion | Comprobacion | Estado | Detalle |
|---|---|---|---|---|
| integridad | entrenamiento | Feature-Feature Correlation | falla_esperada | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('neto_lag_dia', 'neto_rel_lag_dia'), ('neto_media_mismo_bloque_7d', 'neto_rel_media_mismo_bloque_7d')] |
| integridad | entrenamiento | Single Value in Column | pasa | Passed for 21 relevant columns |
| integridad | entrenamiento | Special Characters | pasa | Passed for 21 relevant columns |
| integridad | entrenamiento | Mixed Nulls | pasa | Passed for 21 relevant columns |
| integridad | entrenamiento | Mixed Data Types | pasa | 21 columns passed: found 0 columns with negligible types mix, and 21 columns without any types mix |
| integridad | entrenamiento | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | entrenamiento | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | entrenamiento | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | entrenamiento | Conflicting Labels | pasa | Ratio of samples with conflicting labels: 0% |
| integridad | entrenamiento | Feature Label Correlation | pasa | Passed for 19 relevant columns |
| integridad | entrenamiento | Identifier Label Correlation | pasa | Passed for 1 relevant column |
| integridad | entrenamiento | PercentOfNulls | pasa | Passed for 19 relevant columns |
| integridad | entrenamiento | Outlier Sample Detection | informativo | sin condicion |
| integridad | prueba | Feature-Feature Correlation | falla_esperada | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('neto_lag_dia', 'neto_rel_lag_dia'), ('neto_media_mismo_bloque_7d', 'neto_rel_media_mismo_bloque_7d')] |
| integridad | prueba | Single Value in Column | pasa | Passed for 21 relevant columns |
| integridad | prueba | Special Characters | pasa | Passed for 21 relevant columns |
| integridad | prueba | Mixed Nulls | pasa | Passed for 21 relevant columns |
| integridad | prueba | Mixed Data Types | pasa | 21 columns passed: found 0 columns with negligible types mix, and 21 columns without any types mix |
| integridad | prueba | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | prueba | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | prueba | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | prueba | Conflicting Labels | pasa | Ratio of samples with conflicting labels: 0% |
| integridad | prueba | Feature Label Correlation | pasa | Passed for 19 relevant columns |
| integridad | prueba | Identifier Label Correlation | pasa | Passed for 1 relevant column |
| integridad | prueba | PercentOfNulls | pasa | Passed for 19 relevant columns |
| integridad | prueba | Outlier Sample Detection | informativo | sin condicion |
| integridad | completo | Llave ('estacion_id', 'bloque_inicio') sin nulos (archivo completo) | pasa | 0 celdas vacias en la llave |
| integridad | completo | Llave ('estacion_id', 'bloque_inicio') unica (archivo completo) | pasa | 0 filas con la llave repetida |
| deriva | entrenamiento_vs_prueba | New Category Train Test | falla_esperada | Ratio of samples with a new category is less or equal to 0%: Found 1 features with ratio of new categories above threshold: {'estacion_id': '18.52%'} |
| deriva | entrenamiento_vs_prueba | Feature Drift | falla_esperada | categorical drift score < 0.2 and numerical drift score < 0.2: Failed for 1 out of 20 columns. Found 1 categorical columns with Cramer's V above threshold: {'mes': '0.84'} |
| deriva | entrenamiento_vs_prueba | Multivariate Drift | falla_esperada | Drift value is less than 0.25: Found drift value of: 0.86, corresponding to a domain classifier AUC of: 0.93 |
| deriva | entrenamiento_vs_prueba | Datasets Size Comparison | pasa | Test-Train size ratio is 1 |
| deriva | entrenamiento_vs_prueba | New Label Train Test | pasa | Found 0 new labels in test data: [] |
| deriva | entrenamiento_vs_prueba | String Mismatch Comparison | pasa | No relevant columns to check were found |
| deriva | entrenamiento_vs_prueba | Date Train-Test Leakage (duplicates) | pasa | No leaked dates found |
| deriva | entrenamiento_vs_prueba | Date Train-Test Leakage (overlap) | pasa | No leaked dates found |
| deriva | entrenamiento_vs_prueba | Train Test Samples Mix | pasa | Percent of test data samples that appear in train data: 0.17% |
| deriva | entrenamiento_vs_prueba | Feature Label Correlation Change | pasa | Passed for 20 relevant columns; Passed for 20 relevant columns |
| deriva | entrenamiento_vs_prueba | Label Drift | pasa | Label's drift score Cramer's V is 0.04 |
| deriva | entrenamiento_vs_prueba | Index Train Test Leakage | no_aplica | sin columna de indice; la llave se revisa en el archivo completo y la fuga temporal con Date Train-Test Leakage |

Fallas esperadas de P2 y por que:

- **Feature-Feature Correlation** (entrenamiento): Las variables relativas neto_rel_* son la version absoluta dividida entre la capacidad de la estacion (ver construir_dataset_p2), asi que quedan muy correlacionadas con ella (Spearman > 0.9) por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **Feature-Feature Correlation** (prueba): Las variables relativas neto_rel_* son la version absoluta dividida entre la capacidad de la estacion (ver construir_dataset_p2), asi que quedan muy correlacionadas con ella (Spearman > 0.9) por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **New Category Train Test** (entrenamiento_vs_prueba): Hay estaciones que aparecen en la prueba y nunca en el entrenamiento (99 de 469 en el archivo completo, conteo con pandas): son el 18.52% de las filas de la prueba. La red crece con el tiempo, asi que se espera. Solo se aceptan en estacion_id; si otra variable categorica trae categorias nuevas, el hallazgo vuelve a ser bloqueante. Cuando se agregue una estacion, el modelo no la vio en el entrenamiento: va a la seccion de limitaciones.
- **Feature Drift** (entrenamiento_vs_prueba): El entrenamiento cubre los 12 meses (oct 2024 a mar 2026) y la prueba solo jul a sep, asi que la distribucion de mes difiere por construccion (Cramer's V 0.84). Es estacionalidad de la particion cronologica, no un defecto de los datos. Es la unica variable que deriva: la siguiente tiene un puntaje menor a 0.1, y la etiqueta no deriva (Label Drift 0.04).
- **Multivariate Drift** (entrenamiento_vs_prueba): El clasificador de dominio separa entrenamiento y prueba (AUC 0.93, puntaje 0.86) y la unica variable que contribuye es mes (importancia 1.0), por la misma razon que Feature Drift. Sin mes, el AUC baja a 0.58 y el puntaje a 0.17, bajo el umbral de 0.25; lo que queda es estacion_id: las estaciones nuevas, ya cubiertas arriba.

## P3

- entrenamiento: 370 filas validadas de 370
- prueba: 459 filas validadas de 459

Pasan: 22. Fallas esperadas: 4. Bloqueantes: 0. No aplican: 12.

| Suite | Particion | Comprobacion | Estado | Detalle |
|---|---|---|---|---|
| integridad | entrenamiento | Single Value in Column | falla_esperada | Does not contain only a single value: Found 7 out of 99 columns with a single value: ['sal_lab_h02', 'sal_lab_h03', 'sal_lab_h04', 'sal_fds_h02', 'sal_fds_h03', 'sal_fds_h04', 'lle_fds_h04'] |
| integridad | entrenamiento | Special Characters | pasa | Passed for 99 relevant columns |
| integridad | entrenamiento | Mixed Nulls | pasa | Passed for 99 relevant columns |
| integridad | entrenamiento | Mixed Data Types | pasa | 99 columns passed: found 0 columns with negligible types mix, and 99 columns without any types mix |
| integridad | entrenamiento | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | entrenamiento | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | entrenamiento | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | entrenamiento | Feature-Feature Correlation | pasa | All correlations are less than 0.9 except pairs [] |
| integridad | entrenamiento | PercentOfNulls | pasa | Passed for 99 relevant columns |
| integridad | entrenamiento | Outlier Sample Detection | informativo | sin condicion |
| integridad | entrenamiento | Conflicting Labels | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | entrenamiento | Feature Label Correlation | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | entrenamiento | Identifier Label Correlation | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | prueba | Single Value in Column | falla_esperada | Does not contain only a single value: Found 3 out of 99 columns with a single value: ['sal_lab_h03', 'lle_lab_h03', 'lle_lab_h04'] |
| integridad | prueba | Special Characters | pasa | Passed for 99 relevant columns |
| integridad | prueba | Mixed Nulls | pasa | Passed for 99 relevant columns |
| integridad | prueba | Mixed Data Types | pasa | 99 columns passed: found 0 columns with negligible types mix, and 99 columns without any types mix |
| integridad | prueba | String Mismatch | pasa | Passed for 1 relevant column |
| integridad | prueba | Data Duplicates | pasa | Found 0% duplicate data |
| integridad | prueba | String Length Out Of Bounds | pasa | No relevant columns to check were found |
| integridad | prueba | Feature-Feature Correlation | pasa | All correlations are less than 0.9 except pairs [] |
| integridad | prueba | PercentOfNulls | pasa | Passed for 99 relevant columns |
| integridad | prueba | Outlier Sample Detection | informativo | sin condicion |
| integridad | prueba | Conflicting Labels | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | prueba | Feature Label Correlation | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | prueba | Identifier Label Correlation | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| integridad | completo | Llave ('estacion_id', 'periodo') sin nulos (archivo completo) | pasa | 0 celdas vacias en la llave |
| integridad | completo | Llave ('estacion_id', 'periodo') unica (archivo completo) | pasa | 0 filas con la llave repetida |
| deriva | entrenamiento_vs_prueba | Feature Drift | falla_esperada | categorical drift score < 0.2 and numerical drift score < 0.2: Failed for 8 out of 99 columns. Found 8 numeric columns with Kolmogorov-Smirnov above threshold: {'sal_fds_h05': '0.28', 'sal_fds_h17': '0.21', 'sal_fds_h... |
| deriva | entrenamiento_vs_prueba | Multivariate Drift | falla_esperada | Drift value is less than 0.25: Found drift value of: 0.67, corresponding to a domain classifier AUC of: 0.83 |
| deriva | entrenamiento_vs_prueba | Datasets Size Comparison | pasa | Test-Train size ratio is 1.24 |
| deriva | entrenamiento_vs_prueba | New Category Train Test | pasa | No relevant features to check were found |
| deriva | entrenamiento_vs_prueba | String Mismatch Comparison | pasa | No relevant columns to check were found |
| deriva | entrenamiento_vs_prueba | Train Test Samples Mix | pasa | No samples mix found |
| deriva | entrenamiento_vs_prueba | New Label Train Test | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| deriva | entrenamiento_vs_prueba | Date Train Test Leakage Duplicates | no_aplica | sin columna de tiempo; la particion viene de la columna periodo |
| deriva | entrenamiento_vs_prueba | Date Train Test Leakage Overlap | no_aplica | sin columna de tiempo; la particion viene de la columna periodo |
| deriva | entrenamiento_vs_prueba | Index Train Test Leakage | no_aplica | sin columna de indice; la llave se revisa en el archivo completo y la fuga temporal con Date Train-Test Leakage |
| deriva | entrenamiento_vs_prueba | Feature Label Correlation Change | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |
| deriva | entrenamiento_vs_prueba | Label Drift | no_aplica | requiere etiqueta y esta pregunta no tiene (clustering) |

Fallas esperadas de P3 y por que:

- **Single Value in Column** (entrenamiento): Columnas de perfil horario de las 02 a las 04 h valen 0.0 en todas las estaciones de la particion (entrenamiento: 7 columnas; prueba: 3). Son horas de madrugada sin viajes; la causa exacta (cierre del servicio o poca actividad) no esta verificada. El conjunto cambia entre particiones: 6 de las 7 columnas constantes en entrenamiento no lo son en prueba, y 2 de las 3 de prueba no lo son en entrenamiento. Impacto medido: en prueba esas 6 columnas valen como maximo 0.006 y el StandardScaler ajustado en entrenamiento las deja sin escalar (varianza 0); frente a variables estandarizadas, de desviacion 1, es despreciable.
- **Single Value in Column** (prueba): Columnas de perfil horario de las 02 a las 04 h valen 0.0 en todas las estaciones de la particion (entrenamiento: 7 columnas; prueba: 3). Son horas de madrugada sin viajes; la causa exacta (cierre del servicio o poca actividad) no esta verificada. El conjunto cambia entre particiones: 6 de las 7 columnas constantes en entrenamiento no lo son en prueba, y 2 de las 3 de prueba no lo son en entrenamiento. Impacto medido: en prueba esas 6 columnas valen como maximo 0.006 y el StandardScaler ajustado en entrenamiento las deja sin escalar (varianza 0); frente a variables estandarizadas, de desviacion 1, es despreciable.
- **Feature Drift** (entrenamiento_vs_prueba): Deriva en 8 de 99 columnas: siete de perfil horario (lle_fds_h01, lle_lab_h01, lle_fds_h05, lle_lab_h05, sal_fds_h05, sal_fds_h17, sal_fds_h18) y ratio_fin_semana, con puntaje de 0.21 a 0.42 (KS). Cinco son horas de madrugada, vecinas de las h02 a h04 constantes del hallazgo de integridad. La causa no esta verificada: la prueba tiene 459 estaciones contra 370 del entrenamiento (90 nuevas y 1 que ya no esta), y el cambio de estaciones podria explicar parte. Solo se aceptan esas 8 columnas; otra que derive vuelve a ser bloqueante.
- **Multivariate Drift** (entrenamiento_vs_prueba): El clasificador de dominio separa los periodos (AUC 0.83, puntaje 0.67). Contribuyen lle_fds_h01 (0.65), sal_fds_h05 (0.18) y balance_neto (0.12). Las dos primeras estan entre las 8 con deriva; balance_neto no (su deriva univariada es menor a 0.2) y solo se distingue en conjunto. La causa no esta verificada (ver Feature Drift de P3). Solo se acepta con esas tres como contribuyentes.
