# Resumen de la validacion de datos

- Version de datos: `06b0010a4627`
- Fecha: 2026-10-30
- Herramienta: DeepChecks 0.19.1 (ambiente aparte, ver README)
- Muestra: hasta 200,000 filas por particion en P1 y P2 (semilla 42); P3 completo
- Reportes HTML: `reports/validacion/<pregunta>/` (ignorados por Git: llevan filas)

## P1

- entrenamiento: 200,000 filas validadas de 4,692,984
- prueba: 200,000 filas validadas de 991,632

Pasan: 22. Fallas esperadas: 2. Bloqueantes: 0. No aplican: 2.

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

Fallas esperadas de P1 y por que:

- **Feature-Feature Correlation** (entrenamiento): Las medias moviles de las 24 h y de los 7 dias previos se traslapan (las 24 h estan dentro de las 168 h; ver construir_dataset_p1), asi que su correlacion de Spearman supera 0.9 por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **Feature-Feature Correlation** (prueba): Las medias moviles de las 24 h y de los 7 dias previos se traslapan (las 24 h estan dentro de las 168 h; ver construir_dataset_p1), asi que su correlacion de Spearman supera 0.9 por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.

## P2

- entrenamiento: 200,000 filas validadas de 1,585,048
- prueba: 200,000 filas validadas de 333,960

Pasan: 24. Fallas esperadas: 2. Bloqueantes: 0. No aplican: 0.

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

Fallas esperadas de P2 y por que:

- **Feature-Feature Correlation** (entrenamiento): Las variables relativas neto_rel_* son la version absoluta dividida entre la capacidad de la estacion (ver construir_dataset_p2), asi que quedan muy correlacionadas con ella (Spearman > 0.9) por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.
- **Feature-Feature Correlation** (prueba): Las variables relativas neto_rel_* son la version absoluta dividida entre la capacidad de la estacion (ver construir_dataset_p2), asi que quedan muy correlacionadas con ella (Spearman > 0.9) por construccion. Es redundancia entre variables, no un defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El check usa 10,000 filas de la muestra.

## P3

- entrenamiento: 370 filas validadas de 370
- prueba: 459 filas validadas de 459

Pasan: 18. Fallas esperadas: 2. Bloqueantes: 0. No aplican: 6.

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

Fallas esperadas de P3 y por que:

- **Single Value in Column** (entrenamiento): Columnas de perfil horario de las 02 a las 04 h valen 0.0 en todas las estaciones de la particion (entrenamiento: 7 columnas; prueba: 3). Son horas de madrugada sin viajes; la causa exacta (cierre del servicio o poca actividad) no esta verificada. El conjunto cambia entre particiones: 6 de las 7 columnas constantes en entrenamiento no lo son en prueba, y 2 de las 3 de prueba no lo son en entrenamiento. Impacto medido: en prueba esas 6 columnas valen como maximo 0.006 y el StandardScaler ajustado en entrenamiento las deja sin escalar (varianza 0); frente a variables estandarizadas, de desviacion 1, es despreciable.
- **Single Value in Column** (prueba): Columnas de perfil horario de las 02 a las 04 h valen 0.0 en todas las estaciones de la particion (entrenamiento: 7 columnas; prueba: 3). Son horas de madrugada sin viajes; la causa exacta (cierre del servicio o poca actividad) no esta verificada. El conjunto cambia entre particiones: 6 de las 7 columnas constantes en entrenamiento no lo son en prueba, y 2 de las 3 de prueba no lo son en entrenamiento. Impacto medido: en prueba esas 6 columnas valen como maximo 0.006 y el StandardScaler ajustado en entrenamiento las deja sin escalar (varianza 0); frente a variables estandarizadas, de desviacion 1, es despreciable.
