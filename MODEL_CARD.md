# Model Card - MiBici, demanda por estacion y hora (P1)

Modelo registrado: `mibici-equipo18-p1-demanda` en el Model Registry de MLflow.
Equipo 18, TC5061 Operaciones de aprendizaje automatico, Tec de Monterrey. Noviembre de 2026.

Este documento sigue el formato de Model Cards (Mitchell et al., 2019). Describe el modelo de
P1, que es el que se monitorea, se reentrena y se compara en el A/B. Al final hay un resumen de
los modelos de P2 y P3, que se sirven en la misma API.

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Tarea | Regresion: viajes que saldran de cada estacion en cada hora del dia siguiente |
| Algoritmo | `HistGradientBoostingRegressor` de scikit-learn con perdida Poisson (`hgb_poisson`) |
| Hiperparametros | `config.P1_HIPERPARAMETROS`: learning_rate 0.08, max_iter 400, max_leaf_nodes 63, min_samples_leaf 100, l2_regularization 1.0, early stopping |
| Baseline | `rezago_semanal`: las salidas de la misma hora hace 7 dias |
| Version en produccion | Alias `@champion`, version 1 (run `166442af41ea432bbe181ccd17a571f9`) |
| Version retadora | Alias `@challenger`, version 3, reentrenada hasta agosto de 2026 (Continuous Training) |
| Linaje | Cada version guarda en sus etiquetas el run de origen, `git_commit`, `data_version`, la metrica contra el baseline y el comando `mlflow run` para reproducirla |
| Formato | Modelo de scikit-learn guardado por MLflow en formato skops (no ejecuta codigo al cargarse) |
| Como se sirve | `POST /predicciones/p1` de la API FastAPI; el modelo va dentro de la imagen (`make export-model`) |
| Licencia de los datos | Datos abiertos publicados por MiBici en <https://mibici.net/es/datos-abiertos/> |

## 2. Uso previsto

- **Para que sirve:** que el operador de MiBici planee cuantas bicicletas poner en cada estacion
  al inicio del dia. La prediccion se hace al cierre del dia anterior, por lotes (una vez al dia).
- **Quien lo usa:** el equipo de operacion y rebalanceo de MiBici.
- **Fuera de alcance:**
  - No decide sobre personas ni las perfila: predice por estacion, no por usuario.
  - No sustituye al operador: es un insumo para planear, no una orden automatica.
  - No sirve para predecir dentro del mismo dia con datos en tiempo real (para eso esta P2).
  - No se debe usar en otras ciudades o sistemas de bicicletas sin reentrenar y evaluar.
  - No conoce clima, eventos masivos, cierres de calles ni estaciones fuera de servicio.

## 3. Datos

| Campo | Valor |
|---|---|
| Fuente | Viajes abiertos de MiBici, un CSV por mes |
| Periodo | Octubre de 2024 a septiembre de 2026 (24 meses, unos 9.04 millones de viajes) |
| Version | `data_version` 06b0010a4627: manifiesto con filas y SHA-256 de cada archivo en `references/manifiesto_datos.json` |
| Particion | Cronologica: entrenamiento oct 2024 a mar 2026, validacion abr a jun 2026, prueba jul a sep 2026 |
| Unidad | Una fila por estacion y hora; el objetivo es `salidas` |
| Limpieza | Se descartan viajes de menos de 1 minuto o de mas de 4 horas |

**Privacidad.** Los archivos de MiBici traen identificador de usuario, genero y anio de
nacimiento. `dataset.py` los descarta desde la lectura (`COLUMNAS_PERSONALES`): nunca llegan a
disco procesado, a las variables ni al modelo.

**Variables de entrada** (`features.COLUMNAS_P1`): calendario (hora, dia de la semana, fin de
semana, festivo, mes), atributos de la estacion (capacidad, latitud, longitud), rezagos de
salidas de 24, 48, 168 y 336 horas, llegadas de hace 24 horas y medias recientes de salidas.
Todos los rezagos tienen al menos 24 horas, el horizonte de prediccion, para que no haya fuga de
informacion.

## 4. Metricas y desempeno

La metrica de seleccion es el MAE de validacion, en viajes por estacion y hora (menor es mejor).
La prueba (jul a sep 2026) solo se usa para reportar.

| Conjunto | hgb_poisson (champion) | Baseline `rezago_semanal` | Mejora |
|---|---|---|---|
| Validacion (abr a jun 2026) | 0.752 | 0.950 | -21 % |
| Prueba (jul a sep 2026) | 0.732 | 0.899 | -19 % |

**Desempeno mensual del champion** (MAE, 2026): abr 0.756, may 0.706, jun 0.791, jul 0.720,
ago 0.678, sep 0.797. No hay degradacion sostenida, pero septiembre es el peor mes.

**Por grupo de estaciones.** El error se mide aparte en estaciones conocidas (con historial en el
entrenamiento) y nuevas. En septiembre de 2026, el MAE en estaciones conocidas fue 0.8825 para el
champion y 0.8808 para el challenger.

**A/B testing (champion contra challenger).** Las dos versiones se evaluaron sobre las mismas
336,912 filas de septiembre de 2026 (468 estaciones), que ninguna vio al entrenar.

| Version | Entrenada hasta | MAE en septiembre de 2026 |
|---|---|---|
| A, `@champion` (v1) | marzo de 2026 | 0.7969 |
| B, `@challenger` (v3) | agosto de 2026 | 0.7951 |

MAE(A) - MAE(B) = +0.0018, con IC 95 % de [+0.0010, +0.0026] por bootstrap pareado por dia
(10,000 remuestreos). La regla de decision se fijo antes de ver los resultados
(`config.AB_*`): B gana si el intervalo queda completo sobre 0 y si en estaciones conocidas no
es mas de 1 % peor. Las dos condiciones se cumplen, asi que la regla indica promover B. La
mejora es pequena (0.2 %). Detalle completo en `reports/monitoreo/ab_test.md`.

## 5. Limitaciones

- Mas error en horas pico, donde los conteos son altos y variables.
- Mas error en estaciones nuevas sin historial: sus rezagos y medias estan vacios o son cortos.
- No conoce clima, eventos masivos ni estaciones fuera de servicio; un dia atipico se predice como
  uno normal.
- Predice salidas esperadas (un promedio), no un intervalo: no dice que tan segura es cada
  prediccion.
- La red crecio de unas 368 a 468 estaciones con viajes en 2026; mientras siga creciendo, el
  modelo vera estaciones que no conoce.

## 6. Riesgos y mitigacion

| Riesgo | Efecto | Mitigacion |
|---|---|---|
| Subestimar zonas con poco historial (estaciones nuevas) | Menos bicicletas donde si se necesitan, peor servicio en zonas que apenas se integran a la red | El error se mide aparte en estaciones conocidas y nuevas; el A/B exige no empeorar en conocidas |
| Drift por crecimiento de la red o por temporada | El error sube sin que nadie lo note | Reporte de drift con Evidently (`make drift`) y MAE mensual; reentrenamiento con datos recientes (`make retrain`) |
| Promover un modelo peor | Peor planeacion | Regla del A/B fijada antes de ver resultados; un champion peor que el baseline se rechaza en el registro |
| Usar el modelo para decidir sobre personas | Uso no previsto, riesgo de privacidad | El modelo no recibe datos personales; este documento lo declara fuera de alcance |

## 7. Monitoreo

- **Drift:** `make drift` compara las variables de P1 del entrenamiento (oct 2024 a mar 2026)
  contra jul a sep 2026 con Evidently: distancia de Wasserstein normalizada para numericas y de
  Jensen-Shannon para categoricas, umbral 0.1, sobre 200 mil filas. Hace tres comparaciones para
  separar causas: principal, sin estaciones nuevas y misma temporada. Salidas en
  `reports/monitoreo/resumen_drift.md`.
- **Hallazgo:** 99 de las 468 estaciones del periodo actual son nuevas. El MAE mensual no muestra
  degradacion sostenida, pero septiembre (0.797) es el peor mes, lo que motivo el reentrenamiento.
- **Continuous Training:** `make retrain` reentrena el mismo modelo con datos hasta agosto de 2026
  y lo registra como `@challenger`; `make abtest` lo compara y, con `PROMOVER=si`, mueve el alias
  solo si la regla lo indica.

## 8. Seguridad y operacion

- CI en cada pull request: `pip-audit` de dependencias e imagen, `bandit` sobre el codigo y
  revision de credenciales.
- Contenedor sin root (usuario 10001), imagen base fijada por digest, puertos solo en 127.0.0.1.
- `GET /salud` responde 503 si falta un modelo; `GET /modelo` expone version, run y linaje del
  modelo que esta sirviendo.
- Capacidad medida con Locust: unas 95 peticiones por segundo con 1 vCPU y 193 con 2 vCPU y un
  hilo por proceso, sin peticiones fallidas (`reports/carga/resumen_carga.md`).

## 9. Como reproducirlo

```bash
mlflow run . -e ingest --env-manager local
mlflow run . -e features --env-manager local
mlflow run . -e train -P pregunta=p1 -P modelo=rezago_semanal --env-manager local
mlflow run . -e train -P pregunta=p1 -P modelo=hgb_poisson --env-manager local
mlflow run . -e register -P pregunta=p1 --env-manager local
```

Requiere el servidor de MLflow arriba (`make mlflow-server`) y un arbol de trabajo limpio.
El comando exacto de cada version esta en su etiqueta `comando_reproduccion` del Model Registry.

## 10. Otros modelos que sirve la API

| Pregunta | Modelo registrado | Champion | Metrica de seleccion | Champion | Baseline |
|---|---|---|---|---|---|
| P2 - Riesgo de vaciado o llenado en bloques de 3 h (clasificacion) | `mibici-equipo18-p2-riesgo` | `hgb_ponderado` (gradient boosting con peso de las clases de riesgo buscado) | F1 macro de validacion | 0.607 | 0.537 (`frecuencia_mismo_bloque`) |
| P3 - Grupos de estaciones por perfil de uso (clustering) | `mibici-equipo18-p3-grupos` | `jerarquico_pca` (Ward, 3 grupos, sobre PCA al 90 % de varianza) | Silhouette de entrenamiento y estabilidad (ARI) | 0.135, ARI 0.662 | -0.052 (`cuartiles_volumen`) |

Limitaciones principales: en P2 las clases de riesgo son raras (cerca de 1.5 % de los bloques
cada una), asi que habra alertas falsas; en P3 la estructura de grupos es debil (silhouette
baja), por lo que los grupos sirven para orientar la planeacion, no como categorias exactas.
Ninguno de los dos usa datos personales.

## Fuente de las cifras

Las cifras vienen de las corridas registradas en MLflow y de los reportes en `reports/`. Si se
vuelve a correr algun paso, actualizar este documento con los nuevos valores.
