# TC5061-Proyecto-Equipo18

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta de MLOps sobre la operacion de [MiBici](https://mibici.net/es/datos-abiertos/), el
sistema de bicicleta publica del Area Metropolitana de Guadalajara. Proyecto del equipo 18 de
la materia TC5061 (Tec de Monterrey, MNA). Esta es la primera entrega del proyecto integrador:
la base reproducible (datos, variables, experimento base y flujo de trabajo en equipo) sobre la
que se construiran las entregas siguientes.

> Convencion del repositorio: codigo, documentacion, commits y descripciones de PR van en
> espanol y solo con caracteres ASCII (sin acentos ni enie). El PDF de la entrega si puede
> llevarlos.

## Proposito y preguntas

Con los viajes publicados por MiBici se responden tres preguntas, cada una con un tipo de
modelo distinto. Las decisiones de formulacion viven en `tc5061_proyecto_equipo18/config.py`.

| # | Pregunta | Tarea | Unidad y horizonte | Quien lo usa |
|---|---|---|---|---|
| P1 | Cuantos viajes saldran de una estacion en un periodo futuro? | Regresion | Estacion x hora, un dia antes | Operador del sistema (planeacion de flota) |
| P2 | Estara una estacion en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificacion (`sin_riesgo`, `riesgo_vaciado`, `riesgo_llenado`) | Estacion x bloque de 3 h | Equipo de rebalanceo |
| P3 | Que grupos de estaciones comparten patrones de uso y donde estan? | Clustering | Perfil de uso por estacion | Planeacion urbana y expansion |

MiBici no publica el inventario de bicicletas por estacion. Para P2 el riesgo se aproxima con
los viajes y la capacidad: un bloque esta en riesgo cuando el flujo neto dentro del bloque
alcanza el 50 % de la capacidad de la estacion, suponiendo que arranca a media capacidad.

## Datos

- **Fuente:** [portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/):
  un CSV de viajes por mes y la nomenclatura de estaciones. Las URL exactas de cada archivo
  estan en `references/manifiesto_datos.json`.
- **Rango:** 2024-10 a 2026-09 (24 meses). Particion cronologica: entrenamiento hasta
  2026-03-31, validacion abril a junio de 2026 y prueba julio a septiembre de 2026.
- **Datos personales:** las columnas de usuario, genero y anio de nacimiento nunca se leen,
  nunca se guardan y nunca se registran en MLflow.
- **Versionado:** `references/manifiesto_datos.json` guarda, por archivo, el numero de
  registros y el hash SHA-256 del archivo crudo y del contenido limpio. La version de los datos
  (`data_version`) es un hash de esos hashes y es el valor que se registra en MLflow. Los datos
  no se suben a Git; el manifiesto si.
- **Tamano real:** 24 meses, 9,043,167 viajes y 484 estaciones (470 activas), version de datos
  `06b0010a4627`; el detalle por mes esta en `reports/resumen_datos.md` (se regenera con el
  comando `resumen`).

## Estructura del repositorio

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

```
|-- .github/                      <- Plantilla de Pull Request
|-- .env.example                  <- Variables de entorno de ejemplo (copiar a .env)
|-- Makefile                      <- Atajos para Linux y macOS: make mlflow-server, make train, make test
|-- MLproject                     <- MLflow Project: puntos de entrada ingest, features, train y search
|-- python_env.yaml               <- Ambiente del MLflow Project (versiones de requirements.txt)
|-- README.md
|-- pyproject.toml                <- Metadatos del paquete y configuracion de pytest y ruff
|-- requirements.txt              <- Dependencias con version fija (==)
|
|-- data/                         <- Ignorado por Git, se genera con la ingesta
|   |-- raw/                      <- CSV originales, tal como los publica MiBici
|   |-- interim/                  <- Conteos intermedios (estacion x hora)
|   +-- processed/                <- Parquet limpio y conjuntos de P1, P2 y P3
|
|-- docs/                         <- Proyecto mkdocs por defecto de la plantilla
|-- models/                       <- Modelos entrenados (ignorado por Git)
|-- notebooks/                    <- Solo exploracion, p. ej. 1.0-ja-exploracion-viajes.ipynb
|-- references/                   <- manifiesto_datos.json, ML Canvas y arquitectura (md, png, html)
|-- reports/                      <- propuesta_entrega1.md (Parte A) y resumen_datos.md
|   +-- figures/                  <- Figuras generadas y capturas de MLflow
|-- tests/                        <- Pruebas de ingesta, de fuga de informacion y de P1, P2 y P3
|
+-- tc5061_proyecto_equipo18/     <- Codigo fuente
    |-- config.py                 <- Unico lugar de rutas, fechas, semillas y parametros
    |-- dataset.py                <- Ingesta: descarga, normaliza, Parquet y manifiesto (entry point ingest)
    |-- features.py               <- Variables de P1, P2 y P3 sin fuga (entry point features)
    |-- seguimiento.py            <- Registra en MLflow las etapas de datos
    |-- plots.py                  <- Figuras y mapas
    +-- modeling/
        |-- comun.py              <- Registro comun: etiquetas, parametros, artefactos, conjuntos y modelo
        |-- p1_regresion.py       <- P1: modelos del catalogo de config.py (regresion)
        |-- p2_clasificacion.py   <- P2: modelos del catalogo de config.py (clasificacion)
        |-- p3_clustering.py      <- P3: modelos del catalogo de config.py (clustering)
        |-- train.py              <- Entry point train: un modelo del catalogo por corrida
        |-- busqueda.py           <- Entry point search: busqueda con nested runs
        |-- registro.py           <- Entry point register: Model Registry con aliases champion/challenger
        +-- predict.py            <- Plantilla de inferencia (siguientes entregas)
```

## Instalacion

Requiere Python 3.14 y unos 2 GB libres en disco (los CSV crudos pesan cerca de 0.75 GB y los Parquet limpios
cerca de 0.1 GB; las variables de `make features` ocupan algo mas).

```bash
git clone https://github.com/mna-mlops-team18/tc5061-proyecto-equipo18.git
cd tc5061-proyecto-equipo18

python3.14 -m venv .venv
source .venv/bin/activate
make requirements          # instala requirements.txt (versiones fijas) y el paquete

cp .env.example .env       # opcional: solo para cambiar la URI de MLflow o usar un espejo
```

En Windows (PowerShell), donde normalmente no hay `make`:

```powershell
git clone https://github.com/mna-mlops-team18/tc5061-proyecto-equipo18.git
cd tc5061-proyecto-equipo18

py -3.14 -m venv .venv
.venv\Scripts\activate     # si lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env   # opcional
```

Sin `make`, cada atajo equivale al comando que aparece en su comentario (por ejemplo,
`make data` es `python -m tc5061_proyecto_equipo18.dataset ingest`).

Para abrir los notebooks usa un editor con soporte de Jupyter (por ejemplo VS Code) y
selecciona el kernel del ambiente `.venv`; `ipykernel` ya viene instalado.

## Ingesta de datos

Sin pasos manuales; todos los comandos se corren desde la raiz con el ambiente activado.

```bash
mlflow run . -e ingest --env-manager local            # o: python -m tc5061_proyecto_equipo18.dataset ingest
python -m tc5061_proyecto_equipo18.dataset verify     # o: make verify
python -m tc5061_proyecto_equipo18.dataset resumen    # regenera reports/resumen_datos.md
```

- `ingest` descarga los meses del rango y la nomenclatura a `data/raw/` (omite lo que ya
  existe), detecta la codificacion (UTF-8 o Windows-1252) y el formato de fecha de cada
  archivo, unifica los nombres de columna y guarda un Parquet por mes en
  `data/processed/viajes/`. Si el manifiesto ya existe, comprueba que los datos
  reconstruidos sean identicos y falla listando las diferencias si no lo son.
- `verify` compara los datos locales con el manifiesto versionado. Si coincide, tienes
  exactamente los mismos datos que el resto del equipo.

Despues de la ingesta, las variables se construyen con:

```bash
mlflow run . -e features --env-manager local          # o: python -m tc5061_proyecto_equipo18.features
python -m pytest                                      # o: make test
```

Los comandos `mlflow run` necesitan el servidor de MLflow (seccion siguiente).

## Experimentos en MLflow

**Servidor de tracking.** El Model Registry necesita una base de datos, asi que se usa un
servidor local de MLflow. En una terminal propia, desde la raiz del repo y con el ambiente
activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000   # o: make mlflow-server
```

En la terminal donde se corre `mlflow run` (PowerShell; en bash usar `export`):

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "mibici-equipo18"
```

**MLflow Project.** `MLproject` tiene los puntos de entrada `ingest`, `features`, `train`
(`pregunta`, `modelo` y cualquier hiperparametro con `-P`), `search` (busqueda con nested
runs) y `register` (Model Registry, abajo). Los modelos, hiperparametros por defecto y
rejillas de busqueda viven en `config.py` (`P1_MODELOS`, `P2_MODELOS`, `P3_MODELOS` y
`P*_BUSQUEDA`):

```bash
mlflow run . -e train -P pregunta=p1 -P modelo=rezago_semanal --env-manager local
mlflow run . -e train -P pregunta=p1 -P modelo=hgb_poisson -P learning_rate=0.05 --env-manager local
mlflow run . -e search -P pregunta=p3 --env-manager local
```

Cada corrida registra las etiquetas `pregunta`, `etapa` (baseline, candidato, busqueda),
`modelo`, `familia_modelo`, `equipo`, `data_version`, `git_commit` y `git_sucio`; los
parametros (semilla, features, hiperparametros); metricas con el conjunto en el nombre
(`validacion_mae`, `prueba_f1_macro`, ...) y de costo; y artefactos en `config/`, `datos/`,
`figuras/` (mas `tablas/` y `mapas/` en P3), el modelo con firma y los conjuntos de datos con
`mlflow.log_input`. La propuesta de la entrega 1 esta en `reports/propuesta_entrega1.md`.

**Model Registry y ciclo de vida.** El modelo elegido de cada pregunta se registra con el
punto de entrada `register` (`modeling/registro.py`) en un modelo con nombre fijo
(`config.MODELOS_REGISTRADOS`):

| Pregunta | Modelo registrado |
|---|---|
| P1 | `mibici-equipo18-p1-demanda` |
| P2 | `mibici-equipo18-p2-riesgo` |
| P3 | `mibici-equipo18-p3-grupos` |

```bash
mlflow run . -e register -P pregunta=p1 --env-manager local                            # o: make register PREGUNTA=p1
mlflow run . -e register -P pregunta=p2 -P run_id=<run elegido> --env-manager local
mlflow run . -e register -P pregunta=p3 -P challenger_run_id=ninguno --env-manager local
```

Usamos **aliases**, no etapas: las etapas (Staging, Production) estan obsoletas desde MLflow
2.9 y los aliases son el esquema vigente. `@champion` es la version en produccion y
`@challenger` la siguiente mejor, que se puede evaluar en paralelo antes de promoverla.
Promover es mover el alias; quien consume el modelo siempre usa la misma URI:

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo18-p1-demanda@champion")
```

Como se elige: solo cuentan runs con modelo (etapa `candidato` o `busqueda`) lanzados con
`mlflow run` desde un arbol de trabajo limpio y con la misma `data_version` que el baseline,
ordenados por la metrica de `config.METRICAS_SELECCION`. Sin `-P run_id`, el champion es el
mejor y el challenger el siguiente con otra configuracion; con `-P run_id` el equipo fija el
champion (por ejemplo, uno mas estable o mas barato) y `-P challenger_run_id` fija o quita
(`ninguno`) el challenger. Un champion que no mejora al baseline se rechaza. Cada version
guarda su linaje en etiquetas y descripcion (experimento, run de origen, `git_commit`,
`data_version`, metrica contra el baseline y comando para reproducirlo), y el comando
verifica que `models:/<nombre>@champion` se carga y predice antes de terminar.

## Validacion de datos

La validacion revisa los datasets procesados de P1, P2 y P3 (`data/processed/`) con
DeepChecks 0.19.1, no los CSV crudos. Vive en la carpeta `validacion_datos/`, fuera del
paquete y de `tests/`, y corre en un ambiente propio.

### Por que un ambiente aparte

DeepChecks 0.19.1 no se importa con el stack fijado en `requirements.txt` (Python 3.14,
NumPy 2, pandas 3, scikit-learn 1.9). Se comprobaron tres causas:

1. Importa `pkg_resources`, que las versiones recientes de `setuptools` ya no incluyen.
2. Su modulo tabular pide a scikit-learn el scorer `max_error`, que ya no existe.
3. Usa `np.Inf`, eliminado en NumPy 2.

Se consideraron tres opciones:

| Opcion | Idea | Decision y razon |
|---|---|---|
| **A. Ambiente aparte** | DeepChecks solo lee los Parquet; nunca necesita los modelos. Python 3.12 con su propio `requirements-validacion.txt`. | **Elegida.** Aisla el problema sin tocar nada de lo ya entregado. |
| B. Bajar versiones de todo el proyecto | Mover el proyecto a NumPy, pandas y scikit-learn anteriores. | Descartada. Los modelos registrados en la semana 6, sus firmas y la afirmacion de reproducir desde un clon limpio dependen de las versiones actuales; habria que rehacer esa semana para poder usar una herramienta de revision. |
| C. Parchar al importar | Definir en tiempo de ejecucion los nombres que faltan (`pkg_resources`, `max_error`, `np.Inf`). | Descartada. Una herramienta de validacion con parches propios no es confiable ni facil de explicar a una auditoria: no sabriamos si un resultado es del check o del parche. |

Costos de A, y como se contienen:

- **Dos copias de la definicion de los datos.** `validacion_datos/contrato.py` es pandas puro
  y repite las listas de variables y la funcion de particion de `features.py`, que no se puede
  importar en el ambiente 3.12 (arrastra `mlflow`). La prueba `tests/test_contrato_validacion.py`
  corre en el ambiente principal, sin datos, y falla si alguna de las dos copias cambia sola.
- **pyarrow queda por debajo de 21.** pyarrow 21 o mayor exige NumPy 2 y DeepChecks necesita
  `numpy<2`. pandas 2.2 lee sin problema los Parquet escritos con pandas 3 (`hora` y
  `bloque_inicio` llegan como `datetime64[ms]`).
- **`pytest` del ambiente principal no ve estas pruebas** (`testpaths = ["tests"]`).

### Crear el ambiente

Requiere [uv](https://docs.astral.sh/uv/) (descarga Python 3.12 si no esta instalado).

```bash
make venv-validacion       # uv venv .venv-validacion --python 3.12 + instala requirements-validacion.txt
```

En Windows (PowerShell), sin `make`:

```powershell
py -3.12 -m venv .venv-validacion
.venv-validacion\Scripts\python -m pip install -r requirements-validacion.txt
```

### Correr la validacion

Necesita los datos y las variables (`make data` y `make features`). Se corre desde la raiz del
repo; el ambiente no instala el paquete, solo importa `config.py`.

```bash
make validate              # exige datos: si faltan, falla
```

Equivale a `VALIDACION_REQUIERE_DATOS=1 .venv-validacion/bin/python -m pytest validacion_datos -rs`.
En Windows:

```powershell
$env:VALIDACION_REQUIERE_DATOS = "1"
.venv-validacion\Scripts\python -m pytest validacion_datos -rs
```

Sin `VALIDACION_REQUIERE_DATOS`, la misma orden omite las pruebas cuando faltan datos (ver la
tabla de abajo). Tarda menos de un minuto. P1 y P2 se validan con una muestra de hasta
200,000 filas por particion (semilla 42, ver `contrato.MUESTRA_POR_DEFECTO`) porque tienen
millones de filas; P3 se valida completo. El tamano de cada muestra queda en el resumen.

### Que revisa cada suite

Cada pregunta se parte con el mismo corte cronologico que el entrenamiento
(`entrenamiento` y `prueba`, fechas en `config.py`) y se declara explicitamente la etiqueta,
las variables y cuales son categoricas. DeepChecks no tiene que adivinarlas.

| Palabra de la rubrica | Check de DeepChecks | Suite | Estado |
|---|---|---|---|
| Valores faltantes | `PercentOfNulls` (condicion: 0 % de nulos, agregado a mano porque `data_integrity()` no lo trae), `MixedNulls` | integridad | Corre |
| Tipos de datos | `MixedDataTypes`, `SpecialCharacters`, `StringMismatch`, `StringLengthOutOfBounds` | integridad | Corre |
| Duplicados | `DataDuplicates` (sobre la muestra) y, con pandas sobre el archivo completo, llave sin nulos y unica: `(estacion_id, hora)` en P1, `(estacion_id, bloque_inicio)` en P2, `(estacion_id, periodo)` en P3 | integridad | Corre |
| Constantes | `IsSingleValue` | integridad | Corre |
| Redundancia y etiqueta | `FeatureFeatureCorrelation`, `FeatureLabelCorrelation`, `IdentifierLabelCorrelation`, `ConflictingLabels`, `OutlierSampleDetection` (sin condicion, solo informativo) | integridad | Corre |
| Deriva (drift) | `FeatureDrift`, `LabelDrift`, `NewCategoryTrainTest`, `MultivariateDrift` | entrenamiento contra prueba (`train_test_validation`) | Corre |
| Fuga temporal | `DateTrainTestLeakageOverlap`, `DateTrainTestLeakageDuplicates`, `TrainTestSamplesMix` | entrenamiento contra prueba | Corre |

Los checks que dependen de la etiqueta no aplican donde no hay una: P3 es clustering (no
tiene etiqueta) y `ConflictingLabels` no aplica a la regresion de P1. En la suite de
entrenamiento contra prueba tampoco aplican `NewLabelTrainTest` (regresion de P1 y P3),
`IndexTrainTestLeakage` (los conjuntos no tienen indice), los checks de etiqueta de P3 y los
de fecha de P3 (no tiene columna de tiempo; la particion viene de `periodo`). No se omiten en
silencio: se quitan de la suite y el resumen los lista como `no_aplica` con la razon.

### Entrenamiento contra prueba

DeepChecks recomienda `train_test_validation()` justo despues de partir los datos y antes de
tener un modelo, para comprobar que las particiones son representativas: clases con balance
parecido, sin cambios grandes en las distribuciones y sin fuga de datos. Nuestra particion es
cronologica, asi que esa idea se cumple a medias, y el resumen lo separa asi:

- **La fuga temporal debe ser cero y bloquea.** `DateTrainTestLeakageOverlap`,
  `DateTrainTestLeakageDuplicates` pasan en P1 y P2, y `TrainTestSamplesMix` en las tres
  preguntas. La prueba (jul a sep 2026) es posterior al entrenamiento (oct 2024 a mar 2026), con la
  validacion (abr a jun 2026) de por medio. `test_sin_fuga_temporal` lo exige.
- **La deriva se mide y se explica, no se oculta.** La prueba es el futuro, asi que difiere del
  entrenamiento por construccion (estacionalidad, estaciones nuevas). Cada deriva esperada se
  limita a las columnas documentadas: si deriva otra, el hallazgo vuelve a ser bloqueante.
- **Los umbrales son los de DeepChecks.** No se movio ninguno para poner todo en verde.

Para `MultivariateDrift` el limite lo fijan las variables que mas contribuyen al clasificador de
dominio (importancia de al menos 0.1, `IMPORTANCIA_MINIMA`; es un criterio nuestro, no de
DeepChecks). `DatasetsSizeComparison` compara las muestras, no las particiones completas: los
tamanos reales estan en el resumen. En esta suite `estacion_id` si se conserva, porque es lo
que permite a `NewCategoryTrainTest` mostrar las estaciones nuevas.
`estacion_id` se declara categorica solo para que `NewCategoryTrainTest` muestre las
estaciones que aparecen despues del corte; no es entrada de ningun modelo y se excluye de
la suite de integridad, donde solo genera correlaciones sin sentido.

### Esperado y bloqueante

Cada check termina en uno de seis estados. Solo dos bloquean:

| Estado | Significado | Bloquea |
|---|---|---|
| `pasa` | La condicion se cumple | No |
| `falla_esperada` | Falla, pero esta en `ESPERADAS` (`validacion_datos/suites.py`) con la razon verificada | No |
| `informativo` | El check no tiene condicion (solo reporte) | No |
| `no_aplica` | Se quito de la suite porque no corresponde a la pregunta; se declara | No |
| `falla_bloqueante` | Falla y no esta justificada, o falla una columna o par que no estaba en la lista | **Si** |
| `no_corrio` | El check se rompio o no pudo correr | **Si** |

No se aflojo ningun umbral para poner todo en verde. Una falla esperada se limita a las
columnas o pares concretos documentados; si aparece otra, vuelve a ser bloqueante.

| Pregunta | Check | Por que se espera | Alcance exacto |
|---|---|---|---|
| P1 | Feature-Feature Correlation | Las medias moviles de las 24 h y de los 7 dias previos se traslapan, su correlacion supera 0.9 por construccion | `salidas_media_24h_previa` con `salidas_media_7d` |
| P2 | Feature-Feature Correlation | `neto_rel_*` es la version absoluta dividida entre la capacidad de la estacion | `neto_lag_dia` con `neto_rel_lag_dia`; `neto_media_mismo_bloque_7d` con `neto_rel_media_mismo_bloque_7d` |
| P3 | Single Value in Column | Perfiles horarios de las 02 a las 04 h valen 0.0 en todas las estaciones de la particion (7 columnas en entrenamiento, 3 en prueba; el conjunto cambia). En prueba esas columnas valen como maximo 0.006 y el `StandardScaler` ajustado en entrenamiento las deja sin escalar | Solo las columnas `sal_*` y `lle_*` de las horas 02, 03 y 04 |

Son redundancia entre variables, no defectos de los datos. Para P3 la causa de fondo (cierre
del servicio o poca actividad nocturna) no esta verificada.

Deriva entrenamiento contra prueba (valores de la corrida con la version de datos del resumen;
el resumen lleva la razon completa de cada fila):

| Pregunta | Check | Por que se espera | Alcance exacto |
|---|---|---|---|
| P1 y P2 | New Category Train Test | 99 estaciones de la prueba no estan en el entrenamiento: 17.73 % (P1) y 18.52 % (P2) de las filas de prueba. La red crece | Solo `estacion_id` |
| P1 y P2 | Feature Drift | El entrenamiento cubre los 12 meses y la prueba solo jul a sep: `mes` difiere por construccion (Cramer's V 0.84). Ninguna otra variable pasa de 0.1 y la etiqueta no deriva (0.03 y 0.04) | Solo `mes` |
| P1 y P2 | Multivariate Drift | AUC de 0.92 y 0.93, con `mes` como unica variable que contribuye. Sin `mes` el AUC baja a 0.58 (puntaje 0.15 y 0.17, bajo el umbral de 0.25); lo que queda es `estacion_id` | Contribuyente: solo `mes` |
| P3 | Feature Drift | 8 de 99 columnas con puntaje de 0.21 a 0.42, cinco de ellas horas de madrugada. La causa no esta verificada: la prueba tiene 459 estaciones contra 370 (90 nuevas, 1 que ya no esta) | `lle_fds_h01`, `lle_lab_h01`, `lle_fds_h05`, `lle_lab_h05`, `sal_fds_h05`, `sal_fds_h17`, `sal_fds_h18`, `ratio_fin_semana` |
| P3 | Multivariate Drift | AUC de 0.83. Contribuyen `lle_fds_h01`, `sal_fds_h05` y `balance_neto`; la causa no esta verificada | Contribuyentes: esas tres |

Las estaciones nuevas y la estacionalidad pasan a las limitaciones del modelo: las estaciones
que abren despues del corte no estan en el entrenamiento, y la prueba cubre solo tres meses.

### Datos faltantes o desactualizados

Los datos no estan en Git, asi que "sin datos" es un estado de la instalacion, no un defecto
de los datos.

| Donde y como | Estado de los datos | Resultado |
|---|---|---|
| Ambiente principal: `make test`, CI | cualquiera | Las pruebas de validacion no se recolectan. La prueba guarda (`tests/test_contrato_validacion.py`) si corre y no necesita datos |
| Ambiente de validacion: `pytest validacion_datos` | faltan | **Omitida**, con el mensaje `corre make data && make features` |
| Ambiente de validacion, cualquier modo | desactualizados (`data_version` de `features_meta.json` distinta de la version del manifiesto) | **Falla**: se validarian datos que no corresponden al manifiesto |
| `make validate` (`VALIDACION_REQUIERE_DATOS=1`) | faltan | **Falla** con un mensaje claro: no se puede declarar "validado" sin datos |
| Ambiente de validacion | presentes | Corre; falla solo si hay hallazgos bloqueantes |
| Ambiente principal corre `pytest validacion_datos` por error | - | Se omite con un aviso en lugar de un `ImportError` (`pytest.importorskip`) |

### Donde quedan los reportes

| Que | Donde | En Git |
|---|---|---|
| Reportes HTML de DeepChecks | `reports/validacion/<p1\|p2\|p3>/<suite>_<particion>.html` | No (`.gitignore`): incluyen filas de datos y se regeneran en cada corrida |
| Resumen | [`reports/validacion/resumen_validacion.md`](reports/validacion/resumen_validacion.md) | Si: solo metadatos (version de datos, tamano de muestra, estado de cada check y explicacion de cada falla esperada) |

El resumen se reescribe en cada corrida. Compara su `Version de datos` con la de
`references/manifiesto_datos.json` para saber si esta al dia.

## Flujo de trabajo con Git y GitHub

Seguimos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. La rama `main` esta protegida: nadie sube directo a `main`, todo cambio entra por Pull
   Request con al menos 1 aprobacion de otra persona del equipo.
2. Se crea una rama corta desde `main`, se hacen commits pequenos y se abre un Pull Request
   con la plantilla de `.github/pull_request_template.md` completa.
3. Cada integrante abre al menos un Pull Request y revisa al menos uno de otra persona. Una
   revision util nombra algo concreto: un comando que se corrio, un riesgo, una pregunta.
4. Se integra con _squash and merge_ y se borra la rama.

**Ramas:** `tipo/descripcion-corta-en-minusculas`, por ejemplo `feat/ingesta-reproducible`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/) en la
forma `tipo(alcance): descripcion en imperativo`, por ejemplo
`feat(dataset): ingesta reproducible y versionada de MiBici`.

| Tipo | Se usa para |
|---|---|
| `feat` | Funcionalidad nueva |
| `fix` | Correccion de un error |
| `docs` | Documentacion, README, ML Canvas, diagramas, reportes |
| `data` | Cambios al manifiesto de datos (`references/manifiesto_datos.json`) y a su resumen. Rama `data/*`, por ejemplo `data/manifiesto-inicial` |
| `chore` / `refactor` | Configuracion, estructura o limpieza sin cambiar comportamiento |

Nunca se versionan datos de MiBici, modelos serializados, carpetas de MLflow (`mlruns/`,
`mlartifacts/`, `mlflow.db`) ni credenciales (`.env`); el `.gitignore` los excluye.

## Buenas practicas

Practicas obligatorias de la entrega y como las aplica el repositorio:

| Ambito | Practica | Como la aplicamos | Donde se verifica |
|---|---|---|---|
| MLflow | Un experimento por proyecto y runs con nombre descriptivo | Experimento `mibici-equipo18`; runs `p<n>-<etapa>-<modelo>`, p. ej. `p1-baseline-rezago_semanal` | Interfaz de MLflow |
| MLflow | Registrar lo necesario para reproducir el run | Parametros `features.mes_inicio`, `features.mes_fin`, `data_version`, `semilla`, `modelo` e hiperparametros | Parametros del run |
| MLflow | Ligar cada run con su codigo y su pregunta | Etiquetas `git_commit`, `git_sucio`, `pregunta`, `etapa`, `familia_modelo`, `equipo` | Etiquetas del run |
| MLflow | Metricas con el conjunto en el nombre | `validacion_mae`, `prueba_f1_macro`, `prueba_estabilidad_ari`... | Metricas del run |
| MLflow | Sin credenciales ni datos personales | Usuario, genero y anio de nacimiento no se leen del CSV; la URI de MLflow va en `.env`; los artefactos solo tienen agregados por estacion | Interfaz de MLflow y codigo |
| Estructura | Notebooks solo para exploracion, con la convencion de la plantilla | `notebooks/1.0-ja-exploracion-viajes.ipynb`; la logica vive en el modulo | Carpeta `notebooks/` |
| Estructura | Configuracion en un solo lugar | Rutas, fechas, semillas, URL e hiperparametros solo en `config.py` | `config.py` |
| Estructura | Dependencias fijas y entorno reproducible desde un clon limpio | `requirements.txt` con `==` y Python 3.14; instalacion documentada arriba | Archivo de dependencias y README |
| Datos y Git | Ningun dato, modelo, carpeta de MLflow ni credencial en Git | `.gitignore` excluye `data/`, CSV, Parquet, `mlruns/`, `mlartifacts/`, `mlflow.db`, modelos y `.env` | `.gitignore` e historial |
| Datos y Git | Todo cambio entra a `main` por Pull Request revisado | `main` protegida con 1 aprobacion minima; squash and merge | Pestana Pull Requests |

## Integrantes y roles

| Integrante | GitHub | Rol |
|---|---|---|
| Alberto Labrada Soto | [@betolabrada](https://github.com/betolabrada) | Repositorio y flujo de trabajo, manifiesto de datos, README |
| Jesus Alvizo | [@JesusAlvizo26](https://github.com/JesusAlvizo26) | Configuracion, ingesta reproducible, feature engineering y propuesta |
| Lizeth Ruiz | [@LizethRuiz](https://github.com/LizethRuiz) | Base del experimento en MLflow, P1 (regresion) y P2 (clasificacion) |
| Jessica Paola Ruiz | [@Jessicapaolarc](https://github.com/Jessicapaolarc) | P3 (clustering) |

## Documentacion

- [Propuesta de solucion, entrega 1](reports/propuesta_entrega1.md)
- [ML Canvas](references/ml_canvas.md) - [Arquitectura](references/arquitectura.md)
- [Resumen de los datos](reports/resumen_datos.md) - [Manifiesto de datos](references/manifiesto_datos.json)
