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
|-- Makefile                      <- Atajos: make data, make verify, make features, make train, make test
|-- MLproject                     <- MLflow Project: un run del experimento con `mlflow run`
|-- python_env.yaml               <- Ambiente del MLflow Project
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
    |-- dataset.py                <- Ingesta: descarga, normaliza, Parquet y manifiesto
    |-- features.py               <- Variables de P1, P2 y P3 sin fuga de informacion
    |-- plots.py                  <- Figuras y mapas
    +-- modeling/
        |-- comun.py              <- Registro comun en MLflow: run, etiquetas, parametros, metricas, modelo
        |-- p1_regresion.py       <- P1: rezago semanal vs. HistGradientBoosting Poisson
        |-- p2_clasificacion.py   <- P2: frecuencia del bloque vs. HistGradientBoosting balanceado
        |-- p3_clustering.py      <- P3: cuartiles de volumen vs. K-means sobre PCA
        |-- train.py              <- Orquesta el experimento base (--todos, --pregunta, --modelo)
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
make data       # python -m tc5061_proyecto_equipo18.dataset ingest
make verify     # python -m tc5061_proyecto_equipo18.dataset verify
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
make features   # python -m tc5061_proyecto_equipo18.features
make test       # pytest
```

## Experimento base en MLflow

Un experimento, `mibici-equipo18`, con un baseline y un candidato por pregunta (6 runs). Se
corre desde el codigo del repositorio, despues de la ingesta y de `make features`:

```bash
make train      # python -m tc5061_proyecto_equipo18.modeling.train --todos
python -m tc5061_proyecto_equipo18.modeling.train --pregunta P2 --modelo candidato   # un solo run
make mlflow-ui  # mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://localhost:5000
```

Tambien como MLflow Project (un run por comando, desde la raiz del repo):

```bash
mlflow run . -P pregunta=P1 -P tipo_modelo=candidato --env-manager local --experiment-name mibici-equipo18
```

| Pregunta | Baseline (run) | Candidato (run) | Metricas |
|---|---|---|---|
| P1 | `p1-baseline-rezago_semanal`: salidas de la misma hora hace 7 dias | `p1-candidato-hgb_poisson`: HistGradientBoosting con perdida Poisson | `mae`, `rmse`, `wape`, `devianza_poisson`, `r2` |
| P2 | `p2-baseline-frecuencia_mismo_bloque`: clase mas frecuente del bloque en 7 dias | `p2-candidato-hgb_balanceado`: HistGradientBoosting con pesos balanceados | `f1_macro`, `balanced_accuracy`, `recall_*`, `precision_*`, `pr_auc_*` |
| P3 | `p3-baseline-cuartiles_volumen`: cuartiles del volumen diario | `p3-candidato-kmeans_pca`: K-means sobre PCA, k por silhouette | `silhouette`, `davies_bouldin`, `calinski_harabasz`, `estabilidad_ari` |

Cada run registra (ver `modeling/comun.py`):

- **Parametros:** semilla, parametros de features (rango de fechas, cortes de la particion,
  filtros, horizonte, rezagos, bloque y umbral de riesgo), `data_version`, `modelo` y sus
  hiperparametros.
- **Metricas** con el conjunto en el nombre: `entrenamiento_*`, `validacion_*`, `prueba_*`.
- **Etiquetas:** `equipo`, `pregunta`, `rol` (baseline o candidato), `modelo`, `data_version`,
  `git_commit` y `git_sucio`.
- **Artefactos:** figuras de diagnostico, `features_meta.json`, el manifiesto de datos y, en
  los candidatos, el modelo en formato MLflow (skops). P3 agrega el mapa interactivo de grupos
  y las tablas de resumen y asignaciones.

Para que el `git_commit` de cada run sea fiel, corre el experimento sin cambios pendientes en
archivos versionados (`git_sucio = false`). La propuesta completa y la lectura de resultados
estan en `reports/propuesta_entrega1.md`.

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
| MLflow | Un experimento por proyecto y runs con nombre descriptivo | Experimento `mibici-equipo18`; runs `p<n>-<rol>-<modelo>`, p. ej. `p1-baseline-rezago_semanal` | Interfaz de MLflow |
| MLflow | Registrar lo necesario para reproducir el run | Parametros `features.mes_inicio`, `features.mes_fin`, `data_version`, `semilla`, `modelo` e hiperparametros | Parametros del run |
| MLflow | Ligar cada run con su codigo y su pregunta | Etiquetas `git_commit`, `git_sucio`, `pregunta`, `rol`, `equipo` | Etiquetas del run |
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
