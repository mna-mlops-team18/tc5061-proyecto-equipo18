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
| `test` | Pruebas automatizadas, validacion de datos y chequeos de seguridad |
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
| Seguridad | Sin vulnerabilidades conocidas en dependencias ni en el codigo | `make security`: `pip-audit`, `bandit` y `tests/test_seguridad.py` (credenciales, archivos sensibles y `.gitignore`) | `reports/seguridad/resumen_seguridad.md` |
| Datos y Git | Todo cambio entra a `main` por Pull Request revisado | `main` protegida con 1 aprobacion minima; squash and merge | Pestana Pull Requests |

## Seguridad

Chequeos basicos de seguridad, con un solo comando desde el ambiente `.venv` activado:

```bash
make security
```

Equivale a:

```bash
pip-audit --skip-editable --strict                 # vulnerabilidades conocidas en dependencias
bandit -r tc5061_proyecto_equipo18                 # analisis estatico del codigo
python -m pytest tests/test_seguridad.py           # credenciales, archivos sensibles y .gitignore
```

- **Dependencias (`pip-audit`).** Revisa todos los paquetes instalados en `.venv`, incluidas las
  dependencias de las dependencias, contra la base de vulnerabilidades de PyPI (OSV).
  `--skip-editable` omite el propio paquete del proyecto (no esta publicado) y `--strict` falla
  si algun paquete no se puede revisar. Corre sobre el ambiente instalado con
  `requirements.txt`; actualiza `pip` antes (`make requirements` ya lo hace), porque las
  versiones viejas de `pip` tienen vulnerabilidades propias.
- **Codigo (`bandit`).** Busca patrones inseguros en el modulo (y en `validacion_datos/` cuando
  exista). Las pruebas no se analizan: usan `assert` a proposito. Cada excepcion se marca en el
  codigo con `# nosec <regla>` y su razon; hoy solo hay dos, en la funcion que consulta el
  commit con `git` (argumentos fijos, sin shell y con la ruta absoluta del ejecutable).
- **Credenciales y archivos sensibles (`tests/test_seguridad.py`).** Comprueba que `.gitignore`
  excluya `.env`, la base y los artefactos de MLflow, los datos de MiBici y los modelos
  serializados; que ningun archivo versionado sea de un tipo sensible (llaves, `.env`, bases de
  datos, Parquet, modelos); que ningun archivo versionado tenga algo con forma de credencial
  (llaves de nube, tokens, contrasenas asignadas); y que `.env.example` no traiga valores. Estas
  pruebas tambien corren con `make test`.
- **Credenciales en variables de entorno.** El codigo no tiene credenciales: la URI de MLflow y
  el espejo de datos se leen de variables de entorno (`.env`, que no se versiona; la plantilla es
  `.env.example`). Si se usa un servidor remoto de MLflow, `MLFLOW_TRACKING_USERNAME` y
  `MLFLOW_TRACKING_PASSWORD` van solo en `.env`.
- **Datos personales.** Las columnas de usuario, genero y anio de nacimiento no se leen del CSV
  (`tests/test_dataset.py` lo verifica) y el notebook se versiona sin salidas.

Los resultados de la ultima revision (incluido el historial completo de Git) estan en
`reports/seguridad/resumen_seguridad.md`.

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
