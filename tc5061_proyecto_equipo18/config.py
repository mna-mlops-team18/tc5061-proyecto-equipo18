"""Configuracion central del proyecto MiBici (equipo 18).

Este es el UNICO lugar donde se definen rutas, fechas, semillas, fuentes de datos y
parametros de los experimentos. El resto de los modulos importa sus valores desde aqui,
asi que cambiar una decision del equipo (por ejemplo, el rango de fechas) es cambiar una
sola linea en un solo archivo.

Las variables de entorno (archivo ``.env``, que nunca se versiona) solo se usan para
valores que dependen de la maquina, como la URI de seguimiento de MLflow.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carga variables de entorno desde .env si existe (ver .env.example)
load_dotenv()

# =============================================================================
# Rutas
# =============================================================================
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REFERENCES_DIR = PROJ_ROOT / "references"

# Archivos que escriben y leen los modulos (todos dentro de data/, ignorado por Git,
# salvo el manifiesto y el resumen, que solo contienen metadatos y si se versionan).
RAW_VIAJES_DIR = RAW_DATA_DIR / "viajes"  # CSV originales, uno por mes
VIAJES_DIR = PROCESSED_DATA_DIR / "viajes"  # Parquet limpio, uno por mes
ESTACIONES_PARQUET = PROCESSED_DATA_DIR / "estaciones.parquet"
CONTEOS_HORA_PARQUET = INTERIM_DATA_DIR / "conteos_estacion_hora.parquet"
DATASET_P1 = PROCESSED_DATA_DIR / "p1_demanda_estacion_hora.parquet"
DATASET_P2 = PROCESSED_DATA_DIR / "p2_riesgo_estacion_bloque.parquet"
DATASET_P3 = PROCESSED_DATA_DIR / "p3_perfiles_estacion.parquet"
FEATURES_META = PROCESSED_DATA_DIR / "features_meta.json"
MANIFIESTO_DATOS = REFERENCES_DIR / "manifiesto_datos.json"
RESUMEN_DATOS_MD = REPORTS_DIR / "resumen_datos.md"
RESULTADOS_ENTREGA2_MD = REPORTS_DIR / "resultados_entrega2.md"
EDA_FIGURAS_DIR = FIGURES_DIR / "eda"
# Archivos que cada run guarda en config/ para poder reproducirlo (rutas relativas a la raiz)
ARCHIVOS_CONFIGURACION = (
    "MLproject",
    "python_env.yaml",
    "requirements.txt",
    "tc5061_proyecto_equipo18/config.py",
)


def ruta_viajes_mes(mes: str) -> Path:
    """Parquet limpio de un mes ``AAAA-MM`` (lo escribe dataset.py y lo lee features.py)."""
    return VIAJES_DIR / f"viajes_{mes.replace('-', '_')}.parquet"


def ruta_predicciones(pregunta: str, particion: str) -> Path:
    """Predicciones que escribe predict.py para una pregunta y una particion."""
    return PROCESSED_DATA_DIR / f"predicciones_{pregunta.lower()}_{particion}.parquet"


# =============================================================================
# Rango de fechas y particion temporal (decision del equipo; ver la propuesta)
# =============================================================================
# 24 meses recientes: dos ciclos anuales completos, red de estaciones actual,
# sin el periodo de pandemia y sin los meses que el portal no publica (2023-04 y 2023-05).
MES_INICIO = "2024-10"  # primer mes que se descarga (inclusive)
MES_FIN = "2026-09"  # ultimo mes que se descarga (inclusive)

# Particion temporal: entrenamiento -> validacion -> prueba, siempre hacia adelante.
FIN_ENTRENAMIENTO = "2026-03-31"  # entrenamiento: MES_INICIO a esta fecha (18 meses)
FIN_VALIDACION = "2026-06-30"  # validacion: abr-jun 2026; prueba: jul-sep 2026

# =============================================================================
# Semilla unica para todo lo aleatorio (muestreo, modelos, clustering)
# =============================================================================
SEMILLA = 42

# =============================================================================
# Fuente de datos: portal de datos abiertos de MiBici
# =============================================================================
URL_PORTAL_DATOS = "https://mibici.net/es/datos-abiertos/"
URL_BASE_OFICIAL = "https://mibici.net/site/assets/files/"
# Se puede descargar desde un espejo con la variable de entorno MIBICI_URL_BASE;
# el manifiesto siempre registra la URL oficial.
URL_BASE_ARCHIVOS = os.getenv("MIBICI_URL_BASE", URL_BASE_OFICIAL)
USER_AGENT = "tc5061-equipo18-mlops/0.1 (proyecto academico, Tec de Monterrey)"

RUTA_ESTACIONES = "1118/nomenclatura_2026_09.csv"

# El identificador numerico de cada archivo no se puede deducir del mes, por eso se
# guarda el catalogo completo tal como aparece en el portal (consultado en oct. 2026).
RUTAS_VIAJES: dict[str, str | None] = {
    "2014-12": "1079/datos_abiertos_2014_12.csv",
    "2015-01": "1084/datos_abiertos_2015_01.csv",
    "2015-02": "1097/datos_abiertos_2015_02.csv",
    "2015-03": "1098/datos_abiertos_2015_03.csv",
    "2015-04": "1099/datos_abiertos_2015_04.csv",
    "2015-05": "1100/datos_abiertos_2015_05.csv",
    "2015-06": "1101/datos_abiertos_2015_06.csv",
    "2015-07": "1102/datos_abiertos_2015_07.csv",
    "2015-08": "1103/datos_abiertos_2015_08.csv",
    "2015-09": "1104/datos_abiertos_2015_09.csv",
    "2015-10": "1105/datos_abiertos_2015_10.csv",
    "2015-11": "1106/datos_abiertos_2015_11.csv",
    "2015-12": "1107/datos_abiertos_2015_12.csv",
    "2016-01": "1020/datos_abiertos_2016_01-1.csv",
    "2016-02": "1021/datos_abiertos_2016_02.csv",
    "2016-03": "1023/datos_abiertos_2016_03.csv",
    "2016-04": "1029/datos_abiertos_2016_04.csv",
    "2016-05": "1058/datos_abiertos_2016_05.csv",
    "2016-06": "1080/datos_abiertos_2016_06.csv",
    "2016-07": "1081/datos_abiertos_2016_07.csv",
    "2016-08": "1082/datos_abiertos_2016_08.csv",
    "2016-09": "1083/datos_abiertos_2016_09.csv",
    "2016-10": "1108/datos_abiertos_2016_10.csv",
    "2016-11": "1109/datos_abiertos_2016_11.csv",
    "2016-12": "1110/datos_abiertos_2016_12.csv",
    "2017-01": "1022/datos_abiertos_2017_01.csv",
    "2017-02": "1111/datos_abiertos_2017_02.csv",
    "2017-03": "1112/datos_abiertos_2017_03.csv",
    "2017-04": "1115/datos_abiertos_2017_04-1.csv",
    "2017-05": "1116/datos_abiertos_2017_05.csv",
    "2017-06": "1119/datos_abiertos_2017_06.csv",
    "2017-07": "1120/datos_abiertos_2017_07.csv",
    "2017-08": "1122/datos_abiertos_2017_08.csv",
    "2017-09": "1123/datos_abiertos_2017_09.csv",
    "2017-10": "1124/datos_abiertos_2017_10.csv",
    "2017-11": "1197/datos_abiertos_2017_11.csv",
    "2017-12": "1198/datos_abiertos_2017_12.csv",
    "2018-01": "1200/datos_abiertos_2018_01.csv",
    "2018-02": "1201/datos_abiertos_2018_02.csv",
    "2018-03": "1202/datos_abiertos_2018_03.csv",
    "2018-04": "1203/datos_abiertos_2018_04.csv",
    "2018-05": "1204/datos_abiertos_2018_05.csv",
    "2018-06": "1205/datos_abiertos_2018_06.csv",
    "2018-07": "1206/datos_abiertos_2018_07.csv",
    "2018-08": "1207/datos_abiertos_2018_08.csv",
    "2018-09": "1208/datos_abiertos_2018_09.csv",
    "2018-10": "1209/datos_abiertos_2018_10.csv",
    "2018-11": "1210/datos_abiertos_2018_11.csv",
    "2018-12": "1211/datos_abiertos_2018_12.csv",
    "2019-01": "1214/datos_abiertos_2019_01.csv",
    "2019-02": "1217/datos_abiertos_2019_02.csv",
    "2019-03": "1218/datos_abiertos_2019_03.csv",
    "2019-04": "1219/datos_abiertos_2019_04.csv",
    "2019-05": "1220/datos_abiertos_2019_05.csv",
    "2019-06": "1221/datos_abiertos_2019_06.csv",
    "2019-07": "1222/datos_abiertos_2019_07.csv",
    "2019-08": "1223/datos_abiertos_2019_08.csv",
    "2019-09": "1224/datos_abiertos_2019_09.csv",
    "2019-10": "1225/datos_abiertos_2019_10.csv",
    "2019-11": "1227/datos_abiertos_2019_11.csv",
    "2019-12": "1228/datos_abiertos_2019_12.csv",
    "2020-01": "1230/datos_abiertos_2020_01.csv",
    "2020-02": "1231/datos_abiertos_2020_02.csv",
    "2020-03": "1232/datos_abiertos_2020_03.csv",
    "2020-04": "1235/datos_abiertos_2020_04.csv",
    "2020-05": "1236/datos_abiertos_2020_05.csv",
    "2020-06": "1237/datos_abiertos_2020_06.csv",
    "2020-07": "1238/datos_abiertos_2020_07.csv",
    "2020-08": "1239/datos_abiertos_2020_08.csv",
    "2020-09": "1240/datos_abiertos_2020_09.csv",
    "2020-10": "1241/datos_abiertos_2020_10.csv",
    "2020-11": "1317/datos_abiertos_2020_11.csv",
    "2020-12": "1318/datos_abiertos_2020_12.csv",
    "2021-01": "1320/datos_abiertos_2021_01.csv",
    "2021-02": "1323/datos_abiertos_2021_02.csv",
    "2021-03": "1322/datos_abiertos_2021_03.csv",
    "2021-04": "2129/datos_abiertos_2021_04.csv",
    "2021-05": "3292/datos_abiertos_2021_05.csv",
    "2021-06": "4572/datos_abiertos_2021_06.csv",
    "2021-07": "5728/datos_abiertos_2021_07.csv",
    "2021-08": "7073/datos_abiertos_2021_08.csv",
    "2021-09": "8461/datos_abiertos_2021_09.csv",
    "2021-10": "10088/datos_abiertos_2021_10.csv",
    "2021-11": "11538/datos_abiertos_2021_11.csv",
    "2021-12": "12780/datos_abiertos_2021_12.csv",
    "2022-01": "14797/datos_abiertos_2022_01.csv",
    "2022-02": "16831/datos_abiertos_2022_02.csv",
    "2022-03": "19034/datos_abiertos_2022_03.csv",
    "2022-04": "20338/datos_abiertos_2022_04.csv",
    "2022-05": "21842/datos_abiertos_2022_05.csv",
    "2022-06": "23473/datos_abiertos_2022_06.csv",
    "2022-07": "25235/datos_abiertos_2022_07.csv",
    "2022-08": "27484/datos_abiertos_2022_08.csv",
    "2022-09": "29663/datos_abiertos_2022_09.csv",
    "2022-10": "31507/datos_abiertos_2022_10.csv",
    "2022-11": "33115/datos_abiertos_2022_11.csv",
    "2022-12": "34432/datos_abiertos_2022_12.csv",
    "2023-01": "36762/datos_abiertos_2023_01.csv",
    "2023-02": "40197/datos_abiertos_2023_02.csv",
    "2023-03": "43035/datos_abiertos_2023_03.csv",
    "2023-04": None,  # el portal no publica enlace para este mes
    "2023-05": None,  # el portal no publica enlace para este mes
    "2023-06": "48116/datos_abiertos_2023_06.csv",
    "2023-07": "49448/datos_abiertos_2023_07.csv",
    "2023-08": "51295/datos_abiertos_2023_08.csv",
    "2023-09": "53344/datos_abiertos_2023_09.csv",
    "2023-10": "54935/datos_abiertos_2023_10.csv",
    "2023-11": "57344/datos_abiertos_2023_11.csv",
    "2023-12": "58715/datos_abiertos_2023_12.csv",
    "2024-01": "61450/datos_abiertos_2024_01.csv",
    "2024-02": "77251/datos_abiertos_2024_02.csv",
    "2024-03": "77252/datos_abiertos_2024_03.csv",
    "2024-04": "80028/datos_abiertos_2024_04.csv",
    "2024-05": "81663/datos_abiertos_2024_05.csv",
    "2024-06": "83901/datos_abiertos_2024_06.csv",
    "2024-07": "85500/datos_abiertos_2024_07.csv",
    "2024-08": "87665/datos_abiertos_2024_08.csv",
    "2024-09": "89736/datos_abiertos_2024_09.csv",
    "2024-10": "91758/datos_abiertos_2024_10.csv",
    "2024-11": "93516/datos_abiertos_2024_11.csv",
    "2024-12": "94962/datos_abiertos_2024_12.csv",
    "2025-01": "97653/datos_abiertos_2025_01.csv",
    "2025-02": "99879/datos_abiertos_2025_02.csv",
    "2025-03": "101986/datos_abiertos_2025_03.csv",
    "2025-04": "103813/datos_abiertos_2025_04.csv",
    "2025-05": "105513/datos_abiertos_2025_05.csv",
    "2025-06": "107088/datos_abiertos_2025_06.csv",
    "2025-07": "108898/datos_abiertos_2025_07.csv",
    "2025-08": "111021/datos_abiertos_2025_08.csv",
    "2025-09": "113216/datos_abiertos_2025_09.csv",
    "2025-10": "115104/datos_abiertos_2025_10.csv",
    "2025-11": "116773/datos_abiertos_2025_11.csv",
    "2025-12": "118153/datos_abiertos_2025_12.csv",
    "2026-01": "120163/datos_abiertos_2026_01.csv",
    "2026-02": "121761/datos_abiertos_2026_02.csv",
    "2026-03": "123236/datos_abiertos_2026_03.csv",
    "2026-04": "124927/datos_abiertos_2026_04.csv",
    "2026-05": "125943/datos_abiertos_2026_05.csv",
    "2026-06": "127734/datos_abiertos_2026_06.csv",
    "2026-07": "129006/datos_abiertos_2026_07.csv",
    "2026-08": "130362/datos_abiertos_2026_08.csv",
    "2026-09": "131825/datos_abiertos_2026_09.csv",
}

# =============================================================================
# Limpieza de viajes (features.filtrar_viajes_validos)
# =============================================================================
DURACION_MIN_SEG = 60  # menos de 1 minuto: desanclaje fallido o prueba de bicicleta
DURACION_MAX_SEG = 4 * 60 * 60  # mas de 4 horas: bicicleta mal anclada o registro anomalo

# =============================================================================
# P1 - Regresion: salidas por estacion y hora, con un dia de anticipacion
# =============================================================================
P1_HORIZONTE_H = 24  # la prediccion se hace al cierre del dia anterior
P1_REZAGOS_H = (24, 48, 168, 336)  # todos >= horizonte: nada posterior a la prediccion
P1_MAX_FILAS_ENTRENAMIENTO = 3_000_000  # muestreo con SEMILLA para acotar memoria
P1_FILAS_IMPORTANCIA = 50_000  # filas de validacion para la importancia por permutacion
P1_HIPERPARAMETROS = {
    "loss": "poisson",  # conteos no negativos
    "learning_rate": 0.08,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
}
# Catalogo de P1: nombre del modelo -> familia e hiperparametros por defecto. Son los valores
# que usa `mlflow run . -e train -P pregunta=p1 -P modelo=<nombre>`; cualquiera se cambia en
# una corrida con -P <hiperparametro>=<valor>.
P1_MODELOS = {
    # Baseline de la entrega 1: salidas de la misma hora hace 7 dias (regla, sin entrenamiento)
    "rezago_semanal": {"familia": "regla", "baseline": True, "hiperparametros": {}},
    # Candidato de la entrega 1: gradient boosting con perdida Poisson
    "hgb_poisson": {"familia": "gradient_boosting", "hiperparametros": P1_HIPERPARAMETROS},
    # Mismo algoritmo con perdida cuadratica: mide cuanto aporta la perdida Poisson
    "hgb_cuadratico": {
        "familia": "gradient_boosting",
        "hiperparametros": {**P1_HIPERPARAMETROS, "loss": "squared_error"},
    },
    # Modelo lineal generalizado Poisson con hora, dia y mes en one-hot: barato e interpretable
    "glm_poisson": {"familia": "lineal", "hiperparametros": {"alpha": 0.0001, "max_iter": 100}},
}
# Busqueda de hiperparametros (nested runs): una corrida hija por combinacion de la rejilla
P1_BUSQUEDA = {
    "modelo": "hgb_poisson",
    "rejilla": {"learning_rate": [0.05, 0.15], "max_leaf_nodes": [31, 127]},
}

# =============================================================================
# P2 - Clasificacion: riesgo de vaciarse o llenarse en un bloque horario
# =============================================================================
P2_HORAS_BLOQUE = 3  # bloques 00-03, 03-06, ..., 21-24
P2_UMBRAL_RIESGO = 0.5  # flujo neto >= 50 % de la capacidad dentro del bloque
P2_CLASES = {0: "sin_riesgo", 1: "riesgo_vaciado", 2: "riesgo_llenado"}
P2_FILAS_IMPORTANCIA = 50_000  # filas de validacion para la importancia por permutacion
P2_HIPERPARAMETROS = {
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "class_weight": "balanced",  # las clases de riesgo son minoritarias
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
}
# Catalogo de P2 (mismo formato que P1_MODELOS). `peso_riesgo` es el peso de las clases de
# riesgo frente a "sin riesgo": 1 = sin ponderar; "balanced" equivale a cerca de 65, porque
# cada clase de riesgo es el 1.5 % de los bloques.
P2_MODELOS = {
    # Baseline de la entrega 1: clase mas frecuente del mismo bloque en los 7 dias previos
    "frecuencia_mismo_bloque": {"familia": "regla", "baseline": True, "hiperparametros": {}},
    # Candidato de la entrega 1: gradient boosting con pesos de clase balanceados
    "hgb_balanceado": {"familia": "gradient_boosting", "hiperparametros": P2_HIPERPARAMETROS},
    # Mismo algoritmo con un peso intermedio: menos alertas falsas que "balanced"
    "hgb_ponderado": {
        "familia": "gradient_boosting",
        "hiperparametros": {
            **{
                clave: valor
                for clave, valor in P2_HIPERPARAMETROS.items()
                if clave != "class_weight"
            },
            "peso_riesgo": 5.0,
        },
    },
    # Regresion logistica multinomial con variables escaladas: referencia lineal y barata
    "logistica": {
        "familia": "lineal",
        "hiperparametros": {"C": 1.0, "peso_riesgo": 5.0, "max_iter": 1000},
    },
}
# La rejilla incluye el 5 por defecto de hgb_ponderado: asi el candidato es un punto de la busqueda
P2_BUSQUEDA = {"modelo": "hgb_ponderado", "rejilla": {"peso_riesgo": [1.0, 3.0, 5.0, 10.0, 30.0]}}

# =============================================================================
# P3 - Clustering: grupos de estaciones con patrones de uso parecidos
# =============================================================================
P3_MIN_VIAJES = 500  # salidas + llegadas minimas en el periodo para perfilar una estacion
# Espacio comun en que se agrupa y se evalua (escalado + PCA ajustados con entrenamiento):
# el mismo para todos los modelos, para que sus metricas sean comparables.
P3_VARIANZA_PCA = 0.90
# Catalogo de P3 (mismo formato que P1_MODELOS)
P3_MODELOS = {
    # Baseline de la entrega 1: grupos por cuartiles del volumen diario
    "cuartiles_volumen": {"familia": "regla", "baseline": True, "hiperparametros": {"grupos": 4}},
    # Candidato de la entrega 1: K-means; k = 3 fue el de mayor silhouette en la entrega 1
    "kmeans_pca": {"familia": "kmeans", "hiperparametros": {"n_clusters": 3, "n_init": 20}},
    # Mezcla gaussiana: grupos con forma y tamanio distintos (covarianza diagonal)
    "gmm_pca": {
        "familia": "mezcla_gaussiana",
        "hiperparametros": {"n_components": 3, "covariance_type": "diag", "n_init": 5},
    },
    # Jerarquico aglomerativo (Ward); las estaciones nuevas van al centroide mas cercano
    "jerarquico_pca": {
        "familia": "jerarquico",
        "hiperparametros": {"n_clusters": 3, "linkage": "ward"},
    },
}
P3_BUSQUEDA = {"modelo": "kmeans_pca", "rejilla": {"n_clusters": [2, 3, 4, 5, 6, 7, 8]}}

# =============================================================================
# MLflow
# =============================================================================
EQUIPO = "equipo18"
MLFLOW_EXPERIMENTO = "mibici-equipo18"
# Servidor local de tracking con base de datos (lo pide el Model Registry). Se levanta con
#   mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# y se puede cambiar con la variable de entorno MLFLOW_TRACKING_URI (o en .env).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# Solo sin servidor (por ejemplo, en las pruebas): artefactos en mlruns/, ignorado por Git
MLFLOW_ARTEFACTOS_DIR = PROJ_ROOT / "mlruns"
# MLflow guarda los modelos de scikit-learn en formato skops (no ejecuta codigo al cargarlos).
# Desde skops 0.15, los arboles de HistGradientBoosting no son "de confianza" por defecto;
# como solo cargamos modelos que entrenamos nosotros, se declaran aqui de forma explicita.
SKOPS_TIPOS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

# Catalogos y busquedas de las tres preguntas (definidos arriba, en la seccion de cada una)
MODELOS = {"P1": P1_MODELOS, "P2": P2_MODELOS, "P3": P3_MODELOS}
BUSQUEDAS = {"P1": P1_BUSQUEDA, "P2": P2_BUSQUEDA, "P3": P3_BUSQUEDA}
# Metrica con la que se compara y se elige cada modelo (las de la entrega 1) y su sentido
METRICAS_SELECCION = {
    "P1": ("validacion_mae", "min"),
    "P2": ("validacion_f1_macro", "max"),
    "P3": ("entrenamiento_silhouette", "max"),
}

# =============================================================================
# Model Registry: un modelo registrado por pregunta; la version en produccion lleva el
# alias "champion" y, opcionalmente, la siguiente mejor lleva "challenger"
# =============================================================================
MODELOS_REGISTRADOS = {
    "P1": "mibici-equipo18-p1-demanda",
    "P2": "mibici-equipo18-p2-riesgo",
    "P3": "mibici-equipo18-p3-grupos",
}
DESCRIPCIONES_REGISTRO = {
    "P1": (
        "P1 - Demanda de MiBici (regresion). Predice cuantos viajes saldran de cada estacion en "
        "cada hora del dia siguiente. Entrada: variables de features.COLUMNAS_P1, calculadas al "
        "cierre del dia anterior. Salida: viajes esperados. Lo usa el operador para planear la "
        "flota. Se elige con el MAE de validacion (viajes por estacion y hora)."
    ),
    "P2": (
        "P2 - Riesgo de desabasto de MiBici (clasificacion). Por estacion y bloque de 3 h "
        "predice 0 = sin riesgo, 1 = riesgo de vaciado o 2 = riesgo de llenado. Entrada: "
        "variables de features.COLUMNAS_P2 al inicio del bloque. Lo usa el equipo de "
        "rebalanceo para priorizar estaciones. Se elige con el F1 macro de validacion."
    ),
    "P3": (
        "P3 - Grupos de estaciones de MiBici (clustering). Asigna cada estacion a un grupo "
        "segun su perfil de uso: reparto horario de salidas y llegadas, volumen, relacion fin "
        "de semana/laboral y balance. El escalado y la PCA van dentro del modelo. Lo usa "
        "planeacion para zonificar el rebalanceo y la expansion. Se elige con la silhouette y "
        "la estabilidad (ARI) de los grupos."
    ),
}
ALIAS_PRODUCCION = "champion"
ALIAS_RETADOR = "challenger"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
