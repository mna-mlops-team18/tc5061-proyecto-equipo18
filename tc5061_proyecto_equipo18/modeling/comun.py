"""
Contrato de cada modulo de pregunta (``p1_regresion.py``, ``p2_clasificacion.py`` y
``p3_clustering.py``), que ``train.py`` descubre e invoca::

    PREGUNTA = "P1"

    def entrenar(roles: tuple[str, ...] = ROLES) -> list[str]:
        '''Entrena los modelos de ``roles`` y devuelve los run_id de MLflow.'''
        datos = cargar_dataset(config.DATASET_P1, columna_tiempo="hora")
        with iniciar_run(PREGUNTA, "baseline", "rezago_semanal") as run:
            predicho = datos["validacion"]["salidas_lag_168h"]
            registrar_metricas({"mae": ...}, "validacion")
        return [run.info.run_id]
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import subprocess

import joblib
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import (
    PARTICIONES,
    asignar_particion,
    cargar_metadatos_features,
)
from tc5061_proyecto_equipo18.plots import guardar_figura

PREGUNTAS = ("P1", "P2", "P3")
ROLES = ("baseline", "candidato")


# =============================================================================
# Configuracion de MLflow
# =============================================================================
def configurar_mlflow(
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experimento: str = config.MLFLOW_EXPERIMENTO,
    artefactos_dir: Path = config.MLFLOW_ARTEFACTOS_DIR,
) -> str:
    """Apunta MLflow a la base del proyecto y activa el experimento del equipo.

    Si el experimento no existe se crea con sus artefactos en ``mlruns/`` (ignorado por Git),
    de modo que cualquier clon del repo produce la misma estructura. Devuelve su id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    existente = mlflow.get_experiment_by_name(experimento)
    if existente is None:
        artefactos_dir.mkdir(parents=True, exist_ok=True)
        experimento_id = mlflow.create_experiment(
            experimento,
            artifact_location=artefactos_dir.resolve().as_uri(),
            tags={"equipo": config.EQUIPO},
        )
    else:
        experimento_id = existente.experiment_id
    mlflow.set_experiment(experiment_id=experimento_id)
    return experimento_id


def estado_git() -> dict[str, str]:
    """Commit actual y si hay cambios sin confirmar, para saber que codigo produjo cada run."""
    try:
        commit = _git("rev-parse", "HEAD")
        sucio = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "desconocido", "git_sucio": "desconocido"}
    return {"git_commit": commit, "git_sucio": str(sucio).lower()}


def _git(*argumentos: str) -> str:
    resultado = subprocess.run(
        ["git", *argumentos], cwd=config.PROJ_ROOT, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


# =============================================================================
# Datos
# =============================================================================
def separar_particiones(df: pd.DataFrame, columna_tiempo: str | None) -> dict[str, pd.DataFrame]:
    """Separa un dataset en entrenamiento, validacion y prueba con el corte temporal de config.

    Con ``columna_tiempo=None`` se usa la columna ``periodo`` que ya trae el dataset (P3, cuyos
    perfiles se calculan por separado en cada periodo).
    """
    if columna_tiempo is None:
        etiquetas = df["periodo"]
    else:
        etiquetas = asignar_particion(df[columna_tiempo])
    return {
        particion: df[etiquetas == particion].reset_index(drop=True) for particion in PARTICIONES
    }


def cargar_dataset(ruta: Path, columna_tiempo: str | None) -> dict[str, pd.DataFrame]:
    """Lee el Parquet de una pregunta (lo escribe features.py) y lo separa por particion."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta: python -m tc5061_proyecto_equipo18.features"
        )
    return separar_particiones(pd.read_parquet(ruta), columna_tiempo)


def muestrear(df: pd.DataFrame, max_filas: int, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Muestra aleatoria reproducible de a lo mas ``max_filas`` filas (acota la memoria)."""
    if len(df) <= max_filas:
        return df
    return df.sample(n=max_filas, random_state=semilla).reset_index(drop=True)


# =============================================================================
# Runs, metricas y artefactos
# =============================================================================
def nombre_run(pregunta: str, rol: str, modelo: str) -> str:
    """Nombre legible y unico por combinacion, por ejemplo ``p1-candidato-hgb_poisson``."""
    return f"{pregunta.lower()}-{rol}-{modelo}"


@contextmanager
def iniciar_run(
    pregunta: str,
    rol: str,
    modelo: str,
    parametros: dict | None = None,
    meta_features: dict | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Abre un run de MLflow con los tags y parametros comunes a todo el experimento.

    Registra la version de los datos (``data_version``), que se valida contra el manifiesto al
    leer los metadatos de features: si los datos o config.py cambiaron, falla antes de entrenar.
    ``data_version`` y ``modelo`` quedan como tags y tambien como parametros, junto a
    ``parametros`` (los hiperparametros propios del modelo).
    """
    if pregunta not in PREGUNTAS:
        raise ValueError(f"Pregunta desconocida: {pregunta}; opciones {PREGUNTAS}")
    if rol not in ROLES:
        raise ValueError(f"Rol desconocido: {rol}; opciones {ROLES}")
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    tags = {
        "equipo": config.EQUIPO,
        "pregunta": pregunta,
        "rol": rol,
        "modelo": modelo,
        "data_version": meta["data_version"],
        **estado_git(),
    }
    with mlflow.start_run(run_name=nombre_run(pregunta, rol, modelo), tags=tags) as run:
        mlflow.log_params(
            {
                "semilla": config.SEMILLA,
                **{f"features.{clave}": valor for clave, valor in meta["parametros"].items()},
                **(parametros or {}),
                # Tambien como parametros (no solo tags) para compararlos junto a los
                # hiperparametros; van al final para que ``parametros`` no los sobrescriba.
                "data_version": meta["data_version"],
                "modelo": modelo,
            }
        )
        mlflow.log_dict(meta, "datos/features_meta.json")
        if config.MANIFIESTO_DATOS.exists():
            mlflow.log_artifact(str(config.MANIFIESTO_DATOS), "datos")
        yield run


def registrar_metricas(metricas: dict[str, float], particion: str) -> None:
    """Registra metricas con el prefijo de la particion: ``{"mae": 1.2}`` -> ``validacion_mae``."""
    if particion not in PARTICIONES:
        raise ValueError(f"Particion desconocida: {particion}; opciones {PARTICIONES}")
    mlflow.log_metrics(
        {f"{particion}_{nombre}": float(valor) for nombre, valor in metricas.items()}
    )


def _run_activo() -> mlflow.ActiveRun:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("Llama a esta funcion dentro de `with iniciar_run(...)`")
    return run


def registrar_figura(figura: Figure, nombre: str) -> Path:
    """Guarda la figura en reports/figures/experimento_base/ y la sube al run activo."""
    run = _run_activo()
    ruta = guardar_figura(
        figura, config.FIGURAS_EXPERIMENTO_DIR / f"{run.info.run_name}_{nombre}.png"
    )
    mlflow.log_artifact(str(ruta), "figuras")
    return ruta


def registrar_modelo(modelo, ejemplo: pd.DataFrame) -> str:
    """Registra un estimador de scikit-learn en el run activo y guarda una copia local.

    MLflow lo guarda en formato skops (no ejecuta codigo arbitrario al cargarlo) con la firma
    de entrada y salida inferida de ``ejemplo``. La copia en models/ (ignorada por Git) es la
    que usa predict.py sin depender del servidor de MLflow. Devuelve la URI del modelo.
    """
    run = _run_activo()
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    info = mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
        skops_trusted_types=config.SKOPS_TIPOS_CONFIABLES,
        signature=firma,
        input_example=ejemplo.head(5),
    )
    ruta_local = config.ruta_modelo(run.info.run_name)
    ruta_local.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_local)
    return info.model_uri
