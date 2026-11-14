"""Continuous Training de P1: reentrena el champion con datos recientes y lo registra como challenger.

Punto de entrada ``retrain`` del MLflow Project, desde la raiz del repositorio::

    mlflow run . -e retrain --env-manager local      # o: make retrain

Que hace:

1. Busca la version ``@champion`` de P1 en el Model Registry y el run que la entreno: de ahi
   salen el modelo (``hgb_poisson``, por ejemplo) y sus hiperparametros exactos.
2. Entrena ese mismo modelo, con los mismos hiperparametros y el mismo tope de filas
   (``config.P1_MAX_FILAS_ENTRENAMIENTO``), con todos los datos desde ``config.MES_INICIO``
   hasta ``config.REENTRENAMIENTO_FIN`` (incluye jul-ago 2026, del periodo actual). Asi la
   unica diferencia con el champion son los datos, y el A/B mide solo el efecto de entrenar
   con datos recientes.
3. Lo registra en un run de MLflow (etiqueta ``etapa = reentrenamiento``) y como **nueva
   version del mismo modelo registrado**, con su linaje: version y run del champion del que
   sale, periodo de entrenamiento, commit y version de datos.
4. Le asigna el alias ``@challenger`` (``config.ALIAS_RETADOR``). No toca ``@champion``:
   promoverlo es decision del A/B testing (``mlflow run . -e abtest``).

Lo que NO hace: evaluar. El modelo reentrenado ya vio la validacion (abr-jun) y parte de la
prueba (jul-ago), asi que sus metricas en esas particiones no son comparables con las del
champion. La comparacion justa es la del A/B, sobre ``config.AB_EVALUACION`` (sep 2026), un mes
que ninguno de los dos modelos vio. Por eso este run no lleva la etiqueta ``candidato`` y la
seleccion del ``register`` lo ignora.
"""

from loguru import logger
import mlflow
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import pandas as pd
import typer

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1, cargar_metadatos_features
from tc5061_proyecto_equipo18.modeling import p1_regresion, registro
from tc5061_proyecto_equipo18.modeling.comun import (
    configurar_mlflow,
    cronometro,
    es_baseline,
    especificacion,
    huella_conjunto,
    iniciar_run,
    muestrear,
    registrar_convergencia,
    registrar_costos,
    registrar_modelo,
    resolver_hiperparametros,
    vigilar_convergencia,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

PREGUNTA = "P1"
ETAPA = "reentrenamiento"
COLUMNA_TIEMPO = "hora"


# =============================================================================
# Champion de origen
# =============================================================================
def champion_actual(
    nombre: str = config.MODELOS_REGISTRADOS[PREGUNTA],
) -> tuple[ModelVersion, Run]:
    """Version ``@champion`` de P1 y el run que la entreno."""
    cliente = mlflow.MlflowClient()
    try:
        version = cliente.get_model_version_by_alias(nombre, config.ALIAS_PRODUCCION)
    except MlflowException as error:
        raise RuntimeError(
            f"{nombre} no tiene version @{config.ALIAS_PRODUCCION}. Registra primero el "
            "champion: mlflow run . -e register -P pregunta=p1 --env-manager local"
        ) from error
    return version, mlflow.get_run(version.run_id)


def configuracion_del_champion(run: Run) -> tuple[str, dict]:
    """Modelo e hiperparametros con que se entreno el champion (de su run, no de config.py).

    Se leen del run porque el champion pudo salir de una busqueda o de un ``-P`` que cambio
    algun hiperparametro; los valores (texto en MLflow) se convierten al tipo de config.py.
    """
    modelo = run.data.tags.get("modelo", "")
    if modelo not in config.P1_MODELOS:
        raise RuntimeError(f"El run {run.info.run_id} no es de un modelo de P1: {modelo!r}")
    if es_baseline(PREGUNTA, modelo):
        raise RuntimeError(f"El champion es el baseline {modelo!r}: no hay modelo que reentrenar")
    claves = especificacion(PREGUNTA, modelo)["hiperparametros"]
    cambios = {clave: run.data.params[clave] for clave in claves if clave in run.data.params}
    return modelo, resolver_hiperparametros(PREGUNTA, modelo, cambios)


# =============================================================================
# Datos
# =============================================================================
def fin_del_dia(fecha: str) -> pd.Timestamp:
    """``"2026-08-31"`` -> ``2026-08-31 23:00``, la ultima hora del dia (fechas inclusivas)."""
    return pd.Timestamp(fecha) + pd.Timedelta(hours=23)


def verificar_periodos(
    fin: str = config.REENTRENAMIENTO_FIN, evaluacion: tuple[str, str] = config.AB_EVALUACION
) -> None:
    """El reentrenamiento debe terminar antes del mes del A/B: si no, B se evaluaria con datos
    que ya vio y la comparacion favoreceria a B sin razon."""
    if fin_del_dia(fin) >= pd.Timestamp(evaluacion[0]):
        raise ValueError(
            f"REENTRENAMIENTO_FIN ({fin}) debe ser anterior a AB_EVALUACION ({evaluacion[0]}): "
            "el challenger no puede entrenar con el mes con que se evalua."
        )
    if fin_del_dia(fin) <= fin_del_dia(config.FIN_ENTRENAMIENTO):
        raise ValueError(
            f"REENTRENAMIENTO_FIN ({fin}) no agrega datos: el champion ya entreno hasta "
            f"{config.FIN_ENTRENAMIENTO}."
        )


def datos_reentrenamiento(
    p1: pd.DataFrame,
    fin: str = config.REENTRENAMIENTO_FIN,
    evaluacion: tuple[str, str] = config.AB_EVALUACION,
) -> pd.DataFrame:
    """Filas de P1 desde el inicio de los datos hasta el final del dia ``fin``."""
    verificar_periodos(fin, evaluacion)
    tiempo = pd.to_datetime(p1[COLUMNA_TIEMPO])
    # Si los datos no pasan de ``fin``, faltan meses recientes: el reentrenado no tendria los
    # datos que se supone que agrega y el A/B no tendria con que evaluar
    if tiempo.max() <= fin_del_dia(fin):
        raise ValueError(
            f"Los datos de P1 terminan en {tiempo.max():%Y-%m-%d}, sin pasar de "
            f"REENTRENAMIENTO_FIN ({fin}). Corre make data && make features con los meses "
            "recientes."
        )
    return p1[tiempo <= fin_del_dia(fin)].reset_index(drop=True)


def cargar_p1() -> pd.DataFrame:
    """Parquet completo de P1 (lo escribe features.py)."""
    if not config.DATASET_P1.exists():
        raise FileNotFoundError(
            f"No existe {config.DATASET_P1}. Ejecuta: mlflow run . -e features --env-manager local"
        )
    return pd.read_parquet(config.DATASET_P1)


def registrar_conjunto(tabla: pd.DataFrame, data_version: str) -> None:
    """Registra el conjunto de reentrenamiento con ``mlflow.log_input`` (esquema y huella)."""
    fuente = config.DATASET_P1.relative_to(config.PROJ_ROOT).as_posix()
    conjunto = mlflow.data.from_pandas(
        tabla[[*COLUMNAS_P1, OBJETIVO_P1]],
        source=fuente,
        name=f"p1-{ETAPA}-{data_version}",
        targets=OBJETIVO_P1,
        digest=huella_conjunto(tabla[[*COLUMNAS_P1, OBJETIVO_P1]], data_version),
    )
    mlflow.log_input(conjunto, context=ETAPA)


# =============================================================================
# Entrenamiento y registro
# =============================================================================
def entrenar(
    p1: pd.DataFrame,
    modelo: str,
    hiperparametros: dict,
    origen: ModelVersion,
    meta_features: dict | None = None,
    fin: str = config.REENTRENAMIENTO_FIN,
) -> Run:
    """Entrena el modelo del champion con los datos hasta ``fin`` y lo guarda en un run."""
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    datos = datos_reentrenamiento(p1, fin)
    tiempo = pd.to_datetime(datos[COLUMNA_TIEMPO])
    parametros = {
        "reentrenamiento_inicio": f"{tiempo.min():%Y-%m-%d}",
        "reentrenamiento_fin": fin,
        "filas_reentrenamiento": len(datos),
        "estaciones_reentrenamiento": int(datos["estacion_id"].nunique()),
        "max_filas_entrenamiento": config.P1_MAX_FILAS_ENTRENAMIENTO,
        "version_origen": str(origen.version),
        "run_origen": origen.run_id,
    }
    descripcion = (
        f"Continuous Training: el modelo de la version {origen.version} (@champion) con los "
        f"mismos hiperparametros, reentrenado con datos de {parametros['reentrenamiento_inicio']} "
        f"a {fin}. Reproducir: mlflow run . -e retrain --env-manager local"
    )
    with iniciar_run(
        PREGUNTA,
        modelo,
        ETAPA,
        hiperparametros,
        meta,
        nombre=f"p1-{ETAPA}-{modelo}-hasta-{fin}",
        parametros=parametros,
        descripcion=descripcion,
    ) as run:
        mlflow.set_tags({"version_origen": str(origen.version), "run_origen": origen.run_id})
        registrar_conjunto(datos, meta["data_version"])
        muestra = muestrear(datos, config.P1_MAX_FILAS_ENTRENAMIENTO)
        estimador = p1_regresion.construir_estimador(modelo, hiperparametros)
        with cronometro() as tiempo_ajuste, vigilar_convergencia() as avisos:
            estimador.fit(muestra[COLUMNAS_P1], muestra[OBJETIVO_P1])
        registrar_costos(tiempo_ajuste["segundos"])
        registrar_convergencia(estimador, avisos)
        registrar_modelo(estimador, datos[COLUMNAS_P1].head(100))
        run_id = run.info.run_id
    return mlflow.get_run(run_id)


def etiquetas_linaje(run: Run, origen: ModelVersion) -> dict[str, str]:
    """Etiquetas de la nueva version: de que champion sale y con que datos se reentreno."""
    tags, params = run.data.tags, run.data.params
    return {
        "experimento": mlflow.get_experiment(run.info.experiment_id).name,
        "experimento_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "git_commit": tags.get("git_commit", "desconocido"),
        "data_version": tags.get("data_version", "desconocido"),
        "pregunta": PREGUNTA,
        "etapa": ETAPA,
        "modelo": tags.get("modelo", ""),
        "familia_modelo": tags.get("familia_modelo", ""),
        "version_origen": str(origen.version),
        "run_origen": origen.run_id,
        "reentrenamiento_inicio": params.get("reentrenamiento_inicio", ""),
        "reentrenamiento_fin": params.get("reentrenamiento_fin", ""),
        "comando_reproduccion": "mlflow run . -e retrain --env-manager local",
    }


def registrar_challenger(run: Run, origen: ModelVersion) -> ModelVersion:
    """Registra el modelo del run como nueva version de P1 y le asigna ``@challenger``."""
    model_id = registro.model_id_de(run)
    if model_id is None:
        raise RuntimeError(f"El run {run.info.run_id} no guardo un modelo")
    nombre = registro.asegurar_modelo_registrado(PREGUNTA)
    etiquetas = etiquetas_linaje(run, origen)
    version = mlflow.register_model(f"models:/{model_id}", nombre, tags=etiquetas)
    cliente = mlflow.MlflowClient()
    cliente.update_model_version(
        nombre,
        version.version,
        description=(
            f"Challenger por Continuous Training: version {origen.version} (@champion) "
            f"reentrenada con datos de {etiquetas['reentrenamiento_inicio']} a "
            f"{etiquetas['reentrenamiento_fin']}, mismos hiperparametros. Run "
            f"{run.info.run_name} ({run.info.run_id}), commit {etiquetas['git_commit']}, datos "
            f"{etiquetas['data_version']}. Se evalua contra el champion con el A/B testing "
            f"(mlflow run . -e abtest)."
        ),
    )
    registro.asignar_alias(nombre, config.ALIAS_RETADOR, version)
    return cliente.get_model_version(nombre, version.version)


def reentrenar(
    p1: pd.DataFrame | None = None,
    meta_features: dict | None = None,
    fin: str = config.REENTRENAMIENTO_FIN,
) -> ModelVersion:
    """Flujo completo: champion -> reentrenamiento -> nueva version con ``@challenger``."""
    verificar_periodos(fin)  # antes de leer datos o entrenar: un error de config falla rapido
    origen, run_origen = champion_actual()
    modelo, hiperparametros = configuracion_del_champion(run_origen)
    logger.info(
        f"Champion: version {origen.version} ({modelo}, run {run_origen.info.run_name}); "
        f"se reentrena con datos hasta {fin}"
    )
    datos_meta = meta_features if meta_features is not None else cargar_metadatos_features()
    if run_origen.data.tags.get("data_version") != datos_meta["data_version"]:
        logger.warning(
            f"El champion se entreno con data_version {run_origen.data.tags.get('data_version')} "
            f"y los datos actuales son {datos_meta['data_version']}: el A/B comparara modelos "
            "entrenados con datos procesados distinto."
        )
    run = entrenar(
        p1 if p1 is not None else cargar_p1(), modelo, hiperparametros, origen, datos_meta, fin
    )
    version = registrar_challenger(run, origen)
    filas = registro.verificar_carga(version.name, config.ALIAS_RETADOR)
    logger.success(
        f"{version.name} version {version.version} = @{config.ALIAS_RETADOR} "
        f"(sale de la version {origen.version}; carga y predice {filas} filas)"
    )
    return version


@app.command()
def main():
    """Reentrena el champion de P1 con datos hasta config.REENTRENAMIENTO_FIN (challenger)."""
    verificar_periodos()
    configurar_mlflow()
    reentrenar()


if __name__ == "__main__":
    app()
