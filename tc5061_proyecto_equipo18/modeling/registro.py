"""Model Registry: registra el modelo elegido de cada pregunta y lo lleva a produccion con aliases.

Punto de entrada ``register`` del MLflow Project, desde la raiz del repositorio::

    mlflow run . -e register -P pregunta=p1 --env-manager local
    mlflow run . -e register -P pregunta=p2 -P run_id=<run_id elegido> --env-manager local
    mlflow run . -e register -P pregunta=p3 -P challenger_run_id=ninguno --env-manager local

Esquema de ciclo de vida: **aliases** (el esquema vigente de MLflow; las etapas Staging y
Production estan obsoletas desde MLflow 2.9). Cada pregunta tiene un modelo registrado
(``config.MODELOS_REGISTRADOS``) y en el:

- ``@champion`` (``config.ALIAS_PRODUCCION``) apunta a la version en produccion;
- ``@challenger`` (``config.ALIAS_RETADOR``) apunta a la siguiente mejor, para compararla en
  paralelo antes de promoverla.

Quien consume el modelo siempre lo carga como ``models:/<nombre>@champion``: promover otra
version es mover el alias, sin cambiar codigo ni URIs.

Seleccion. Se consideran los runs con modelo de la pregunta (etapa ``candidato`` o
``busqueda``) lanzados por el MLflow Project, desde un arbol de trabajo limpio y con la misma
``data_version`` que el baseline. Se ordenan con ``config.METRICAS_SELECCION``. Por defecto
el champion es el primero y el challenger el siguiente con otra configuracion (otro modelo o
hiperparametros); ``-P run_id=...`` fija el champion cuando el equipo elige otro por
estabilidad o costo. Un champion que no mejora al baseline se rechaza.

Linaje: cada version guarda en sus etiquetas y descripcion el experimento, el run de origen,
el commit, la version de datos, el modelo, sus metricas y el comando para reproducirlo.
"""

from dataclasses import dataclass
import os

from loguru import logger
import mlflow
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import pandas as pd
import typer

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling.comun import (
    configurar_mlflow,
    especificacion,
    normalizar_pregunta,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

ETAPAS_REGISTRABLES = ("candidato", "busqueda")
AUTO = "auto"  # valor por defecto de -P run_id y -P challenger_run_id en el MLproject
NINGUNO = "ninguno"  # -P challenger_run_id=ninguno: no asignar challenger


@dataclass
class Seleccion:
    """Resultado de comparar los runs de una pregunta contra su baseline."""

    pregunta: str
    metrica: str
    sentido: str
    baseline: Run
    ranking: pd.DataFrame  # una fila por run registrable, del mejor al peor


# =============================================================================
# Seleccion de runs
# =============================================================================
def _filtro_runs(pregunta: str, etapa: str, solo_oficiales: bool) -> str:
    filtro = (
        f"tags.pregunta = '{pregunta}' AND tags.etapa = '{etapa}' "
        "AND attributes.status = 'FINISHED'"
    )
    if solo_oficiales:
        filtro += " AND tags.`mlflow.source.type` = 'PROJECT' AND tags.git_sucio = 'false'"
    return filtro


def _buscar_runs(experimento_id: str, filtro: str, metrica: str, sentido: str) -> list[Run]:
    orden = "ASC" if sentido == "min" else "DESC"
    return list(
        mlflow.MlflowClient().search_runs(
            [experimento_id], filtro, order_by=[f"metrics.{metrica} {orden}"], max_results=5000
        )
    )


def model_id_de(run: Run) -> str | None:
    """Id del modelo que guardo el run con ``log_model`` (los baselines no guardan modelo)."""
    salidas = run.outputs.model_outputs if run.outputs else []
    return salidas[0].model_id if salidas else None


def mejora_relativa(valor: float, baseline: float, sentido: str) -> float:
    """Mejora respecto al baseline como fraccion de su valor (positiva = mejor que el baseline)."""
    if baseline == 0:
        return float("nan")
    diferencia = baseline - valor if sentido == "min" else valor - baseline
    return diferencia / abs(baseline)


def configuracion_de(pregunta: str, run: Run) -> tuple:
    """Modelo e hiperparametros de un run: dos runs con la misma son el mismo modelo."""
    modelo = run.data.tags.get("modelo", "")
    try:
        claves = especificacion(pregunta, modelo)["hiperparametros"]
    except ValueError:
        claves = {}
    return (modelo, *(run.data.params.get(clave) for clave in sorted(claves)))


def seleccionar(
    pregunta: str, experimento_id: str | None = None, solo_oficiales: bool = True
) -> Seleccion:
    """Compara los runs registrables de una pregunta contra el baseline mas reciente.

    ``solo_oficiales=True`` (lo normal) exige runs lanzados por el MLflow Project desde un arbol
    de trabajo sin cambios, que son los que se pueden reproducir con su commit.
    """
    pregunta = normalizar_pregunta(pregunta)
    metrica, sentido = config.METRICAS_SELECCION[pregunta]
    if experimento_id is None:
        experimento_id = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENTO).experiment_id

    baselines = list(
        mlflow.MlflowClient().search_runs(
            [experimento_id],
            _filtro_runs(pregunta, "baseline", solo_oficiales),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    )
    if not baselines:
        raise RuntimeError(
            f"No hay un baseline de {pregunta} para comparar. Corre: mlflow run . -e train "
            f"-P pregunta={pregunta.lower()} -P modelo=<baseline> --env-manager local"
        )
    baseline = baselines[0]
    data_version = baseline.data.tags.get("data_version")
    if metrica not in baseline.data.metrics:
        raise RuntimeError(f"El baseline {baseline.info.run_id} no tiene la metrica {metrica}")
    valor_baseline = baseline.data.metrics[metrica]

    filas = []
    for etapa in ETAPAS_REGISTRABLES:
        filtro = _filtro_runs(pregunta, etapa, solo_oficiales)
        filtro += f" AND tags.data_version = '{data_version}'"
        for run in _buscar_runs(experimento_id, filtro, metrica, sentido):
            model_id = model_id_de(run)
            if model_id is None or metrica not in run.data.metrics:
                continue
            valor = run.data.metrics[metrica]
            filas.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "etapa": etapa,
                    "modelo": run.data.tags.get("modelo"),
                    "model_id": model_id,
                    metrica: valor,
                    "mejora_vs_baseline": mejora_relativa(valor, valor_baseline, sentido),
                    "configuracion": configuracion_de(pregunta, run),
                }
            )
    ranking = pd.DataFrame(filas)
    if not ranking.empty:
        ranking = ranking.sort_values(metrica, ascending=sentido == "min", ignore_index=True)
    return Seleccion(pregunta, metrica, sentido, baseline, ranking)


def elegir(
    seleccion: Seleccion, run_id: str = AUTO, challenger_run_id: str = AUTO
) -> tuple[pd.Series, pd.Series | None]:
    """Champion y challenger (o None) a partir del ranking.

    El champion es el ``run_id`` pedido o, con ``auto``, el mejor del ranking; debe mejorar al
    baseline. El challenger es el pedido, ninguno, o el mejor siguiente con otra configuracion
    (una hija de la busqueda puede repetir la configuracion del candidato).
    """
    ranking = seleccion.ranking
    if ranking.empty:
        raise RuntimeError(
            f"No hay runs con modelo de {seleccion.pregunta} con la misma data_version que el "
            "baseline. Entrena los candidatos con `mlflow run . -e train` desde main."
        )

    def fila(identificador: str) -> pd.Series:
        coincide = ranking[ranking["run_id"] == identificador]
        if coincide.empty:
            raise RuntimeError(
                f"El run {identificador} no es registrable para {seleccion.pregunta}: debe ser "
                "un candidato o una hija de busqueda con modelo, lanzado con `mlflow run` desde "
                "un arbol limpio y con la misma data_version que el baseline."
            )
        return coincide.iloc[0]

    champion = ranking.iloc[0] if run_id == AUTO else fila(run_id)
    if not champion["mejora_vs_baseline"] > 0:
        raise RuntimeError(
            f"{champion['run_name']} no mejora al baseline en {seleccion.metrica} "
            f"({champion[seleccion.metrica]:.4f} contra "
            f"{seleccion.baseline.data.metrics[seleccion.metrica]:.4f}): no se registra."
        )

    if challenger_run_id == NINGUNO:
        return champion, None
    if challenger_run_id != AUTO:
        retador = fila(challenger_run_id)
        if retador["run_id"] == champion["run_id"]:
            raise RuntimeError("El challenger debe ser un run distinto del champion")
        return champion, retador
    otros = ranking[ranking["configuracion"].map(lambda c: c != champion["configuracion"])]
    return champion, (otros.iloc[0] if not otros.empty else None)


# =============================================================================
# Registro y aliases
# =============================================================================
def asegurar_modelo_registrado(pregunta: str) -> str:
    """Crea el modelo registrado de la pregunta si no existe y actualiza su descripcion."""
    nombre = config.MODELOS_REGISTRADOS[pregunta]
    cliente = mlflow.MlflowClient()
    metrica, sentido = config.METRICAS_SELECCION[pregunta]
    etiquetas = {
        "equipo": config.EQUIPO,
        "pregunta": pregunta,
        "metrica_seleccion": f"{metrica} ({sentido})",
        "ciclo_de_vida": f"aliases: {config.ALIAS_PRODUCCION} y {config.ALIAS_RETADOR}",
    }
    try:
        cliente.get_registered_model(nombre)
    except MlflowException:
        cliente.create_registered_model(nombre, tags=etiquetas)
    cliente.update_registered_model(nombre, description=config.DESCRIPCIONES_REGISTRO[pregunta])
    for clave, valor in etiquetas.items():
        cliente.set_registered_model_tag(nombre, clave, valor)
    return nombre


def etiquetas_linaje(seleccion: Seleccion, fila: pd.Series) -> dict[str, str]:
    """Etiquetas de una version: de donde salio el modelo y como se compara con el baseline."""
    run = mlflow.get_run(fila["run_id"])
    tags = run.data.tags
    experimento = mlflow.get_experiment(run.info.experiment_id)
    baseline = seleccion.baseline
    return {
        "experimento": experimento.name,
        "experimento_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "git_commit": tags.get("git_commit", "desconocido"),
        "data_version": tags.get("data_version", "desconocido"),
        "pregunta": seleccion.pregunta,
        "etapa": tags.get("etapa", ""),
        "modelo": tags.get("modelo", ""),
        "familia_modelo": tags.get("familia_modelo", ""),
        seleccion.metrica: f"{fila[seleccion.metrica]:.6g}",
        "baseline_run_id": baseline.info.run_id,
        f"baseline_{seleccion.metrica}": f"{baseline.data.metrics[seleccion.metrica]:.6g}",
        "mejora_vs_baseline": f"{fila['mejora_vs_baseline']:.4f}",
        "comando_reproduccion": tags.get("mlflow.note.content", "").removeprefix("Reproducir: "),
    }


def descripcion_version(etiquetas: dict[str, str], metrica: str) -> str:
    return (
        f"{etiquetas['modelo']} ({etiquetas['familia_modelo']}) del run {etiquetas['run_name']} "
        f"({etiquetas['run_id']}), experimento {etiquetas['experimento']}, commit "
        f"{etiquetas['git_commit'][:12]}, datos {etiquetas['data_version']}. "
        f"{metrica} = {etiquetas[metrica]} contra {etiquetas[f'baseline_{metrica}']} del baseline "
        f"(mejora {float(etiquetas['mejora_vs_baseline']):.1%}). "
        f"Reproducir: {etiquetas['comando_reproduccion']}"
    )


def registrar_version(seleccion: Seleccion, fila: pd.Series) -> ModelVersion:
    """Registra el modelo de un run como version (o reutiliza la que ya existe) con su linaje."""
    nombre = asegurar_modelo_registrado(seleccion.pregunta)
    cliente = mlflow.MlflowClient()
    existentes = cliente.search_model_versions(
        f"name = '{nombre}' AND run_id = '{fila['run_id']}'"
    )
    etiquetas = etiquetas_linaje(seleccion, fila)
    if existentes:
        version = existentes[0]
        logger.info(f"{nombre}: el run {fila['run_name']} ya es la version {version.version}")
    else:
        version = mlflow.register_model(f"models:/{fila['model_id']}", nombre, tags=etiquetas)
        logger.info(f"{nombre}: version {version.version} creada desde {fila['run_name']}")
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(nombre, version.version, clave, valor)
    cliente.update_model_version(
        nombre, version.version, description=descripcion_version(etiquetas, seleccion.metrica)
    )
    return cliente.get_model_version(nombre, version.version)


def asignar_alias(nombre: str, alias: str, version: ModelVersion | None) -> None:
    """Apunta ``alias`` a la version; con None lo quita (por ejemplo, si ya no hay challenger)."""
    cliente = mlflow.MlflowClient()
    if version is not None:
        cliente.set_registered_model_alias(nombre, alias, version.version)
        return
    try:
        cliente.delete_registered_model_alias(nombre, alias)
    except MlflowException:
        pass  # el alias no existia


def verificar_carga(nombre: str, alias: str = config.ALIAS_PRODUCCION) -> int:
    """Carga ``models:/<nombre>@<alias>`` y predice su ejemplo de entrada; da cuantas filas."""
    uri = f"models:/{nombre}@{alias}"
    modelo = mlflow.pyfunc.load_model(uri)
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise RuntimeError(f"{uri} no tiene ejemplo de entrada para verificar la carga")
    predicho = modelo.predict(ejemplo)
    logger.success(f"{uri} se carga y predice {len(predicho)} filas de su ejemplo de entrada")
    return len(predicho)


def registrar(
    pregunta: str,
    run_id: str = AUTO,
    challenger_run_id: str = AUTO,
    experimento_id: str | None = None,
    solo_oficiales: bool = True,
) -> dict[str, str]:
    """Selecciona, registra y asigna aliases para una pregunta; devuelve un resumen."""
    seleccion = seleccionar(pregunta, experimento_id, solo_oficiales)
    champion, retador = elegir(seleccion, run_id, challenger_run_id)
    nombre = config.MODELOS_REGISTRADOS[seleccion.pregunta]

    version_champion = registrar_version(seleccion, champion)
    asignar_alias(nombre, config.ALIAS_PRODUCCION, version_champion)
    version_retador = registrar_version(seleccion, retador) if retador is not None else None
    asignar_alias(nombre, config.ALIAS_RETADOR, version_retador)
    filas_verificadas = verificar_carga(nombre)

    resumen = {
        "modelo_registrado": nombre,
        "metrica_seleccion": seleccion.metrica,
        "champion_version": version_champion.version,
        "champion_run_id": champion["run_id"],
        "champion_run_name": champion["run_name"],
        "champion_mejora_vs_baseline": f"{champion['mejora_vs_baseline']:.4f}",
        "challenger_version": version_retador.version if version_retador else NINGUNO,
        "challenger_run_name": retador["run_name"] if retador is not None else NINGUNO,
        "verificacion": f"models:/{nombre}@{config.ALIAS_PRODUCCION} ({filas_verificadas} filas)",
    }
    logger.info("Ranking:\n" + seleccion.ranking.drop(columns="configuracion").to_string())
    return resumen


def registrar_en_run_del_proyecto(pregunta: str, resumen: dict[str, str]) -> None:
    """Deja el resumen en el run que crea ``mlflow run . -e register`` (etapa ``registro``)."""
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        return
    cliente = mlflow.MlflowClient()
    etiquetas = {
        "mlflow.runName": f"{pregunta.lower()}-registro-{config.ALIAS_PRODUCCION}",
        "equipo": config.EQUIPO,
        "pregunta": pregunta,
        "etapa": "registro",
        **resumen,
    }
    for clave, valor in etiquetas.items():
        cliente.set_tag(run_id, clave, valor)
    cliente.log_dict(run_id, resumen, "registro/resumen.json")


@app.command()
def main(
    pregunta: str = typer.Option(..., "--pregunta", "-p", help="p1, p2 o p3."),
    run_id: str = typer.Option(
        AUTO, "--run_id", "--run-id", help="Run del champion; auto = el mejor."
    ),
    challenger_run_id: str = typer.Option(
        AUTO,
        "--challenger_run_id",
        "--challenger-run-id",
        help="Run del challenger; auto = el siguiente con otra configuracion; o ninguno.",
    ),
):
    """Registra el modelo elegido de una pregunta y le asigna los aliases de produccion.

    ``mlflow run`` pasa los ``-P`` que no declara el MLproject como ``--nombre valor``; por eso
    ``run_id`` y ``challenger_run_id`` aceptan tambien la forma con guion bajo.
    """
    pregunta = normalizar_pregunta(pregunta)
    experimento_id = configurar_mlflow()
    resumen = registrar(pregunta, run_id, challenger_run_id, experimento_id)
    registrar_en_run_del_proyecto(pregunta, resumen)
    for clave, valor in resumen.items():
        logger.info(f"{clave}: {valor}")


if __name__ == "__main__":
    app()
