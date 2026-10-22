"""Registro en MLflow de las etapas de datos (puntos de entrada ``ingest`` y ``features``).

``mlflow run . -e ingest`` crea un run antes de ejecutar ``dataset.py`` y se lo indica con la
variable de entorno ``MLFLOW_RUN_ID``. Esta funcion deja en ese run la version de los datos y
sus metadatos (etiqueta ``etapa = datos``), de modo que la preparacion de datos tambien queda
trazable junto a los entrenamientos. Si el comando se corre fuera de ``mlflow run``, no hace
nada: la ingesta y las variables funcionan igual sin MLflow.
"""

import os
from pathlib import Path

from tc5061_proyecto_equipo18 import config


def lanzado_por_mlflow_run() -> bool:
    """True si el proceso lo inicio ``mlflow run`` (MLflow define ``MLFLOW_RUN_ID``)."""
    return bool(os.getenv("MLFLOW_RUN_ID"))


def registrar_etapa_datos(
    nombre: str,
    data_version: str,
    parametros: dict | None = None,
    metricas: dict[str, float] | None = None,
    archivos: list[Path] | None = None,
    diccionarios: dict[str, dict] | None = None,
) -> bool:
    """Registra una etapa de datos en el run de ``mlflow run``; devuelve si registro algo.

    ``archivos`` y ``diccionarios`` van a la carpeta de artefactos ``datos/`` (por ejemplo, el
    manifiesto de datos y los metadatos de features). Solo se registran agregados y metadatos,
    nunca viajes ni datos personales.
    """
    if not lanzado_por_mlflow_run():
        return False
    import mlflow  # solo se necesita dentro de mlflow run

    tags = {"equipo": config.EQUIPO, "etapa": "datos", "data_version": data_version}
    with mlflow.start_run(run_name=nombre, tags=tags) as run:
        existentes = mlflow.get_run(run.info.run_id).data.params
        nuevos = {
            clave: valor for clave, valor in (parametros or {}).items() if clave not in existentes
        }
        if nuevos:
            mlflow.log_params(nuevos)
        if metricas:
            mlflow.log_metrics({clave: float(valor) for clave, valor in metricas.items()})
        for ruta in archivos or []:
            if Path(ruta).exists():
                mlflow.log_artifact(str(ruta), "datos")
        for nombre_archivo, contenido in (diccionarios or {}).items():
            mlflow.log_dict(contenido, f"datos/{nombre_archivo}")
    return True
