"""Pruebas del registro de las etapas de datos (puntos de entrada ingest y features)."""

import mlflow

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.seguimiento import registrar_etapa_datos


def test_fuera_de_mlflow_run_no_registra_nada(mlflow_temporal):
    assert not registrar_etapa_datos("datos-ingesta", "abc123def456", metricas={"meses": 24})
    assert mlflow.active_run() is None


def test_dentro_de_mlflow_run_deja_version_y_metadatos(mlflow_temporal, monkeypatch):
    """Simula `mlflow run . -e ingest`: el proyecto crea el run y pasa su id por MLFLOW_RUN_ID."""
    cliente = mlflow.MlflowClient()
    proyecto = cliente.create_run(mlflow.get_experiment_by_name("prueba").experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", proyecto.info.run_id)

    assert registrar_etapa_datos(
        "datos-ingesta",
        "abc123def456",
        parametros={"mes_inicio": "2024-10"},
        metricas={"meses": 24, "total_viajes": 9_043_167},
        archivos=[config.MANIFIESTO_DATOS],
        diccionarios={"resumen.json": {"filas": 3}},
    )
    run = mlflow.get_run(proyecto.info.run_id)
    assert run.info.run_name == "datos-ingesta"
    assert run.data.tags["etapa"] == "datos" and run.data.tags["data_version"] == "abc123def456"
    assert run.data.params["mes_inicio"] == "2024-10"
    assert run.data.metrics["total_viajes"] == 9_043_167
    rutas = {a.path for a in cliente.list_artifacts(run.info.run_id, "datos")}
    assert {"datos/manifiesto_datos.json", "datos/resumen.json"} <= rutas
