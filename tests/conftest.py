"""Fixtures compartidas por las pruebas del experimento (P1, P2 y P3)."""

import mlflow
import pytest

from tc5061_proyecto_equipo18.modeling import comun


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """MLflow en una base SQLite temporal, sin servidor; nada se escribe en el repo."""
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    comun.configurar_mlflow(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experimento="prueba",
        artefactos_dir=tmp_path / "mlruns",
    )
    yield
    while mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(None)
