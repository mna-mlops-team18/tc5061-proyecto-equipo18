"""Fixtures compartidas por las pruebas del experimento (P1, P2 y P3)."""

import mlflow
import pytest

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """MLflow en una base SQLite temporal; nada se escribe en el repo."""
    monkeypatch.setattr(config, "FIGURAS_EXPERIMENTO_DIR", tmp_path / "figuras")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    comun.configurar_mlflow(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experimento="prueba",
        artefactos_dir=tmp_path / "mlruns",
    )
    yield
    mlflow.set_tracking_uri(None)
