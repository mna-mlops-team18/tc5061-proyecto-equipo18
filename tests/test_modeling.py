"""Pruebas de la base del experimento: particiones, registro en MLflow y orquestacion."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun, train

META = {
    "data_version": "abc123def456",
    "parametros": {"mes_inicio": "2024-10", "p1_rezagos_h": [24, 48]},
}


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


def test_separar_particiones_por_fecha_y_por_periodo():
    fechas = pd.to_datetime([config.FIN_ENTRENAMIENTO, config.FIN_VALIDACION, "2026-09-30"])
    partes = comun.separar_particiones(pd.DataFrame({"hora": fechas}), "hora")
    assert [len(partes[p]) for p in ("entrenamiento", "validacion", "prueba")] == [1, 1, 1]

    perfiles = pd.DataFrame({"periodo": ["entrenamiento", "entrenamiento", "prueba"]})
    partes = comun.separar_particiones(perfiles, columna_tiempo=None)
    assert [len(partes[p]) for p in ("entrenamiento", "validacion", "prueba")] == [2, 0, 1]


def test_muestrear_es_reproducible():
    df = pd.DataFrame({"x": range(100)})
    assert comun.muestrear(df, 10).equals(comun.muestrear(df, 10))
    assert comun.muestrear(df, 1000) is df


def test_run_registra_version_de_datos_metricas_y_modelo(mlflow_temporal):
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.ones(20)})
    modelo = DummyRegressor().fit(X, np.arange(20.0))

    with comun.iniciar_run("P1", "baseline", "media", {"estrategia": "mean"}, META) as run:
        comun.registrar_metricas({"mae": 1.5}, "validacion")
        uri = comun.registrar_modelo(modelo, X)

    datos = mlflow.get_run(run.info.run_id).data
    assert datos.tags["data_version"] == "abc123def456"
    assert datos.tags["pregunta"] == "P1" and datos.tags["rol"] == "baseline"
    assert "git_commit" in datos.tags
    assert datos.params["semilla"] == str(config.SEMILLA)
    assert datos.params["features.mes_inicio"] == "2024-10"
    assert datos.metrics == {"validacion_mae": 1.5}
    assert run.info.run_name == "p1-baseline-media"
    assert config.ruta_modelo(run.info.run_name).exists()
    assert mlflow.sklearn.load_model(uri).predict(X).shape == (20,)


def test_validaciones_de_run(mlflow_temporal):
    with pytest.raises(ValueError), comun.iniciar_run("P4", "baseline", "x", meta_features=META):
        pass
    with (
        comun.iniciar_run("P2", "candidato", "x", meta_features=META),
        pytest.raises(ValueError),
    ):
        comun.registrar_metricas({"f1": 0.5}, "test")


def test_train_omite_preguntas_sin_modulo(mlflow_temporal, monkeypatch):
    llamadas = []

    class ModuloFalso:
        @staticmethod
        def entrenar(roles):
            llamadas.append(roles)
            return ["run-1"]

    monkeypatch.setattr(train, "configurar_mlflow", lambda: "0")
    monkeypatch.setattr(
        train, "cargar_modulo", lambda pregunta: ModuloFalso if pregunta == "P1" else None
    )
    runs = train.ejecutar(["P1", "P2"], ("baseline",))
    assert runs == {"P1": ["run-1"]}
    assert llamadas == [("baseline",)]


def test_cargar_modulo_inexistente_devuelve_none():
    # Mientras no exista p3_clustering.py, train.py debe omitirlo en lugar de fallar
    if (config.PROJ_ROOT / "tc5061_proyecto_equipo18/modeling/p3_clustering.py").exists():
        pytest.skip("El modulo de P3 ya existe")
    assert train.cargar_modulo("P3") is None
