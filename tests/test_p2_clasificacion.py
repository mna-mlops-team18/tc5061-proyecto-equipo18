"""Pruebas de P2: metricas, baseline sin fuga y runs completos en MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P2, OBJETIVO_P2
from tc5061_proyecto_equipo18.modeling import comun, p2_clasificacion

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}


@pytest.fixture
def datos_p2():
    """Dataset sintetico de P2: tres estaciones, bloques de 3 h y las tres clases en cada periodo."""
    rng = np.random.default_rng(config.SEMILLA)
    bloques = pd.DatetimeIndex(
        [
            *pd.date_range("2026-03-01", periods=8 * 30, freq="3h"),  # entrenamiento
            *pd.date_range("2026-04-06", periods=8 * 7, freq="3h"),  # validacion
            *pd.date_range("2026-07-06", periods=8 * 7, freq="3h"),  # prueba
        ]
    )
    filas = []
    for estacion in (1, 2, 3):
        df = pd.DataFrame({"estacion_id": estacion, "bloque_inicio": bloques})
        for columna in COLUMNAS_P2:
            df[columna] = rng.normal(size=len(df))
        df["bloque_dia"] = df["bloque_inicio"].dt.hour
        df[OBJETIVO_P2] = rng.choice([0, 1, 2], size=len(df), p=[0.7, 0.15, 0.15])
        # Frecuencias historicas parecidas a la clase real, como en los datos de MiBici
        df["frac_vaciado_mismo_bloque_7d"] = np.clip(
            (df[OBJETIVO_P2] == 1) * 0.6 + rng.uniform(0, 0.3, len(df)), 0, 1
        )
        df["frac_llenado_mismo_bloque_7d"] = np.clip(
            (df[OBJETIVO_P2] == 2) * 0.6 + rng.uniform(0, 0.3, len(df)), 0, 1
        ) * (1 - df["frac_vaciado_mismo_bloque_7d"])
        filas.append(df)
    return comun.separar_particiones(pd.concat(filas, ignore_index=True), "bloque_inicio")


def test_metricas_clasificacion_perfectas():
    real = np.array([0, 1, 2, 0])
    probabilidades = np.eye(3)[real]
    metricas = p2_clasificacion.metricas_clasificacion(real, probabilidades)
    assert metricas["f1_macro"] == pytest.approx(1.0)
    assert metricas["recall_vaciado"] == pytest.approx(1.0)
    assert metricas["pr_auc_llenado"] == pytest.approx(1.0)


def test_pr_auc_sin_positivos_es_nan():
    real = np.array([0, 0, 1])
    metricas = p2_clasificacion.metricas_clasificacion(real, np.eye(3)[real])
    assert np.isnan(metricas["pr_auc_llenado"])


def test_baseline_usa_frecuencias_del_mismo_bloque():
    datos = pd.DataFrame(
        {
            "frac_vaciado_mismo_bloque_7d": [0.0, 0.8, 0.1],
            "frac_llenado_mismo_bloque_7d": [0.0, 0.1, 0.7],
        }
    )
    probabilidades = p2_clasificacion.probabilidades_baseline(datos)
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1)
    assert list(np.argmax(probabilidades, axis=1)) == [0, 1, 2]


def test_probabilidades_modelo_rellena_clases_ausentes():
    X = pd.DataFrame({"x": np.arange(40.0)})
    y = np.array([0, 2] * 20)  # sin la clase 1 en el ajuste
    modelo = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    probabilidades = p2_clasificacion.probabilidades_modelo(modelo, X)
    assert probabilidades.shape == (40, 3)
    assert (probabilidades[:, 1] == 0).all()


def test_entrenar_registra_baseline_y_candidato(mlflow_temporal, datos_p2):
    run_ids = p2_clasificacion.entrenar(datos=datos_p2, meta_features=META)
    assert len(run_ids) == 2

    baseline, candidato = (mlflow.get_run(run_id) for run_id in run_ids)
    for run in (baseline, candidato):
        assert run.data.params["data_version"] == "abc123def456"
        for particion in ("validacion", "prueba"):
            for metrica in ("f1_macro", "recall_vaciado", "precision_llenado", "pr_auc_vaciado"):
                assert f"{particion}_{metrica}" in run.data.metrics
    assert baseline.data.params["modelo"] == "frecuencia_mismo_bloque"
    assert candidato.data.params["class_weight"] == "balanced"

    artefactos = {
        a.path for a in mlflow.MlflowClient().list_artifacts(candidato.info.run_id, "figuras")
    }
    for figura in ("matriz_confusion", "curvas_pr", "riesgo_por_bloque", "importancias"):
        assert any(figura in ruta for ruta in artefactos)
    assert config.ruta_modelo(candidato.info.run_name).exists()
    assert not config.ruta_modelo(baseline.info.run_name).exists()
