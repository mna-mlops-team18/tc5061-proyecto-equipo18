"""Pruebas de P1: metricas, baseline sin fuga y runs completos en MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1
from tc5061_proyecto_equipo18.modeling import comun, p1_regresion

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}


@pytest.fixture
def datos_p1():
    """Dataset sintetico de P1 con dos estaciones y algunos dias en cada particion."""
    rng = np.random.default_rng(config.SEMILLA)
    horas = pd.DatetimeIndex(
        [
            *pd.date_range("2026-03-25", periods=24 * 7, freq="h"),  # entrenamiento
            *pd.date_range("2026-04-06", periods=24 * 2, freq="h"),  # validacion
            *pd.date_range("2026-07-06", periods=24 * 2, freq="h"),  # prueba
        ]
    )
    filas = []
    for estacion in (1, 2):
        df = pd.DataFrame({"estacion_id": estacion, "hora": horas})
        for columna in COLUMNAS_P1:
            df[columna] = rng.poisson(3, len(df)).astype(float)
        df["hora_dia"] = df["hora"].dt.hour
        df[OBJETIVO_P1] = rng.poisson(1 + df["salidas_lag_168h"])
        filas.append(df)
    return comun.separar_particiones(pd.concat(filas, ignore_index=True), "hora")


def test_metricas_regresion():
    metricas = p1_regresion.metricas_regresion(np.array([0, 2, 4]), np.array([0, 2, 2]))
    assert metricas["mae"] == pytest.approx(2 / 3)
    assert metricas["wape"] == pytest.approx(2 / 6)
    # Una prediccion de 0 no rompe la devianza Poisson
    assert np.isfinite(metricas["devianza_poisson"])


def test_baseline_es_el_rezago_semanal(datos_p1):
    validacion = datos_p1["validacion"]
    np.testing.assert_array_equal(
        p1_regresion.predecir_baseline(validacion), validacion["salidas_lag_168h"]
    )


def test_entrenar_registra_baseline_y_candidato(mlflow_temporal, datos_p1):
    run_ids = p1_regresion.entrenar(datos=datos_p1, meta_features=META)
    assert len(run_ids) == 2

    baseline, candidato = (mlflow.get_run(run_id) for run_id in run_ids)
    for run in (baseline, candidato):
        assert run.data.params["data_version"] == "abc123def456"
        for particion in ("validacion", "prueba"):
            for metrica in ("mae", "rmse", "wape", "devianza_poisson", "r2"):
                assert f"{particion}_{metrica}" in run.data.metrics
    assert baseline.data.params["modelo"] == "rezago_semanal"
    assert candidato.data.params["modelo"] == "hgb_poisson"
    assert candidato.data.params["loss"] == "poisson"

    artefactos = {
        a.path for a in mlflow.MlflowClient().list_artifacts(candidato.info.run_id, "figuras")
    }
    assert any("importancias" in ruta for ruta in artefactos)
    assert config.ruta_modelo(candidato.info.run_name).exists()
    assert not config.ruta_modelo(baseline.info.run_name).exists()


def test_entrenar_solo_un_rol(mlflow_temporal, datos_p1):
    run_ids = p1_regresion.entrenar(("baseline",), datos=datos_p1, meta_features=META)
    assert mlflow.get_run(run_ids[0]).data.tags["rol"] == "baseline"
