"""Pruebas de P1: metricas, baseline sin fuga y runs completos en MLflow para cada modelo."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1
from tc5061_proyecto_equipo18.modeling import comun, p1_regresion

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
CANDIDATOS = [m for m in config.P1_MODELOS if not config.P1_MODELOS[m].get("baseline")]


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
        df["dia_semana"] = df["hora"].dt.dayofweek
        df["mes"] = df["hora"].dt.month
        df[OBJETIVO_P1] = rng.poisson(1 + df["salidas_lag_168h"])
        filas.append(df)
    return comun.separar_particiones(pd.concat(filas, ignore_index=True), "hora")


def artefactos(run_id: str, carpeta: str) -> set[str]:
    return {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, carpeta)}


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


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_cada_candidato_del_catalogo_se_puede_construir(modelo):
    hiperparametros = comun.resolver_hiperparametros("P1", modelo)
    assert hasattr(p1_regresion.construir_estimador(modelo, hiperparametros), "fit")


def test_baseline_registra_metricas_sin_modelo(mlflow_temporal, datos_p1):
    run_id = p1_regresion.correr("rezago_semanal", datos=datos_p1, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.info.run_name == "p1-baseline-rezago_semanal"
    assert run.data.tags["etapa"] == "baseline" and run.data.tags["familia_modelo"] == "regla"
    for particion in ("validacion", "prueba"):
        for metrica in ("mae", "rmse", "wape", "devianza_poisson", "r2"):
            assert f"{particion}_{metrica}" in run.data.metrics
    assert not run.outputs.model_outputs  # una regla no registra modelo


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_candidato_registra_modelo_figuras_y_conjuntos(mlflow_temporal, datos_p1, modelo):
    run_id = p1_regresion.correr(modelo, datos=datos_p1, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.data.tags["etapa"] == "candidato" and run.data.params["modelo"] == modelo
    assert run.data.metrics["validacion_mae"] >= 0
    assert run.data.metrics["costo_entrenamiento_seg"] >= 0
    assert run.data.metrics["costo_inferencia_ms_por_1000_filas"] >= 0
    figuras = artefactos(run_id, "figuras")
    for figura in ("real_vs_predicho", "residuales", "error_por_hora", "importancias"):
        assert f"figuras/{figura}.png" in figuras
    contextos = {
        t.value
        for d in run.inputs.dataset_inputs
        for t in d.tags
        if t.key == "mlflow.data.context"
    }
    assert contextos == {"entrenamiento", "validacion", "prueba"}

    (salida,) = run.outputs.model_outputs
    cargado = mlflow.pyfunc.load_model(f"models:/{salida.model_id}")
    predicho = cargado.predict(datos_p1["prueba"][COLUMNAS_P1])
    assert len(predicho) == len(datos_p1["prueba"])


def test_hiperparametros_cambiados_quedan_en_el_run(mlflow_temporal, datos_p1):
    hiperparametros = comun.resolver_hiperparametros("P1", "hgb_poisson", {"max_iter": "20"})
    run_id = p1_regresion.correr("hgb_poisson", hiperparametros, datos_p1, META)
    run = mlflow.get_run(run_id)
    assert run.data.params["max_iter"] == "20"
    assert run.info.run_name == "p1-candidato-hgb_poisson-max_iter20"
    assert run.data.metrics["entrenamiento_iteraciones"] <= 20
