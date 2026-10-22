"""Pruebas de P2: metricas, baseline sin fuga, pesos de clase y runs completos en MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P2, OBJETIVO_P2
from tc5061_proyecto_equipo18.modeling import comun, p2_clasificacion

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
CANDIDATOS = [m for m in config.P2_MODELOS if not config.P2_MODELOS[m].get("baseline")]


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


def artefactos(run_id: str, carpeta: str) -> set[str]:
    return {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, carpeta)}


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


def test_peso_riesgo_define_los_pesos_de_clase():
    assert p2_clasificacion.pesos_clase(10) == {0: 1.0, 1: 10.0, 2: 10.0}
    hiperparametros = comun.resolver_hiperparametros("P2", "hgb_ponderado", {"peso_riesgo": "3"})
    estimador = p2_clasificacion.construir_estimador("hgb_ponderado", hiperparametros)
    assert estimador.class_weight == {0: 1.0, 1: 3.0, 2: 3.0}
    assert "peso_riesgo" in hiperparametros  # se registra como hiperparametro del run


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_cada_candidato_del_catalogo_se_puede_construir(modelo):
    hiperparametros = comun.resolver_hiperparametros("P2", modelo)
    assert hasattr(p2_clasificacion.construir_estimador(modelo, hiperparametros), "fit")


def test_baseline_registra_metricas_sin_modelo(mlflow_temporal, datos_p2):
    run_id = p2_clasificacion.correr("frecuencia_mismo_bloque", datos=datos_p2, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.data.tags["etapa"] == "baseline"
    for particion in ("validacion", "prueba"):
        for metrica in ("f1_macro", "recall_vaciado", "precision_llenado", "pr_auc_vaciado"):
            assert f"{particion}_{metrica}" in run.data.metrics
    assert not run.outputs.model_outputs


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_candidato_registra_modelo_figuras_y_conjuntos(mlflow_temporal, datos_p2, modelo):
    run_id = p2_clasificacion.correr(modelo, datos=datos_p2, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.data.tags["etapa"] == "candidato" and run.data.params["modelo"] == modelo
    assert 0 <= run.data.metrics["validacion_f1_macro"] <= 1
    assert run.data.metrics["costo_entrenamiento_seg"] >= 0
    figuras = artefactos(run_id, "figuras")
    for figura in ("matriz_confusion", "curvas_pr", "riesgo_por_bloque", "importancias"):
        assert f"figuras/{figura}.png" in figuras
    assert len(run.inputs.dataset_inputs) == 3

    (salida,) = run.outputs.model_outputs
    cargado = mlflow.pyfunc.load_model(f"models:/{salida.model_id}")
    predicho = np.asarray(cargado.predict(datos_p2["prueba"][COLUMNAS_P2]))
    assert set(np.unique(predicho)) <= {0, 1, 2}
