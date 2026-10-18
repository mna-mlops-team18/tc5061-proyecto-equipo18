"""Pruebas de P3: anti-fuga del baseline, estabilidad de grupos y runs completos en MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import p3_clustering

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
PREFIJOS = ("sal_lab", "lle_lab", "sal_fds", "lle_fds")
HORAS_PICO = {0: [7, 8, 9], 1: [17, 18, 19], 2: [12, 13, 14]}  # tres tipos de estacion


@pytest.fixture
def datos_p3():
    """Perfiles sinteticos de P3: 3 tipos de estacion (por hora pico) en los 3 periodos."""
    rng = np.random.default_rng(config.SEMILLA)
    filas = []
    for periodo, n in (("entrenamiento", 90), ("validacion", 88), ("prueba", 86)):
        for i in range(n):
            tipo = i % 3
            base = np.zeros(24)
            base[HORAS_PICO[tipo]] = 1
            fila = {
                "estacion_id": i,
                "codigo": f"E{i}",
                "nombre": f"Estacion {i}",
                "municipio": "Guadalajara",
                "capacidad": 20,
                "latitud": 20.6 + 0.02 * tipo + rng.normal(0, 0.003),
                "longitud": -103.3 + 0.02 * tipo + rng.normal(0, 0.003),
                "periodo": periodo,
                "viajes_periodo": 1000,
                "log_viajes_dia": rng.normal(3 + tipo, 0.3),
                "ratio_fin_semana": rng.uniform(0, 1),
                "balance_neto": rng.normal(0, 0.1),
            }
            for prefijo in PREFIJOS:
                perfil = base + rng.uniform(0, 0.2, 24)
                perfil = perfil / perfil.sum()
                fila.update({f"{prefijo}_h{h:02d}": perfil[h] for h in range(24)})
            filas.append(fila)
    return pd.DataFrame(filas)


def test_variables_no_incluyen_ubicacion_ni_identificadores(datos_p3):
    columnas = p3_clustering.columnas_variables(datos_p3)
    assert len(columnas) == 4 * 24 + 3
    for prohibida in ("latitud", "longitud", "estacion_id", "viajes_periodo", "capacidad"):
        assert prohibida not in columnas


def test_cortes_del_baseline_salen_solo_de_entrenamiento(datos_p3):
    perfiles = p3_clustering.separar_periodos(datos_p3)
    cortes = p3_clustering.cortes_cuartiles(perfiles["entrenamiento"])
    assert len(cortes) == config.P3_GRUPOS_BASELINE - 1

    # Una estacion de prueba con volumen enorme cae en el grupo mas alto y no mueve los cortes
    extremo = perfiles["prueba"].assign(log_viajes_dia=100.0)
    grupos = p3_clustering.asignar_por_cortes(extremo, cortes)
    assert (grupos == config.P3_GRUPOS_BASELINE - 1).all()
    np.testing.assert_allclose(cortes, p3_clustering.cortes_cuartiles(perfiles["entrenamiento"]))


def test_estabilidad_es_perfecta_si_los_perfiles_no_cambian(datos_p3):
    perfiles = p3_clustering.separar_periodos(datos_p3)
    perfiles["validacion"] = perfiles["entrenamiento"].assign(periodo="validacion")
    columnas = p3_clustering.columnas_variables(perfiles["entrenamiento"])
    prep = p3_clustering.ajustar_preprocesamiento(perfiles["entrenamiento"], columnas)
    cortes = p3_clustering.cortes_cuartiles(perfiles["entrenamiento"])
    metricas, asignaciones, _ = p3_clustering.evaluar(
        prep, columnas, perfiles, lambda df, Z: p3_clustering.asignar_por_cortes(df, cortes)
    )
    assert metricas["validacion_estabilidad_ari"] == pytest.approx(1.0)
    assert set(asignaciones["periodo"]) == {"entrenamiento", "validacion", "prueba"}


def test_entrenar_registra_baseline_y_candidato(mlflow_temporal, datos_p3):
    run_ids = p3_clustering.entrenar(datos=datos_p3, meta_features=META)
    assert len(run_ids) == 2

    baseline, candidato = (mlflow.get_run(run_id) for run_id in run_ids)
    for run in (baseline, candidato):
        assert run.data.params["data_version"] == "abc123def456"
        assert run.data.tags["pregunta"] == "P3"
        for periodo in ("entrenamiento", "validacion", "prueba"):
            assert f"{periodo}_silhouette" in run.data.metrics
        for periodo in ("validacion", "prueba"):
            assert f"{periodo}_estabilidad_ari" in run.data.metrics
    assert baseline.data.params["modelo"] == "cuartiles_volumen"
    assert candidato.data.params["modelo"] == "kmeans_pca"
    # Con tres tipos de estacion bien separados, K-means los recupera
    assert candidato.data.params["k"] == "3"

    artefactos = {
        a.path for a in mlflow.MlflowClient().list_artifacts(candidato.info.run_id, "figuras")
    }
    for figura in ("mapa_grupos", "perfiles_horarios", "barrido_k"):
        assert any(figura in ruta for ruta in artefactos)
    assert config.ruta_modelo(candidato.info.run_name).exists()
    assert not config.ruta_modelo(baseline.info.run_name).exists()
