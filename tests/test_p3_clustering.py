"""Pruebas de P3: anti-fuga del baseline, estabilidad de grupos y runs completos en MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import busqueda, comun, p3_clustering

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
PREFIJOS = ("sal_lab", "lle_lab", "sal_fds", "lle_fds")
HORAS_PICO = {0: [7, 8, 9], 1: [17, 18, 19], 2: [12, 13, 14]}  # tres tipos de estacion
# Candidato de la entrega 1; los demas candidatos del catalogo se implementan despues
CANDIDATOS = ["kmeans_pca"]


@pytest.fixture
def datos_p3() -> dict[str, pd.DataFrame]:
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
                "codigo": f"GDL-{i:03d}",
                "nombre": f"Estacion {i}",
                "municipio": "GDL",
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
    return comun.separar_particiones(pd.DataFrame(filas), columna_tiempo=None)


def artefactos(run_id: str, carpeta: str) -> set[str]:
    return {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, carpeta)}


def test_variables_no_incluyen_ubicacion_ni_identificadores(datos_p3):
    columnas = p3_clustering.columnas_variables(datos_p3["entrenamiento"])
    assert len(columnas) == 4 * 24 + 3
    for prohibida in ("latitud", "longitud", "estacion_id", "viajes_periodo", "capacidad"):
        assert prohibida not in columnas


def test_cortes_del_baseline_salen_solo_de_entrenamiento(datos_p3):
    cortes = p3_clustering.cortes_cuartiles(datos_p3["entrenamiento"], 4)
    assert len(cortes) == 3

    # Una estacion de prueba con volumen enorme cae en el grupo mas alto y no mueve los cortes
    extremo = datos_p3["prueba"].assign(log_viajes_dia=100.0)
    grupos = p3_clustering.asignar_por_cortes(extremo, cortes)
    assert (grupos == 3).all()
    np.testing.assert_allclose(
        cortes, p3_clustering.cortes_cuartiles(datos_p3["entrenamiento"], 4)
    )


def test_estabilidad_es_perfecta_si_los_perfiles_no_cambian(datos_p3):
    perfiles = dict(datos_p3)
    perfiles["validacion"] = perfiles["entrenamiento"].assign(periodo="validacion")
    columnas = p3_clustering.columnas_variables(perfiles["entrenamiento"])
    prep = p3_clustering.ajustar_preprocesamiento(perfiles["entrenamiento"], columnas)
    cortes = p3_clustering.cortes_cuartiles(perfiles["entrenamiento"], 4)
    metricas, asignaciones, _ = p3_clustering.evaluar(
        prep, columnas, perfiles, lambda df, z: p3_clustering.asignar_por_cortes(df, cortes)
    )
    assert metricas["validacion"]["estabilidad_ari"] == pytest.approx(1.0)
    assert metricas["entrenamiento"]["grupos"] == 4
    assert set(asignaciones["periodo"]) == {"entrenamiento", "validacion", "prueba"}


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_cada_candidato_recupera_los_tipos_de_estacion(datos_p3, modelo):
    columnas = p3_clustering.columnas_variables(datos_p3["entrenamiento"])
    prep = p3_clustering.ajustar_preprocesamiento(datos_p3["entrenamiento"], columnas)
    z = prep.transform(datos_p3["entrenamiento"][columnas])
    hiperparametros = comun.resolver_hiperparametros("P3", modelo)
    agrupador = p3_clustering.ajustar_agrupador(modelo, hiperparametros, z)
    tipos = datos_p3["entrenamiento"]["estacion_id"] % 3
    assert adjusted_rand_score(tipos, agrupador.predict(z)) == pytest.approx(1.0)


def test_baseline_registra_metricas_sin_modelo(mlflow_temporal, datos_p3):
    run_id = p3_clustering.correr("cuartiles_volumen", datos=datos_p3, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.data.tags["etapa"] == "baseline" and run.data.params["grupos"] == "4"
    assert "cortes_log_viajes_dia" in run.data.params
    for periodo in ("validacion", "prueba"):
        assert f"{periodo}_estabilidad_ari" in run.data.metrics
    assert not run.outputs.model_outputs


@pytest.mark.parametrize("modelo", CANDIDATOS)
def test_candidato_registra_modelo_mapas_y_tablas(mlflow_temporal, datos_p3, modelo):
    run_id = p3_clustering.correr(modelo, datos=datos_p3, meta_features=META)
    run = mlflow.get_run(run_id)
    assert run.data.tags["etapa"] == "candidato" and run.data.tags["pregunta"] == "P3"
    for periodo in ("entrenamiento", "validacion", "prueba"):
        for metrica in ("silhouette", "davies_bouldin", "calinski_harabasz", "grupos"):
            assert f"{periodo}_{metrica}" in run.data.metrics
    assert run.data.metrics["entrenamiento_grupos"] == 3
    assert {"figuras/mapa_grupos.png", "figuras/perfiles_grupo.png"} <= artefactos(
        run_id, "figuras"
    )
    assert {"tablas/resumen_grupos.csv", "tablas/asignaciones.csv"} <= artefactos(run_id, "tablas")
    assert "mapas/mapa_grupos.html" in artefactos(run_id, "mapas")

    # El modelo registrado recibe el perfil de una estacion y devuelve su grupo
    (salida,) = run.outputs.model_outputs
    cargado = mlflow.pyfunc.load_model(f"models:/{salida.model_id}")
    columnas = p3_clustering.columnas_variables(datos_p3["prueba"])
    grupos = np.asarray(cargado.predict(datos_p3["prueba"][columnas]))
    assert len(np.unique(grupos)) == 3


def test_busqueda_de_k_con_corridas_anidadas(mlflow_temporal, datos_p3):
    padre_id = busqueda.ejecutar_busqueda(
        "p3", datos=datos_p3, meta_features=META, rejilla={"n_clusters": [2, 3, 4]}
    )
    padre = mlflow.get_run(padre_id)
    hijas = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{padre_id}'", output_format="list"
    )
    assert sorted(h.data.params["n_clusters"] for h in hijas) == ["2", "3", "4"]
    assert padre.data.params["mejor_n_clusters"] == "3"  # los datos tienen tres tipos
    assert "figuras/busqueda.png" in artefactos(padre_id, "figuras")
