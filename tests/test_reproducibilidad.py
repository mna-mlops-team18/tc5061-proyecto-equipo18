"""Reproducibilidad del entrenamiento: misma semilla, mismos modelos y mismas metricas.

La semilla del proyecto (``config.SEMILLA``) se pasa a todo lo aleatorio: el muestreo de filas,
la particion interna del gradient boosting para early stopping, la regresion logistica, la PCA,
K-means y la mezcla gaussiana. Estas pruebas entrenan dos veces lo mismo y exigen resultados
identicos (no solo parecidos), primero con los estimadores y despues con el flujo completo que
registra en MLflow. Tambien comprueban que la semilla si importa: con otra semilla el resultado
cambia, asi que la igualdad no es casualidad.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, COLUMNAS_P2, OBJETIVO_P1, OBJETIVO_P2
from tc5061_proyecto_equipo18.modeling import comun, p1_regresion, p2_clasificacion, p3_clustering

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
CANDIDATOS = {
    pregunta: [m for m, spec in config.MODELOS[pregunta].items() if not spec.get("baseline")]
    for pregunta in ("P1", "P2", "P3")
}


# =============================================================================
# Datos sinteticos (la misma semilla del proyecto para que los datos tambien se repitan)
# =============================================================================
@pytest.fixture
def datos_p1():
    """P1: tres estaciones, 14 dias de entrenamiento y 2 de validacion y de prueba."""
    rng = np.random.default_rng(config.SEMILLA)
    horas = pd.DatetimeIndex(
        [
            *pd.date_range("2026-03-18", periods=24 * 14, freq="h"),
            *pd.date_range("2026-04-06", periods=24 * 2, freq="h"),
            *pd.date_range("2026-07-06", periods=24 * 2, freq="h"),
        ]
    )
    partes = []
    for estacion in (1, 2, 3):
        df = pd.DataFrame({"estacion_id": estacion, "hora": horas})
        for columna in COLUMNAS_P1:
            df[columna] = rng.poisson(3, len(df)).astype(float)
        df["hora_dia"] = df["hora"].dt.hour
        df["dia_semana"] = df["hora"].dt.dayofweek
        df["mes"] = df["hora"].dt.month
        df[OBJETIVO_P1] = rng.poisson(1 + df["salidas_lag_168h"])
        partes.append(df)
    return comun.separar_particiones(pd.concat(partes, ignore_index=True), "hora")


@pytest.fixture
def datos_p2():
    """P2: tres estaciones con bloques de 3 h y las tres clases en cada periodo."""
    rng = np.random.default_rng(config.SEMILLA)
    bloques = pd.DatetimeIndex(
        [
            *pd.date_range("2026-03-01", periods=8 * 30, freq="3h"),
            *pd.date_range("2026-04-06", periods=8 * 7, freq="3h"),
            *pd.date_range("2026-07-06", periods=8 * 7, freq="3h"),
        ]
    )
    partes = []
    for estacion in (1, 2, 3):
        df = pd.DataFrame({"estacion_id": estacion, "bloque_inicio": bloques})
        for columna in COLUMNAS_P2:
            df[columna] = rng.normal(size=len(df))
        df["bloque_dia"] = df["bloque_inicio"].dt.hour
        df[OBJETIVO_P2] = rng.choice([0, 1, 2], size=len(df), p=[0.7, 0.15, 0.15])
        df["frac_vaciado_mismo_bloque_7d"] = rng.uniform(0, 0.5, len(df))
        df["frac_llenado_mismo_bloque_7d"] = rng.uniform(0, 0.5, len(df))
        partes.append(df)
    return comun.separar_particiones(pd.concat(partes, ignore_index=True), "bloque_inicio")


@pytest.fixture
def datos_p3():
    """P3: perfiles de 90 estaciones de tres tipos (hora pico distinta) en los tres periodos."""
    rng = np.random.default_rng(config.SEMILLA)
    picos = {0: [7, 8, 9], 1: [17, 18, 19], 2: [12, 13, 14]}
    filas = []
    for periodo in ("entrenamiento", "validacion", "prueba"):
        for i in range(90):
            tipo = i % 3
            fila = {
                "estacion_id": i, "codigo": f"GDL-{i:03d}", "nombre": f"Estacion {i}",
                "municipio": "GDL", "capacidad": 20, "latitud": 20.6, "longitud": -103.3,
                "periodo": periodo, "viajes_periodo": 1000,
                "log_viajes_dia": rng.normal(3 + tipo, 0.3),
                "ratio_fin_semana": rng.uniform(0, 1), "balance_neto": rng.normal(0, 0.1),
            }  # fmt: skip
            for prefijo in ("sal_lab", "lle_lab", "sal_fds", "lle_fds"):
                perfil = rng.uniform(0, 0.2, 24)
                perfil[picos[tipo]] += 1
                perfil = perfil / perfil.sum()
                fila.update({f"{prefijo}_h{h:02d}": perfil[h] for h in range(24)})
            filas.append(fila)
    return comun.separar_particiones(pd.DataFrame(filas), columna_tiempo=None)


def semillas(estimador) -> dict:
    """Todos los parametros ``random_state`` de un estimador (tambien dentro de un Pipeline)."""
    return {
        clave: valor
        for clave, valor in estimador.get_params(deep=True).items()
        if clave.endswith("random_state")
    }


# =============================================================================
# Los estimadores usan la semilla del proyecto
# =============================================================================
@pytest.mark.parametrize(
    ("modulo", "pregunta", "modelo"),
    [(p1_regresion, "P1", m) for m in CANDIDATOS["P1"]]
    + [(p2_clasificacion, "P2", m) for m in CANDIDATOS["P2"]],
)
def test_estimadores_de_p1_y_p2_usan_la_semilla_del_proyecto(modulo, pregunta, modelo):
    estimador = modulo.construir_estimador(
        modelo, comun.resolver_hiperparametros(pregunta, modelo)
    )
    # El GLM Poisson es determinista y no tiene random_state; los demas deben usar la semilla
    assert all(valor == config.SEMILLA for valor in semillas(estimador).values())


@pytest.mark.parametrize("modelo", ["kmeans_pca", "gmm_pca"])
def test_agrupadores_de_p3_usan_la_semilla_del_proyecto(datos_p3, modelo):
    columnas = p3_clustering.columnas_variables(datos_p3["entrenamiento"])
    prep = p3_clustering.ajustar_preprocesamiento(datos_p3["entrenamiento"], columnas)
    z = prep.transform(datos_p3["entrenamiento"][columnas])
    agrupador = p3_clustering.ajustar_agrupador(
        modelo, comun.resolver_hiperparametros("P3", modelo), z
    )
    assert agrupador.random_state == config.SEMILLA
    assert prep.named_steps["pca"].random_state == config.SEMILLA


# =============================================================================
# Dos entrenamientos con la misma semilla dan exactamente lo mismo
# =============================================================================
def entrenar_p1(modelo: str, datos: dict) -> np.ndarray:
    muestra = comun.muestrear(datos["entrenamiento"], config.P1_MAX_FILAS_ENTRENAMIENTO)
    hiperparametros = comun.resolver_hiperparametros("P1", modelo)
    estimador = p1_regresion.construir_estimador(modelo, hiperparametros)
    estimador.fit(muestra[COLUMNAS_P1], muestra[OBJETIVO_P1])
    return estimador.predict(datos["prueba"][COLUMNAS_P1])


def entrenar_p2(modelo: str, datos: dict) -> np.ndarray:
    hiperparametros = comun.resolver_hiperparametros("P2", modelo)
    estimador = p2_clasificacion.construir_estimador(modelo, hiperparametros)
    estimador.fit(datos["entrenamiento"][COLUMNAS_P2], datos["entrenamiento"][OBJETIVO_P2])
    return p2_clasificacion.probabilidades_modelo(estimador, datos["prueba"][COLUMNAS_P2])


def agrupar_p3(modelo: str, datos: dict) -> tuple[np.ndarray, np.ndarray]:
    columnas = p3_clustering.columnas_variables(datos["entrenamiento"])
    prep = p3_clustering.ajustar_preprocesamiento(datos["entrenamiento"], columnas)
    z = prep.transform(datos["entrenamiento"][columnas])
    agrupador = p3_clustering.ajustar_agrupador(
        modelo, comun.resolver_hiperparametros("P3", modelo), z
    )
    z_prueba = prep.transform(datos["prueba"][columnas])
    return z_prueba, agrupador.predict(z_prueba)


@pytest.mark.parametrize("modelo", CANDIDATOS["P1"])
def test_p1_misma_semilla_mismas_predicciones(datos_p1, modelo):
    np.testing.assert_array_equal(entrenar_p1(modelo, datos_p1), entrenar_p1(modelo, datos_p1))


@pytest.mark.parametrize("modelo", CANDIDATOS["P2"])
def test_p2_misma_semilla_mismas_probabilidades(datos_p2, modelo):
    np.testing.assert_array_equal(entrenar_p2(modelo, datos_p2), entrenar_p2(modelo, datos_p2))


@pytest.mark.parametrize("modelo", CANDIDATOS["P3"])
def test_p3_misma_semilla_mismo_espacio_y_mismos_grupos(datos_p3, modelo):
    z_1, grupos_1 = agrupar_p3(modelo, datos_p3)
    z_2, grupos_2 = agrupar_p3(modelo, datos_p3)
    np.testing.assert_array_equal(z_1, z_2)
    np.testing.assert_array_equal(grupos_1, grupos_2)


def test_otra_semilla_cambia_el_resultado(datos_p1, monkeypatch):
    """Control: si la semilla no influyera, las pruebas de arriba no demostrarian nada."""
    con_semilla = entrenar_p1("hgb_poisson", datos_p1)
    monkeypatch.setattr(config, "SEMILLA", config.SEMILLA + 1)
    # Otra semilla separa otro 10 % para early stopping y el modelo cambia
    assert not np.array_equal(con_semilla, entrenar_p1("hgb_poisson", datos_p1))


def test_muestreo_reproducible_y_sensible_a_la_semilla():
    df = pd.DataFrame({"x": range(1_000)})
    assert comun.muestrear(df, 100).equals(comun.muestrear(df, 100))
    assert not comun.muestrear(df, 100).equals(comun.muestrear(df, 100, semilla=7))


# =============================================================================
# Flujo completo: dos corridas registradas en MLflow con la misma semilla
# =============================================================================
def metricas_del_modelo(run_id: str) -> dict[str, float]:
    """Metricas del run sin las de costo (tiempos y tamanio dependen de la computadora)."""
    metricas = mlflow.get_run(run_id).data.metrics
    return {clave: valor for clave, valor in metricas.items() if not clave.startswith("costo_")}


def predicciones_registradas(run_id: str, X: pd.DataFrame) -> np.ndarray:
    (salida,) = mlflow.get_run(run_id).outputs.model_outputs
    return np.asarray(mlflow.pyfunc.load_model(f"models:/{salida.model_id}").predict(X))


@pytest.mark.parametrize(
    ("modulo", "modelo", "fixture", "columnas"),
    [
        (p1_regresion, "hgb_poisson", "datos_p1", COLUMNAS_P1),
        (p2_clasificacion, "hgb_ponderado", "datos_p2", COLUMNAS_P2),
        (p3_clustering, "kmeans_pca", "datos_p3", None),
    ],
)
def test_dos_corridas_registradas_dan_las_mismas_metricas_y_predicciones(
    mlflow_temporal, request, modulo, modelo, fixture, columnas
):
    datos = request.getfixturevalue(fixture)
    primera = modulo.correr(modelo, datos=datos, meta_features=META)
    segunda = modulo.correr(modelo, datos=datos, meta_features=META)
    assert primera != segunda  # dos runs distintos en MLflow...
    metricas = metricas_del_modelo(primera)
    assert any(clave.startswith("prueba_") for clave in metricas)
    # ...con exactamente las mismas metricas (tolerancia 0; una metrica NaN cuenta como igual)
    assert metricas_del_modelo(segunda) == pytest.approx(metricas, rel=0, abs=0, nan_ok=True)

    if columnas is None:  # P3: el modelo recibe las variables de clustering
        columnas = p3_clustering.columnas_variables(datos["prueba"])
    X = datos["prueba"][columnas]
    np.testing.assert_array_equal(
        predicciones_registradas(primera, X), predicciones_registradas(segunda, X)
    )
