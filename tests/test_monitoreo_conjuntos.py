"""Conjuntos del monitoreo de P1 y coherencia de las fechas de config.py.

Corren en el ambiente principal, sin datos ni Evidently: ``monitoreo/conjuntos.py`` es pandas
puro. Si alguien mueve una fecha y deja de cumplirse algo que el A/B testing necesita (por
ejemplo, que la evaluacion sea posterior al entrenamiento de ambas versiones), fallan aqui.
"""

import pandas as pd
import pytest

from monitoreo import conjuntos, sintetico
from tc5061_proyecto_equipo18 import config, features


def fecha(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto)


# =============================================================================
# Fechas de config.py
# =============================================================================
def test_referencia_es_la_particion_de_entrenamiento():
    inicio, fin = config.MONITOREO_REFERENCIA
    assert inicio == f"{config.MES_INICIO}-01"
    assert fin == config.FIN_ENTRENAMIENTO


def test_periodo_actual_es_posterior_a_lo_que_vio_el_champion():
    # Ni entrenamiento ni validacion (con la que se eligio el champion)
    assert fecha(config.MONITOREO_ACTUAL[0]) > fecha(config.FIN_VALIDACION)
    fin_datos = pd.Period(config.MES_FIN, "M").end_time.normalize()
    assert fecha(config.MONITOREO_ACTUAL[1]) <= fin_datos


def test_ab_evalua_despues_del_entrenamiento_de_ambas_versiones():
    inicio_ab, fin_ab = (fecha(texto) for texto in config.AB_EVALUACION)
    assert inicio_ab > fecha(config.FIN_ENTRENAMIENTO)  # version A (champion)
    assert inicio_ab > fecha(config.REENTRENAMIENTO_FIN)  # version B (challenger)
    # La evaluacion es parte del periodo actual y B si incluye datos de ese periodo
    assert fecha(config.MONITOREO_ACTUAL[0]) <= inicio_ab <= fin_ab
    assert fin_ab <= fecha(config.MONITOREO_ACTUAL[1])
    assert fecha(config.REENTRENAMIENTO_FIN) >= fecha(config.MONITOREO_ACTUAL[0])


def test_comparaciones_bien_formadas():
    assert list(config.MONITOREO_COMPARACIONES) == ["principal", "sin_nuevas", "misma_temporada"]
    referencia = config.MONITOREO_COMPARACIONES["misma_temporada"]["referencia"]
    # La misma temporada un anio antes, dentro del entrenamiento del champion
    assert fecha(referencia[0]) >= fecha(config.MONITOREO_REFERENCIA[0])
    assert fecha(referencia[1]) <= fecha(config.FIN_ENTRENAMIENTO)
    for especificacion in config.MONITOREO_COMPARACIONES.values():
        assert set(especificacion) == {"referencia", "actual", "solo_conocidas"}
        for periodo in (especificacion["referencia"], especificacion["actual"]):
            conjuntos.limites(periodo)  # fechas validas y en orden


def test_monitoreo_mensual_cubre_validacion_y_periodo_actual():
    inicio, fin = config.MONITOREO_MENSUAL
    assert fecha(inicio) == fecha(config.FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)
    assert fin == config.MONITOREO_ACTUAL[1]


# =============================================================================
# Variables
# =============================================================================
def test_variables_son_las_del_modelo_mas_el_objetivo():
    assert conjuntos.VARIABLES == [*features.COLUMNAS_P1, features.OBJETIVO_P1]
    assert sorted(conjuntos.CATEGORICAS + conjuntos.NUMERICAS) == sorted(conjuntos.VARIABLES)
    assert set(conjuntos.CATEGORICAS) == {
        "hora_dia",
        "dia_semana",
        "mes",
        "es_fin_semana",
        "es_festivo",
    }


# =============================================================================
# Periodos y comparaciones (datos sinteticos)
# =============================================================================
@pytest.fixture(scope="module")
def p1():
    return sintetico.p1_sintetico(conocidas=8, nuevas=3, cada_dias=6)


def test_en_periodo_incluye_el_ultimo_dia_completo():
    tiempos = pd.Series(
        pd.to_datetime(
            ["2026-06-30 23:00", "2026-07-01 00:00", "2026-09-30 23:00", "2026-10-01 00:00"]
        )
    )
    assert conjuntos.en_periodo(tiempos, ("2026-07-01", "2026-09-30")).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_limites_rechaza_un_periodo_al_reves():
    with pytest.raises(ValueError, match="anterior al inicio"):
        conjuntos.limites(("2026-09-30", "2026-07-01"))


def test_estaciones_conocidas_son_las_de_la_referencia(p1):
    assert conjuntos.estaciones_conocidas(p1) == set(range(1, 9))


def test_principal_incluye_nuevas_y_sin_nuevas_las_quita(p1):
    conocidas = conjuntos.estaciones_conocidas(p1)
    principal = conjuntos.armar_comparacion(
        p1, "principal", config.MONITOREO_COMPARACIONES["principal"], conocidas, muestra=10**9
    )
    sin_nuevas = conjuntos.armar_comparacion(
        p1, "sin_nuevas", config.MONITOREO_COMPARACIONES["sin_nuevas"], conocidas, muestra=10**9
    )
    assert principal.info["estaciones_nuevas_periodo"] == 3
    assert principal.info["fraccion_filas_nuevas_periodo"] == pytest.approx(3 / 11)
    assert principal.info["estaciones_actual"] == 11
    # Sin nuevas: las estaciones contadas son las que se comparan; las nuevas siguen informadas
    assert sin_nuevas.info["estaciones_actual"] == 8
    assert sin_nuevas.info["estaciones_nuevas_periodo"] == 3
    assert set(principal.actual["estacion_id"]) == set(range(1, 12))
    assert set(sin_nuevas.actual["estacion_id"]) == conocidas
    # Las fechas de cada conjunto respetan su periodo
    assert principal.referencia["hora"].max() < fecha("2026-04-01")
    assert principal.actual["hora"].min() >= fecha("2026-07-01")


def test_muestra_reproducible_y_acotada(p1):
    conocidas = conjuntos.estaciones_conocidas(p1)
    especificacion = config.MONITOREO_COMPARACIONES["principal"]
    una = conjuntos.armar_comparacion(p1, "principal", especificacion, conocidas, muestra=500)
    otra = conjuntos.armar_comparacion(p1, "principal", especificacion, conocidas, muestra=500)
    assert len(una.referencia) == len(una.actual) == 500
    assert una.info["filas_referencia"] > 500  # el tamano completo queda para el resumen
    pd.testing.assert_frame_equal(una.referencia, otra.referencia)


def test_comparacion_sin_filas_falla_con_mensaje_claro(p1):
    especificacion = {
        "referencia": ("2020-01-01", "2020-01-31"),
        "actual": config.MONITOREO_ACTUAL,
        "solo_conocidas": False,
    }
    with pytest.raises(ValueError, match="no tiene filas"):
        conjuntos.armar_comparacion(p1, "vacia", especificacion, set())


def test_para_evidently_quita_int8_y_deja_solo_las_variables(p1):
    listo = conjuntos.para_evidently(p1.head(50))
    assert list(listo.columns) == conjuntos.VARIABLES
    assert all(listo[columna].dtype == "int64" for columna in conjuntos.CATEGORICAS)
    assert all(listo[columna].dtype == "float64" for columna in conjuntos.NUMERICAS)


def test_estadisticas_describen_el_cambio():
    referencia = pd.DataFrame({columna: [1, 1, 2, 2] for columna in conjuntos.VARIABLES})
    actual = referencia.copy()
    actual["salidas"] = [2, 2, 4, 4]
    actual["mes"] = [9, 9, 9, 2]
    tabla = conjuntos.estadisticas(referencia, actual).set_index("variable")
    assert tabla.loc["salidas", "cambio_media"] == pytest.approx(1.0)
    assert tabla.loc["mes", "mayor_cambio"] == "9: 0.0% -> 75.0%"
