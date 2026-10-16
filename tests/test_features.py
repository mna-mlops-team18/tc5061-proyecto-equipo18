"""Pruebas de features: sin fuga de informacion, etiquetas y particion temporal."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18.features import (
    COLUMNAS_P1,
    COLUMNAS_P2,
    asignar_particion,
    construir_dataset_p1,
    construir_dataset_p2,
    contar_por_estacion_hora,
    dias_festivos_mx,
    etiquetar_riesgo,
)

CORTE = pd.Timestamp("2025-01-20 00:00")


@pytest.fixture
def estaciones():
    return pd.DataFrame(
        {
            "estacion_id": np.array([1, 2], dtype="int32"),
            "capacidad": np.array([15, 20], dtype="int32"),
            "latitud": [20.67, 20.68],
            "longitud": [-103.35, -103.36],
        }
    )


@pytest.fixture
def conteos():
    rng = np.random.default_rng(0)
    horas = pd.date_range("2025-01-01", "2025-02-15 23:00", freq="h").astype("datetime64[s]")
    partes = [
        pd.DataFrame(
            {
                "estacion_id": np.int32(est),
                "hora": horas,
                "salidas": rng.poisson(3, len(horas)).astype("int32"),
                "llegadas": rng.poisson(3, len(horas)).astype("int32"),
            }
        )
        for est in (1, 2)
    ]
    return pd.concat(partes, ignore_index=True)


def _alterar_futuro(conteos, desde):
    """Copia de los conteos con todo lo ocurrido desde ``desde`` multiplicado por 10."""
    futuro = conteos.copy()
    mascara = futuro["hora"] >= desde
    futuro.loc[mascara, ["salidas", "llegadas"]] *= 10
    return futuro


def test_p1_no_usa_informacion_posterior_a_la_prediccion(conteos, estaciones):
    """Si cambia el futuro, las variables de las horas ya predecibles no deben cambiar.

    Con horizonte de 24 h, la fila de la hora t solo puede depender de datos <= t - 24 h.
    """
    original = construir_dataset_p1(conteos, estaciones).set_index(["estacion_id", "hora"])
    alterado = construir_dataset_p1(_alterar_futuro(conteos, CORTE), estaciones).set_index(
        ["estacion_id", "hora"]
    )
    predecibles = original.index.get_level_values("hora") < CORTE + pd.Timedelta(hours=24)
    pd.testing.assert_frame_equal(
        original.loc[predecibles, COLUMNAS_P1], alterado.loc[predecibles, COLUMNAS_P1]
    )
    # y el cambio si se nota despues del horizonte (la prueba no es trivial)
    assert not original.loc[~predecibles, COLUMNAS_P1].equals(
        alterado.loc[~predecibles, COLUMNAS_P1]
    )


def test_p2_no_usa_el_bloque_que_predice(conteos, estaciones):
    """Las variables de un bloque solo dependen de bloques ya cerrados."""
    original = construir_dataset_p2(conteos, estaciones).set_index(
        ["estacion_id", "bloque_inicio"]
    )
    alterado = construir_dataset_p2(_alterar_futuro(conteos, CORTE), estaciones).set_index(
        ["estacion_id", "bloque_inicio"]
    )
    predecibles = original.index.get_level_values("bloque_inicio") <= CORTE
    pd.testing.assert_frame_equal(
        original.loc[predecibles, COLUMNAS_P2], alterado.loc[predecibles, COLUMNAS_P2]
    )


def test_etiquetar_riesgo():
    neto = pd.Series([8, -8, 7, -7, 0])
    capacidad = pd.Series([15, 15, 15, 15, 15])
    assert etiquetar_riesgo(neto, capacidad, umbral=0.5).tolist() == [1, 2, 0, 0, 0]


def test_asignar_particion_es_cronologica():
    tiempos = pd.Series(
        pd.to_datetime(
            ["2026-03-31 23:00", "2026-04-01 00:00", "2026-06-30 23:00", "2026-07-01 00:00"]
        )
    )
    resultado = asignar_particion(tiempos, "2026-03-31", "2026-06-30").tolist()
    assert resultado == ["entrenamiento", "validacion", "validacion", "prueba"]


def test_dias_festivos_mx():
    festivos = dias_festivos_mx([2024, 2025, 2026])
    for fecha in [
        "2026-02-02",
        "2026-03-16",
        "2026-04-02",
        "2026-04-03",
        "2026-11-16",
        "2024-10-01",
    ]:
        assert pd.Timestamp(fecha) in festivos
    assert pd.Timestamp("2025-10-01") not in festivos


def test_contar_por_estacion_hora_completa_la_malla():
    viajes = pd.DataFrame(
        {
            "origen_id": np.array([1, 1], dtype="int32"),
            "destino_id": np.array([2, 2], dtype="int32"),
            "inicio_viaje": pd.to_datetime(["2025-01-01 08:10", "2025-01-01 10:50"]).astype(
                "datetime64[s]"
            ),
            "fin_viaje": pd.to_datetime(["2025-01-01 08:30", "2025-01-01 11:05"]).astype(
                "datetime64[s]"
            ),
        }
    )
    conteos = contar_por_estacion_hora(viajes, "2025-01", "2025-01")
    est1 = conteos[conteos["estacion_id"] == 1].set_index("hora")
    assert len(est1) == 24  # un dia completo, incluidas las horas sin viajes
    assert est1.loc["2025-01-01 08:00", "salidas"] == 1
    assert est1.loc["2025-01-01 09:00", "salidas"] == 0
    est2 = conteos[conteos["estacion_id"] == 2].set_index("hora")
    assert est2.loc["2025-01-01 11:00", "llegadas"] == 1
