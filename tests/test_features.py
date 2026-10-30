"""Pruebas de features: sin fuga de informacion, etiquetas, particion temporal y casos limite."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo18.features import (
    COLUMNAS_P1,
    COLUMNAS_P2,
    agregar_bloques,
    asignar_particion,
    construir_dataset_p1,
    construir_dataset_p2,
    contar_por_estacion_hora,
    dias_festivos_mx,
    etiquetar_riesgo,
    filtrar_viajes_validos,
    media_movil_previa,
    perfiles_estacion,
    validar_malla_regular,
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


# =============================================================================
# Casos limite de la limpieza y la agregacion
# =============================================================================
def _viajes(duraciones, origenes, destinos=None):
    inicio = pd.Timestamp("2025-01-01 08:00")
    destinos = destinos if destinos is not None else origenes
    return pd.DataFrame(
        {
            "viaje_id": np.arange(len(duraciones)),
            "inicio_viaje": [inicio] * len(duraciones),
            "fin_viaje": [inicio + pd.Timedelta(seconds=s) for s in duraciones],
            "origen_id": np.array(origenes, dtype="int32"),
            "destino_id": np.array(destinos, dtype="int32"),
            "duracion_seg": np.array(duraciones, dtype="int64"),
        }
    )


def test_filtrar_viajes_validos_respeta_los_limites_y_cuenta_cada_causa_una_vez(estaciones):
    # 59 s y 4 h + 1 s fuera; 60 s y 4 h exactas dentro; estacion 99 desconocida;
    # el ultimo es corto y de estacion desconocida a la vez: cuenta solo como corto
    viajes = _viajes([59, 60, 14_400, 14_401, 300, 30], [1, 1, 2, 2, 99, 99])
    validos, resumen = filtrar_viajes_validos(viajes, estaciones, 60, 14_400)
    assert validos["duracion_seg"].tolist() == [60, 14_400]
    assert resumen == {
        "viajes_entrada": 6,
        "descartados_duracion_corta": 2,
        "descartados_duracion_larga": 1,
        "descartados_estacion_desconocida": 1,
        "viajes_validos": 2,
    }


def test_filtrar_viajes_validos_sin_viajes(estaciones):
    validos, resumen = filtrar_viajes_validos(_viajes([], []), estaciones)
    assert validos.empty
    assert resumen["viajes_entrada"] == resumen["viajes_validos"] == 0


def test_contar_por_estacion_hora_ignora_lo_que_cae_fuera_del_rango():
    """Un viaje que sale el 31 de enero a las 23:50 y llega el 1 de febrero (fuera del rango)."""
    viajes = pd.DataFrame(
        {
            "origen_id": np.array([1, 3], dtype="int32"),
            "destino_id": np.array([2, 3], dtype="int32"),
            "inicio_viaje": pd.to_datetime(["2025-01-31 23:50", "2024-12-15 10:00"]),
            "fin_viaje": pd.to_datetime(["2025-02-01 00:10", "2024-12-15 10:20"]),
        }
    )
    conteos = contar_por_estacion_hora(viajes, "2025-01", "2025-01")
    assert set(conteos["estacion_id"]) == {1}  # la 2 solo llega en febrero y la 3 solo en dic.
    assert conteos["hora"].max() == pd.Timestamp("2025-01-31 23:00")
    assert conteos["salidas"].sum() == 1 and conteos["llegadas"].sum() == 0


def test_contar_por_estacion_hora_sin_viajes_en_el_rango_da_un_mensaje_claro():
    viajes = _viajes([300], [1]).assign(
        inicio_viaje=pd.Timestamp("2024-12-15 10:00"), fin_viaje=pd.Timestamp("2024-12-15 10:05")
    )
    with pytest.raises(ValueError, match="No hay viajes entre 2025-01 y 2025-01"):
        contar_por_estacion_hora(viajes, "2025-01", "2025-01")


def test_etiquetar_riesgo_en_el_umbral_exacto():
    """Con capacidad 16 y umbral 0.5 el limite es 8: alcanzarlo ya es riesgo."""
    neto = pd.Series([8, -8, 7, -7])
    capacidad = pd.Series([16, 16, 16, 16])
    assert etiquetar_riesgo(neto, capacidad, umbral=0.5).tolist() == [1, 2, 0, 0]
    assert etiquetar_riesgo(neto, capacidad, umbral=0.25).tolist() == [1, 2, 1, 2]


def test_agregar_bloques_suma_por_bloque_y_rechaza_bloques_que_no_dividen_24(conteos):
    bloques = agregar_bloques(conteos, horas_bloque=3)
    primero = conteos[(conteos["estacion_id"] == 1) & (conteos["hora"] < "2025-01-01 03:00")]
    fila = bloques.iloc[0]
    assert fila["salidas"] == primero["salidas"].sum()
    assert fila["neto"] == primero["salidas"].sum() - primero["llegadas"].sum()
    assert len(bloques) == len(conteos) // 3
    with pytest.raises(ValueError, match="dividir 24"):
        agregar_bloques(conteos, horas_bloque=5)


def test_validar_malla_regular_detecta_horas_faltantes(conteos):
    validar_malla_regular(conteos, "hora", pd.Timedelta(hours=1))  # malla completa: no falla
    con_hueco = conteos.drop(index=10)
    with pytest.raises(ValueError, match="regular"):
        validar_malla_regular(con_hueco, "hora", pd.Timedelta(hours=1))


def test_p1_rechaza_rezagos_menores_al_horizonte_y_mallas_con_huecos(conteos, estaciones):
    with pytest.raises(ValueError, match="futuro"):
        construir_dataset_p1(conteos, estaciones, horizonte_h=24, rezagos_h=(12, 24))
    with pytest.raises(ValueError, match="regular"):
        construir_dataset_p1(conteos.drop(index=10), estaciones)


def test_p1_y_p2_no_tienen_faltantes_y_excluyen_estaciones_sin_capacidad(conteos, estaciones):
    retirada = estaciones.assign(capacidad=np.array([15, 0], dtype="int32"))
    p1 = construir_dataset_p1(conteos, retirada)
    p2 = construir_dataset_p2(conteos, retirada)
    for df, columnas in ((p1, COLUMNAS_P1), (p2, COLUMNAS_P2)):
        assert set(df["estacion_id"]) == {1}
        assert not df[columnas].isna().any().any()
    assert not p1.duplicated(["estacion_id", "hora"]).any()
    assert not p2.duplicated(["estacion_id", "bloque_inicio"]).any()


def test_media_movil_previa_nunca_usa_el_valor_de_la_fila():
    df = pd.DataFrame({"estacion_id": 1, "x": np.arange(1.0, 11.0)})
    media = media_movil_previa(df, "x", ventana=3, desfase=1, grupos=["estacion_id"])
    assert media.iloc[:3].isna().all()  # aun no hay 3 valores previos
    assert media.iloc[3] == pytest.approx(2.0)  # promedio de 1, 2 y 3
    alterado = df.assign(x=df["x"].where(df.index != 9, 1000.0))
    alterada = media_movil_previa(alterado, "x", ventana=3, desfase=1, grupos=["estacion_id"])
    assert alterada.iloc[9] == media.iloc[9]  # cambiar la fila 9 no cambia su propia media


def test_asignar_particion_todo_en_un_solo_periodo():
    tiempos = pd.Series(pd.to_datetime(["2025-01-01", "2025-06-01"]))
    assert set(asignar_particion(tiempos, "2026-03-31", "2026-06-30")) == {"entrenamiento"}


def test_dias_festivos_mx_en_un_anio_de_cambio_de_gobierno():
    festivos = dias_festivos_mx([2030])
    assert pd.Timestamp("2030-10-01") in festivos  # transmision del Poder Ejecutivo
    assert pd.Timestamp("2030-04-18") in festivos  # jueves santo (Pascua: 21 de abril)
    assert pd.Timestamp("2030-04-19") in festivos  # viernes santo
    assert len(festivos) == 10


# =============================================================================
# P3: perfiles de uso por estacion
# =============================================================================
@pytest.fixture
def conteos_perfil():
    """Dos semanas (lunes 3 a domingo 16 de marzo de 2025) para dos estaciones.

    Estacion 1: sale a las 8 y llega a las 18 en dias laborales; sale 2 veces a las 12 en fin
    de semana. Estacion 2: solo sale a las 7 en dias laborales.
    """
    horas = pd.date_range("2025-03-03", "2025-03-16 23:00", freq="h")
    partes = []
    for estacion in (1, 2):
        df = pd.DataFrame({"estacion_id": np.int32(estacion), "hora": horas})
        laboral = df["hora"].dt.dayofweek < 5
        hora = df["hora"].dt.hour
        if estacion == 1:
            df["salidas"] = np.where(laboral & (hora == 8), 1, 0)
            df["salidas"] += np.where(~laboral & (hora == 12), 2, 0)
            df["llegadas"] = np.where(laboral & (hora == 18), 1, 0)
        else:
            df["salidas"] = np.where(laboral & (hora == 7), 1, 0)
            df["llegadas"] = 0
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def test_perfiles_estacion_reparte_por_hora_y_tipo_de_dia(conteos_perfil):
    perfiles = perfiles_estacion(
        conteos_perfil, pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-17"), min_viajes=1
    ).set_index("estacion_id")
    uno = perfiles.loc[1]
    assert uno["sal_lab_h08"] == 1 and uno["lle_lab_h18"] == 1 and uno["sal_fds_h12"] == 1
    for prefijo in ("sal_lab", "lle_lab", "sal_fds"):
        assert sum(uno[f"{prefijo}_h{h:02d}"] for h in range(24)) == pytest.approx(1)
    assert uno["viajes_periodo"] == 28  # 10 dias laborales x 2 + 4 de fin de semana x 2
    assert uno["balance_neto"] == pytest.approx((10 - 18) / 28)
    assert uno["ratio_fin_semana"] == pytest.approx(1.0)  # 2 viajes diarios en ambos tipos

    dos = perfiles.loc[2]  # sin actividad en fin de semana: ceros, nunca NaN
    assert dos["ratio_fin_semana"] == 0
    assert sum(dos[f"sal_fds_h{h:02d}"] for h in range(24)) == 0
    assert not perfiles.isna().any().any()


def test_perfiles_estacion_excluye_estaciones_con_pocos_viajes(conteos_perfil):
    desde, hasta = pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-17")
    perfiles = perfiles_estacion(conteos_perfil, desde, hasta, min_viajes=20)
    assert perfiles["estacion_id"].tolist() == [1]  # la 2 solo tiene 10 viajes
    assert perfiles_estacion(conteos_perfil, desde, hasta, min_viajes=1_000).empty
