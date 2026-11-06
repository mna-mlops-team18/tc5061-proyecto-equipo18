"""Contrato de la API sin servidor ni MLflow: esquemas de Pydantic y armado de entrada y salida."""

import copy
import json

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from api import esquemas
from api.modelo import ErrorPrediccion, armar_entrada, validar_clases, validar_salida
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, COLUMNAS_P2
from tc5061_proyecto_equipo18.modeling.p3_clustering import columnas_variables

EJEMPLO = esquemas.EJEMPLO_PETICION_P1
INSTANCIAS = {
    "p1": esquemas.InstanciaP1,
    "p2": esquemas.InstanciaP2,
    "p3": esquemas.InstanciaP3,
}
PETICIONES = {
    "p1": esquemas.PeticionP1,
    "p2": esquemas.PeticionP2,
    "p3": esquemas.PeticionP3,
}
EJEMPLOS = {
    "p1": EJEMPLO,
    "p2": esquemas.EJEMPLO_PETICION_P2,
    "p3": esquemas.EJEMPLO_PETICION_P3,
}


def instancia(pregunta: str = "p1", **cambios) -> dict:
    """Primera fila del ejemplo de la pregunta con ``cambios`` encima."""
    fila = copy.deepcopy(EJEMPLOS[pregunta]["instancias"][0])
    fila.update(cambios)
    return fila


def campos_con_error(error: ValidationError) -> set[str]:
    return {str(parte) for detalle in error.errors() for parte in detalle["loc"]}


# =============================================================================
# El contrato coincide con los modelos y con la documentacion
# =============================================================================
def test_variables_de_p1_son_las_del_modelo():
    """Mismo nombre y orden que features.COLUMNAS_P1 (con lo que se entreno el modelo)."""
    assert esquemas.COLUMNAS_P1 == COLUMNAS_P1


def test_variables_de_p2_son_las_del_modelo():
    assert esquemas.COLUMNAS_P2 == COLUMNAS_P2


def test_variables_de_p3_son_las_del_agrupador():
    """Las que usa p3_clustering: los perfiles ``sal_*`` y ``lle_*`` y los tres agregados."""
    perfil = [c for c in esquemas.COLUMNAS_P3 if c.startswith(("sal_", "lle_"))]
    entrenamiento = pd.DataFrame(
        columns=["estacion_id", "latitud", "capacidad", *esquemas.COLUMNAS_P3]
    )
    assert set(esquemas.COLUMNAS_P3) == set(columnas_variables(entrenamiento))
    assert len(perfil) == 96
    assert len(esquemas.COLUMNAS_P3) == 99


def test_clases_de_p2_son_las_de_config():
    assert esquemas.CLASES_P2 == config.P2_CLASES


def test_archivo_de_ejemplo_de_p1_es_el_del_esquema():
    archivo = config.PROJ_ROOT / "api" / "ejemplos" / "peticion_p1.json"
    assert json.loads(archivo.read_text(encoding="utf-8")) == EJEMPLO


@pytest.mark.parametrize("pregunta", ["p1", "p2", "p3"])
def test_ejemplo_es_valido(pregunta):
    peticion = PETICIONES[pregunta].model_validate(EJEMPLOS[pregunta])
    assert len(peticion.instancias) == 2
    assert (
        list(peticion.instancias[0].model_dump(exclude={"estacion_id"}))
        == (esquemas.COLUMNAS[pregunta])
    )


@pytest.mark.parametrize("pregunta", ["p1", "p2", "p3"])
def test_estacion_id_es_opcional(pregunta):
    fila = instancia(pregunta)
    del fila["estacion_id"]
    assert INSTANCIAS[pregunta].model_validate(fila).estacion_id is None


def test_capacidad_acepta_decimales_como_la_firma_del_modelo():
    """La firma registrada la declara ``double``: 15.0 es lo que manda el entrenamiento."""
    assert esquemas.InstanciaP1.model_validate(instancia(capacidad=15.0)).capacidad == 15.0
    assert esquemas.InstanciaP2.model_validate(instancia("p2", capacidad=15.0)).capacidad == 15.0


# =============================================================================
# Entradas invalidas
# =============================================================================
@pytest.mark.parametrize(
    ("cambios", "campo"),
    [
        ({"hora_dia": 24}, "hora_dia"),
        ({"hora_dia": -1}, "hora_dia"),
        ({"dia_semana": 7}, "dia_semana"),
        ({"mes": 0}, "mes"),
        ({"capacidad": 0}, "capacidad"),
        ({"latitud": 19.43}, "latitud"),  # Ciudad de Mexico: fuera del area de MiBici
        ({"longitud": -99.13}, "longitud"),
        ({"salidas_lag_24h": -1}, "salidas_lag_24h"),
        ({"salidas_media_7d": 5000}, "salidas_media_7d"),
        ({"hora_dia": "8"}, "hora_dia"),  # texto en lugar de numero
        ({"hora_dia": 8.5}, "hora_dia"),  # decimal en un entero
        ({"es_festivo": True}, "es_festivo"),  # booleano en un entero
        ({"salidas_lag_24h": "7"}, "salidas_lag_24h"),
        ({"salidas_lag_24h": float("nan")}, "salidas_lag_24h"),
        ({"salidas_lag_24h": float("inf")}, "salidas_lag_24h"),
        ({"estacion_id": -5}, "estacion_id"),
        ({"hora": 8}, "hora"),  # campo desconocido (nombre mal escrito)
    ],
)
def test_p1_rechaza_valores_invalidos(cambios, campo):
    with pytest.raises(ValidationError) as error:
        esquemas.InstanciaP1.model_validate(instancia(**cambios))
    assert campo in campos_con_error(error.value)


@pytest.mark.parametrize(
    ("cambios", "campo"),
    [
        ({"bloque_dia": 4}, "bloque_dia"),  # los bloques empiezan en 0, 3, 6, ..., 21
        ({"bloque_dia": 24}, "bloque_dia"),
        ({"bloque_dia": 6.0}, "bloque_dia"),
        ({"dia_semana": 7}, "dia_semana"),
        ({"capacidad": 500}, "capacidad"),
        ({"latitud": 19.43}, "latitud"),
        ({"salidas_lag_1b": -1}, "salidas_lag_1b"),
        ({"salidas_previas_24h": 50000}, "salidas_previas_24h"),
        ({"neto_lag_1b": 5000}, "neto_lag_1b"),
        ({"neto_rel_lag_dia": -80}, "neto_rel_lag_dia"),
        ({"frac_vaciado_mismo_bloque_7d": 1.2}, "frac_vaciado_mismo_bloque_7d"),
        ({"frac_llenado_mismo_bloque_7d": -0.1}, "frac_llenado_mismo_bloque_7d"),
        ({"neto_lag_dia": float("nan")}, "neto_lag_dia"),
        ({"mes": "9"}, "mes"),
        ({"hora_dia": 8}, "hora_dia"),  # variable de P1, no de P2
    ],
)
def test_p2_rechaza_valores_invalidos(cambios, campo):
    with pytest.raises(ValidationError) as error:
        esquemas.InstanciaP2.model_validate(instancia("p2", **cambios))
    assert campo in campos_con_error(error.value)


def test_p2_acepta_flujos_negativos():
    """El neto (salidas - llegadas) es negativo cuando la estacion recibe bicicletas."""
    fila = instancia("p2", neto_lag_1b=-120.0, neto_rel_lag_dia=-4.5)
    assert esquemas.InstanciaP2.model_validate(fila).neto_lag_1b == -120.0


@pytest.mark.parametrize(
    ("cambios", "campo"),
    [
        ({"sal_lab_h08": 1.5}, "sal_lab_h08"),
        ({"lle_fds_h23": -0.01}, "lle_fds_h23"),
        ({"sal_fds_h00": "0.1"}, "sal_fds_h00"),
        ({"log_viajes_dia": -1}, "log_viajes_dia"),
        ({"log_viajes_dia": 40}, "log_viajes_dia"),
        ({"ratio_fin_semana": -0.5}, "ratio_fin_semana"),
        ({"balance_neto": 1.5}, "balance_neto"),
        ({"balance_neto": float("nan")}, "balance_neto"),
        ({"latitud": 20.67}, "latitud"),  # la ubicacion no entra al agrupador
    ],
)
def test_p3_rechaza_valores_invalidos(cambios, campo):
    with pytest.raises(ValidationError) as error:
        esquemas.InstanciaP3.model_validate(instancia("p3", **cambios))
    assert campo in campos_con_error(error.value)


def test_p3_el_perfil_horario_debe_sumar_uno():
    fila = instancia("p3", sal_lab_h08=0.9)  # el ejemplo ya suma 1: ahora suma mucho mas
    with pytest.raises(ValidationError, match=r"sal_lab_hHH deben sumar 1"):
        esquemas.InstanciaP3.model_validate(fila)


def test_p3_un_perfil_sin_viajes_suma_cero():
    """Una estacion sin viajes en fin de semana tiene todo el perfil en 0, no en 1/24."""
    fila = instancia("p3", **{f"sal_fds_h{hora:02d}": 0.0 for hora in range(24)})
    assert esquemas.InstanciaP3.model_validate(fila).sal_fds_h00 == 0.0


@pytest.mark.parametrize("pregunta", ["p1", "p2", "p3"])
def test_rechaza_una_variable_faltante(pregunta):
    fila = instancia(pregunta)
    columna = esquemas.COLUMNAS[pregunta][4]
    del fila[columna]
    with pytest.raises(ValidationError) as error:
        INSTANCIAS[pregunta].model_validate(fila)
    assert columna in campos_con_error(error.value)


@pytest.mark.parametrize(
    ("pregunta", "dia_semana", "es_fin_semana"),
    [(p, d, f) for p in ("p1", "p2") for d, f in [(5, 0), (6, 0), (0, 1), (4, 1)]],
)
def test_fin_de_semana_debe_coincidir_con_el_dia(pregunta, dia_semana, es_fin_semana):
    with pytest.raises(ValidationError, match="es_fin_semana debe ser"):
        INSTANCIAS[pregunta].model_validate(
            instancia(pregunta, dia_semana=dia_semana, es_fin_semana=es_fin_semana)
        )


@pytest.mark.parametrize("pregunta", ["p1", "p2", "p3"])
@pytest.mark.parametrize("cantidad", [0, esquemas.MAX_INSTANCIAS + 1])
def test_tamano_del_lote(pregunta, cantidad):
    with pytest.raises(ValidationError):
        PETICIONES[pregunta].model_validate({"instancias": [instancia(pregunta)] * cantidad})


@pytest.mark.parametrize("pregunta", ["p1", "p2", "p3"])
def test_lote_maximo_es_valido(pregunta):
    filas = [instancia(pregunta)] * esquemas.MAX_INSTANCIAS
    peticion = PETICIONES[pregunta].model_validate({"instancias": filas})
    assert len(peticion.instancias) == len(filas)


# =============================================================================
# Salidas invalidas
# =============================================================================
MODELO = {"nombre": "mibici-equipo18-p1-demanda", "version": "1"}


@pytest.mark.parametrize("valor", [-0.5, float("nan"), float("inf")])
def test_respuesta_p1_rechaza_salidas_negativas_o_no_finitas(valor):
    with pytest.raises(ValidationError):
        esquemas.RespuestaP1(
            modelo=MODELO, n=1, predicciones=[{"estacion_id": 1, "salidas_esperadas": valor}]
        )


def test_respuesta_exige_una_prediccion_por_instancia():
    with pytest.raises(ValidationError, match="predicciones"):
        esquemas.RespuestaP1(modelo=MODELO, n=2, predicciones=[{"salidas_esperadas": 1.0}])
    with pytest.raises(ValidationError, match="predicciones"):
        esquemas.RespuestaP3(modelo=MODELO, n=3, predicciones=[{"grupo": 0}])


@pytest.mark.parametrize(
    ("clase", "riesgo"),
    [(3, "riesgo_vaciado"), (-1, "sin_riesgo"), (1, "sin_riesgo"), (2, "riesgo_vaciado")],
)
def test_respuesta_p2_rechaza_clases_que_no_existen_o_con_otro_nombre(clase, riesgo):
    with pytest.raises(ValidationError):
        esquemas.PrediccionP2(clase=clase, riesgo=riesgo)


def test_respuesta_p2_acepta_cada_clase_con_su_nombre():
    for clase, riesgo in esquemas.CLASES_P2.items():
        assert esquemas.PrediccionP2(clase=clase, riesgo=riesgo).riesgo == riesgo


def test_respuesta_p3_rechaza_grupos_negativos():
    with pytest.raises(ValidationError):
        esquemas.PrediccionP3(grupo=-1)


# =============================================================================
# Armado de la entrada del modelo y revision de su salida
# =============================================================================
TIPOS = {
    "hora_dia": np.dtype("int32"),
    "latitud": np.dtype("float64"),
    "salidas_lag_24h": np.dtype("float32"),
}


def test_entrada_tiene_el_orden_y_los_tipos_de_la_firma():
    filas = [{"salidas_lag_24h": 7, "hora_dia": 8, "latitud": 20.67}]
    entrada = armar_entrada(filas, TIPOS)
    assert list(entrada.columns) == list(TIPOS)
    assert entrada.dtypes.to_dict() == TIPOS
    assert entrada.loc[0, "salidas_lag_24h"] == np.float32(7)


def test_entrada_falla_si_falta_una_variable():
    with pytest.raises(ValueError, match="latitud"):
        armar_entrada([{"hora_dia": 8, "salidas_lag_24h": 7.0}], TIPOS)


def test_salida_recorta_negativos_a_cero():
    assert validar_salida(np.array([-0.2, 1.5]), 2).tolist() == [0.0, 1.5]


def test_salida_acepta_una_columna():
    assert validar_salida(pd.DataFrame({"y": [1.0, 2.0]}), 2).tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("predicho", "n_filas", "mensaje"),
    [
        (np.array([1.0, 2.0, 3.0]), 2, "3 valores para 2 filas"),
        (np.array([1.0, np.nan]), 2, "no finitos"),
        (np.array([np.inf]), 1, "no finitos"),
    ],
)
def test_salida_invalida_es_error_de_prediccion(predicho, n_filas, mensaje):
    with pytest.raises(ErrorPrediccion, match=mensaje):
        validar_salida(predicho, n_filas)


def test_clases_aceptan_enteros_de_cualquier_tipo():
    assert validar_clases(np.array([0, 2, 1], dtype=np.int8), 3, {0, 1, 2}).tolist() == [0, 2, 1]
    assert validar_clases(np.array([0.0, 4.0]), 2).tolist() == [0, 4]  # P3: sin lista de clases
    assert validar_clases(pd.DataFrame({"y": [1, 1]}), 2).tolist() == [1, 1]


@pytest.mark.parametrize(
    ("predicho", "n_filas", "validas", "mensaje"),
    [
        (np.array([0, 1, 2]), 2, None, "3 valores para 2 filas"),
        (np.array([0.0, np.nan]), 2, None, "no finitos"),
        (np.array([0.5]), 1, None, "enteros no negativos"),
        (np.array([-1]), 1, None, "enteros no negativos"),
        (np.array([0, 3]), 2, {0, 1, 2}, r"fuera de \[0, 1, 2\]: \[0, 3\]"),
    ],
)
def test_clases_invalidas_son_error_de_prediccion(predicho, n_filas, validas, mensaje):
    with pytest.raises(ErrorPrediccion, match=mensaje):
        validar_clases(predicho, n_filas, validas)
