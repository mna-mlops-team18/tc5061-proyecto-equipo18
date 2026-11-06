"""Contrato de la API sin servidor ni MLflow: esquemas de Pydantic y armado de entrada y salida."""

import copy
import json

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from api import esquemas
from api.modelo import ErrorPrediccion, armar_entrada, validar_salida
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1

EJEMPLO = esquemas.EJEMPLO_PETICION_P1


def instancia(**cambios) -> dict:
    """Primera fila del ejemplo con ``cambios`` encima."""
    fila = copy.deepcopy(EJEMPLO["instancias"][0])
    fila.update(cambios)
    return fila


def campos_con_error(error: ValidationError) -> set[str]:
    return {str(parte) for detalle in error.errors() for parte in detalle["loc"]}


# =============================================================================
# El contrato coincide con el modelo y con la documentacion
# =============================================================================
def test_variables_de_la_api_son_las_del_modelo():
    """Mismo nombre y orden que features.COLUMNAS_P1 (con lo que se entreno el modelo)."""
    assert esquemas.COLUMNAS_P1 == COLUMNAS_P1


def test_archivo_de_ejemplo_es_el_del_esquema():
    archivo = config.PROJ_ROOT / "api" / "ejemplos" / "peticion_p1.json"
    assert json.loads(archivo.read_text(encoding="utf-8")) == EJEMPLO


def test_ejemplo_es_valido():
    peticion = esquemas.PeticionP1.model_validate(EJEMPLO)
    assert [i.estacion_id for i in peticion.instancias] == [101, 202]


def test_estacion_id_es_opcional():
    fila = instancia()
    del fila["estacion_id"]
    assert esquemas.InstanciaP1.model_validate(fila).estacion_id is None


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
def test_rechaza_valores_invalidos(cambios, campo):
    with pytest.raises(ValidationError) as error:
        esquemas.InstanciaP1.model_validate(instancia(**cambios))
    assert campo in campos_con_error(error.value)


def test_rechaza_una_variable_faltante():
    fila = instancia()
    del fila["mes"]
    with pytest.raises(ValidationError) as error:
        esquemas.InstanciaP1.model_validate(fila)
    assert "mes" in campos_con_error(error.value)


@pytest.mark.parametrize(("dia_semana", "es_fin_semana"), [(5, 0), (6, 0), (0, 1), (4, 1)])
def test_fin_de_semana_debe_coincidir_con_el_dia(dia_semana, es_fin_semana):
    with pytest.raises(ValidationError, match="es_fin_semana debe ser"):
        esquemas.InstanciaP1.model_validate(
            instancia(dia_semana=dia_semana, es_fin_semana=es_fin_semana)
        )


@pytest.mark.parametrize("cantidad", [0, esquemas.MAX_INSTANCIAS + 1])
def test_tamano_del_lote(cantidad):
    with pytest.raises(ValidationError):
        esquemas.PeticionP1.model_validate({"instancias": [instancia()] * cantidad})


def test_lote_maximo_es_valido():
    filas = [instancia()] * esquemas.MAX_INSTANCIAS
    assert len(esquemas.PeticionP1.model_validate({"instancias": filas}).instancias) == len(filas)


# =============================================================================
# Salidas invalidas
# =============================================================================
MODELO = {"nombre": "mibici-equipo18-p1-demanda", "version": "1"}


@pytest.mark.parametrize("valor", [-0.5, float("nan"), float("inf")])
def test_respuesta_rechaza_salidas_negativas_o_no_finitas(valor):
    with pytest.raises(ValidationError):
        esquemas.RespuestaP1(
            modelo=MODELO, n=1, predicciones=[{"estacion_id": 1, "salidas_esperadas": valor}]
        )


def test_respuesta_exige_una_prediccion_por_instancia():
    with pytest.raises(ValidationError, match="predicciones"):
        esquemas.RespuestaP1(modelo=MODELO, n=2, predicciones=[{"salidas_esperadas": 1.0}])


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
