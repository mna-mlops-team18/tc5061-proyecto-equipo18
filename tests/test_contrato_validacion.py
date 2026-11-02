"""Guarda del contrato de la validacion de datos frente al codigo que valida.

``validacion_datos.contrato`` vive en otro ambiente (Python 3.12, sin mlflow) y por eso copia
listas y la particion de ``features.py``. Estas pruebas corren en el ambiente principal, sin
datos ni DeepChecks, y fallan si alguien cambia una variable o un corte de un lado y no del
otro: sin ellas la validacion revisaria en silencio columnas que ya no son las del modelo.
"""

import pandas as pd
import pytest

from tc5061_proyecto_equipo18 import config, features
from tc5061_proyecto_equipo18.modeling import p3_clustering
from validacion_datos import contrato


@pytest.mark.parametrize(
    ("pregunta", "columnas", "objetivo"),
    [
        ("p1", features.COLUMNAS_P1, features.OBJETIVO_P1),
        ("p2", features.COLUMNAS_P2, features.OBJETIVO_P2),
    ],
)
def test_variables_y_etiqueta_son_las_de_features(pregunta, columnas, objetivo):
    esp = contrato.ESPECIFICACIONES[pregunta]
    # estacion_id no es entrada del modelo: se agrega solo para NewCategoryTrainTest
    assert list(esp.caracteristicas) == [*columnas, "estacion_id"]
    assert esp.etiqueta == objetivo


def test_listas_copiadas_son_iguales_a_las_de_features():
    assert contrato.COLUMNAS_P1 == features.COLUMNAS_P1
    assert contrato.COLUMNAS_P2 == features.COLUMNAS_P2
    assert contrato.OBJETIVO_P1 == features.OBJETIVO_P1
    assert contrato.OBJETIVO_P2 == features.OBJETIVO_P2


@pytest.mark.parametrize("pregunta", sorted(contrato.ESPECIFICACIONES))
def test_las_categoricas_estan_entre_las_variables(pregunta):
    esp = contrato.ESPECIFICACIONES[pregunta]
    assert set(esp.categoricas) <= set(esp.caracteristicas)
    # La llave permite revisar unicidad en el archivo completo: su tiempo es el del contrato
    assert esp.columna_tiempo is None or esp.columna_tiempo in esp.llave


def test_variables_de_p3_son_las_que_usa_el_modelo():
    assert contrato.VARIABLES_RESUMEN_P3 == p3_clustering.VARIABLES_RESUMEN
    identificadores = ["estacion_id", "periodo", "codigo", "nombre", "municipio"]
    atributos = ["capacidad", "latitud", "longitud", "viajes_periodo"]
    df = pd.DataFrame(
        columns=[
            *identificadores,
            *atributos,
            *contrato.PERFILES_P3,
            *contrato.VARIABLES_RESUMEN_P3,
        ]
    )
    del_modelo = p3_clustering.columnas_variables(df)
    assert sorted(del_modelo) == sorted(contrato.ESPECIFICACIONES["p3"].caracteristicas)


@pytest.mark.parametrize(
    "cortes",
    [
        (config.FIN_ENTRENAMIENTO, config.FIN_VALIDACION),
        ("2026-04-15", "2026-05-15"),
    ],
)
def test_la_particion_es_la_de_features(cortes):
    # Cruza los dos cortes de config y las horas de cada lado de la medianoche
    tiempos = pd.Series(pd.date_range("2026-03-30", "2026-07-02", freq="h"))
    esperado = features.asignar_particion(tiempos, *cortes)
    pd.testing.assert_series_equal(contrato.asignar_particion(tiempos, *cortes), esperado)
    assert set(esperado) == set(features.PARTICIONES)


def test_el_corte_de_entrenamiento_incluye_su_ultimo_dia():
    tiempos = pd.Series(pd.to_datetime(["2026-03-31 23:00", "2026-04-01 00:00"]))
    assert contrato.asignar_particion(tiempos).tolist() == ["entrenamiento", "validacion"]
