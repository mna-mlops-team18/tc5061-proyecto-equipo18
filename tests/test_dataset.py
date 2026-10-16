"""Pruebas de la ingesta: variantes de formato observadas en el portal de MiBici."""

import pandas as pd
import pytest

from tc5061_proyecto_equipo18.dataset import (
    ALIAS_VIAJES,
    COLUMNAS_PERSONALES,
    TIPOS_VIAJES,
    detectar_codificacion,
    elegir_formato_fecha,
    hash_contenido,
    leer_estaciones_crudas,
    leer_viajes_crudos,
    mapear_columnas,
    meses_en_rango,
    normalizar_nombre,
    texto_a_float,
    tipos_canonicos,
)

# Coordenadas con 17 cifras significativas, como las de la nomenclatura real: con ellas
# pd.to_numeric no siempre da el double mas cercano y el resultado cambia segun el procesador.
COORDENADAS_17_CIFRAS = ["20.194892411578156", "-103.93877840801609", "-103.14841858398947"]

ENCABEZADO_REAL = [
    "Viaje_Id", "Usuario_Id", "Genero", "A\u00f1o_de_nacimiento",
    "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id",
]  # fmt: skip


@pytest.mark.parametrize(
    "original",
    [
        "A\u00f1o_de_nacimiento",
        "A\ufffdo_de_nacimiento",
        "A\u00c3\u00b1o_de_nacimiento",
        "Birth_year",
    ],
)
def test_variantes_de_anio_de_nacimiento_se_reconocen(original):
    assert mapear_columnas([original], ALIAS_VIAJES) == {original: "anio_nacimiento"}


def test_normalizar_nombre_quita_acentos_y_espacios():
    assert normalizar_nombre(" Inicio del viaje ") == "inicio_del_viaje"
    assert normalizar_nombre("A\u00f1o_de_nacimiento") == "ano_de_nacimiento"


def test_encabezado_real_cubre_todo_el_esquema():
    mapa = mapear_columnas(ENCABEZADO_REAL, ALIAS_VIAJES)
    assert set(mapa.values()) >= {
        "viaje_id",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
    }


@pytest.mark.parametrize(
    ("valores", "mes", "esperado"),
    [
        (["01/10/2024 0:00", "15/10/2024 7:05", "31/10/2024 23:59"], "2024-10", "%d/%m/%Y %H:%M"),
        (["2025-06-01 00:00:08", "2025-06-30 23:59:59"], "2025-06", "%Y-%m-%d %H:%M:%S"),
        # dia <= 12: solo el mes esperado distingue dia/mes de mes/dia
        (["05/10/2024 8:00", "11/10/2024 9:30"], "2024-10", "%d/%m/%Y %H:%M"),
    ],
)
def test_elegir_formato_fecha(valores, mes, esperado):
    assert elegir_formato_fecha(pd.Series(valores), mes) == esperado


def test_meses_en_rango_incluye_extremos():
    assert meses_en_rango("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def _escribir(tmp_path, nombre, texto, codificacion):
    ruta = tmp_path / nombre
    ruta.write_bytes(texto.encode(codificacion))
    return ruta


def test_detectar_codificacion(tmp_path):
    utf8 = _escribir(tmp_path, "a.csv", "\ufeffA\u00f1o\n1\n", "utf-8")
    cp1252 = _escribir(tmp_path, "b.csv", "A\u00f1o\n1\n", "cp1252")
    assert detectar_codificacion(utf8) == "utf-8-sig"
    assert detectar_codificacion(cp1252) == "cp1252"


def test_leer_viajes_crudos_normaliza_y_descarta_datos_personales(tmp_path):
    filas = [
        ",".join(f'"{c}"' for c in ENCABEZADO_REAL),
        '"10","7","M","1990","2024-12-01 08:00:00","2024-12-01 08:10:30","5","6"',
        '"11","8",NULL,"1985","2024-12-01 09:00:00","2024-12-01 09:20:00","6",NULL',  # sin destino
        '"10","7","M","1990","2024-12-01 08:00:00","2024-12-01 08:10:30","5","6"',  # duplicado
        '"12","9","F","2001","2024-12-02 18:00:00","2024-12-02 18:05:00","6","5"',
    ]
    ruta = _escribir(tmp_path, "datos_abiertos_2024_12.csv", "\n".join(filas) + "\n", "cp1252")

    lectura = leer_viajes_crudos(ruta, "2024-12")

    assert lectura.codificacion == "cp1252"
    assert lectura.filas_crudas == 4
    assert lectura.descartadas_nulos == 1
    assert lectura.descartadas_duplicados == 1
    assert list(lectura.datos["viaje_id"]) == [10, 12]
    assert list(lectura.datos["duracion_seg"]) == [630, 300]
    assert not set(COLUMNAS_PERSONALES) & set(lectura.datos.columns)
    assert {c: str(t) for c, t in lectura.datos.dtypes.items()} == TIPOS_VIAJES


def test_texto_a_float_redondea_igual_en_cualquier_computadora():
    textos = pd.Series([*COORDENADAS_17_CIFRAS, " 20.5 ", None, "NULL"], dtype="str")
    valores = texto_a_float(textos)
    assert valores.iloc[:4].tolist() == [float(t) for t in [*COORDENADAS_17_CIFRAS, "20.5"]]
    assert valores.iloc[4:].isna().all()


def test_leer_estaciones_crudas(tmp_path):
    """Coordenadas exactas (mismo hash en Windows y en macOS), codigo y estaciones retiradas."""
    texto = (
        "id,name,latitude,longitude,dpcapacity\n"
        "2,(ZPN-002) Av. Patria,20.194892411578156,-103.93877840801609,0\n"
        "1,(GDL-001) C. Epigmenio,20.67,-103.35,15\n"
    )
    ruta = _escribir(tmp_path, "nomenclatura.csv", texto, "utf-8")
    estaciones, codificacion = leer_estaciones_crudas(ruta)
    assert codificacion == "utf-8-sig"
    assert estaciones["estacion_id"].tolist() == [1, 2]
    assert estaciones["latitud"].tolist() == [20.67, float("20.194892411578156")]
    assert estaciones["longitud"].tolist() == [-103.35, float("-103.93877840801609")]
    assert estaciones["codigo"].tolist() == ["GDL-001", "ZPN-002"]
    assert estaciones["retirada"].tolist() == [False, True]  # capacidad 0


def test_hash_contenido_sobrevive_parquet(tmp_path):
    """El hash del manifiesto debe ser igual tras escribir y leer Parquet en cualquier equipo.

    Parquet no tiene unidad de segundos y pyarrow devuelve las fechas en milisegundos;
    ``tipos_canonicos`` las regresa al tipo del esquema antes de calcular el hash.
    """
    pytest.importorskip("pyarrow")
    df = tipos_canonicos(
        pd.DataFrame(
            {
                "viaje_id": [10, 11],
                "inicio_viaje": pd.to_datetime(["2025-06-01 08:00:00", "2025-06-01 09:30:05"]),
                "fin_viaje": pd.to_datetime(["2025-06-01 08:10:30", "2025-06-01 09:45:00"]),
                "origen_id": [5, 6],
                "destino_id": [6, 5],
                "duracion_seg": [630, 895],
            }
        ),
        TIPOS_VIAJES,
    )
    ruta = tmp_path / "viajes.parquet"
    df.to_parquet(ruta, index=False, compression="zstd")
    leido = tipos_canonicos(pd.read_parquet(ruta), TIPOS_VIAJES)
    assert hash_contenido(leido) == hash_contenido(df)


def test_hash_contenido_detecta_cambios():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]})
    otro = df.copy()
    otro.loc[1, "a"] = 9
    assert hash_contenido(df) == hash_contenido(df.copy())
    assert hash_contenido(df) != hash_contenido(otro)
