"""Conjuntos de referencia y actual para el monitoreo de P1 (pandas puro, sin Evidently).

Lo usan ``monitoreo/drift.py`` (ambiente ``.venv-monitoreo``, Python 3.13) y las pruebas del
ambiente principal (``tests/test_monitoreo_conjuntos.py``). Solo importa pandas, ``config`` y
el contrato de la validacion de datos, que tiene la copia vigilada de las variables de P1
(``tests/test_contrato_validacion.py`` la compara con ``features.py``). Por eso corre igual en
los dos ambientes.

Definiciones (fechas en ``config.py``, seccion Monitoreo):

- **Referencia:** las filas de P1 con que se entreno el champion (``MONITOREO_REFERENCIA``).
- **Actual:** el trimestre mas reciente (``MONITOREO_ACTUAL``), que representa produccion.
- **Estacion conocida:** tiene filas en la referencia. Las demas son estaciones nuevas: abrieron
  despues del corte de entrenamiento y el champion nunca las vio.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tc5061_proyecto_equipo18 import config
from validacion_datos import contrato

COLUMNA_TIEMPO = "hora"
COLUMNA_ESTACION = "estacion_id"
# Variables que se vigilan: las entradas del modelo y el objetivo (para ver tambien el drift de
# la etiqueta). estacion_id no es entrada del modelo: las estaciones nuevas se cuentan aparte.
VARIABLES = [*contrato.COLUMNAS_P1, contrato.OBJETIVO_P1]
CATEGORICAS = [columna for columna in contrato.CATEGORICAS_P1 if columna in VARIABLES]
NUMERICAS = [columna for columna in VARIABLES if columna not in CATEGORICAS]


# =============================================================================
# Periodos
# =============================================================================
def limites(periodo: tuple[str, str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``("2026-07-01", "2026-09-30")`` -> desde las 00:00 del 1 de julio hasta antes del 1 de
    octubre (el ultimo dia es inclusivo)."""
    inicio, fin = (pd.Timestamp(fecha) for fecha in periodo)
    if fin < inicio:
        raise ValueError(f"Periodo invalido {periodo}: el fin es anterior al inicio")
    return inicio, fin + pd.Timedelta(days=1)


def en_periodo(tiempos: pd.Series, periodo: tuple[str, str]) -> pd.Series:
    """Mascara de las filas cuyo tiempo cae dentro del periodo (ambos dias inclusivos)."""
    desde, hasta = limites(periodo)
    return (tiempos >= desde) & (tiempos < hasta)


# =============================================================================
# Carga y armado de las comparaciones
# =============================================================================
def cargar_p1(ruta=config.DATASET_P1) -> pd.DataFrame:
    """Lee del Parquet de P1 solo la estacion, la hora, las variables y el objetivo."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corre primero: make data && make features")
    return pd.read_parquet(ruta, columns=[COLUMNA_ESTACION, COLUMNA_TIEMPO, *VARIABLES])


def estaciones_conocidas(
    p1: pd.DataFrame, referencia: tuple[str, str] = config.MONITOREO_REFERENCIA
) -> set[int]:
    """Estaciones que tienen filas en la referencia (las que el champion vio al entrenar)."""
    filas = p1.loc[en_periodo(p1[COLUMNA_TIEMPO], referencia), COLUMNA_ESTACION]
    return set(filas.unique().tolist())


def muestrear(df: pd.DataFrame, n: int, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Muestra aleatoria reproducible de a lo mas ``n`` filas."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=semilla).reset_index(drop=True)


@dataclass
class Comparacion:
    """Un par referencia/actual ya filtrado y muestreado, con su descripcion para el resumen."""

    nombre: str
    referencia: pd.DataFrame
    actual: pd.DataFrame
    info: dict


def armar_comparacion(
    p1: pd.DataFrame,
    nombre: str,
    especificacion: dict,
    conocidas: set[int],
    muestra: int = config.MONITOREO_MUESTRA,
    semilla: int = config.SEMILLA,
) -> Comparacion:
    """Filtra los periodos de una comparacion de ``config.MONITOREO_COMPARACIONES`` y muestrea.

    Con ``solo_conocidas`` se quitan de ambos conjuntos las estaciones nuevas. ``info`` guarda
    los tamanos y estaciones de lo que se compara (antes de muestrear) y, aparte, cuantas
    estaciones nuevas tiene el periodo actual y que fraccion de sus filas son de ellas (antes de
    quitarlas): el dato con que se explica buena parte del drift de la red.
    """
    referencia = p1[en_periodo(p1[COLUMNA_TIEMPO], especificacion["referencia"])]
    actual = p1[en_periodo(p1[COLUMNA_TIEMPO], especificacion["actual"])]
    if referencia.empty or actual.empty:
        raise ValueError(
            f"La comparacion {nombre} no tiene filas: referencia {len(referencia)}, actual "
            f"{len(actual)}. Revisa las fechas en config.py y que los datos cubran el periodo."
        )
    nuevas = ~actual[COLUMNA_ESTACION].isin(conocidas)
    info = {
        "nombre": nombre,
        "referencia": " a ".join(especificacion["referencia"]),
        "actual": " a ".join(especificacion["actual"]),
        "solo_conocidas": bool(especificacion["solo_conocidas"]),
        "estaciones_nuevas_periodo": int(actual.loc[nuevas, COLUMNA_ESTACION].nunique()),
        "fraccion_filas_nuevas_periodo": float(nuevas.mean()),
    }
    if especificacion["solo_conocidas"]:
        referencia = referencia[referencia[COLUMNA_ESTACION].isin(conocidas)]
        actual = actual[~nuevas]
    info["estaciones_referencia"] = int(referencia[COLUMNA_ESTACION].nunique())
    info["estaciones_actual"] = int(actual[COLUMNA_ESTACION].nunique())
    info["filas_referencia"] = len(referencia)
    info["filas_actual"] = len(actual)
    referencia = muestrear(referencia, muestra, semilla)
    actual = muestrear(actual, muestra, semilla)
    info["muestra_referencia"] = len(referencia)
    info["muestra_actual"] = len(actual)
    return Comparacion(nombre, referencia, actual, info)


# =============================================================================
# Preparacion para Evidently y estadisticas para interpretar
# =============================================================================
def para_evidently(df: pd.DataFrame) -> pd.DataFrame:
    """Solo las variables vigiladas, con tipos que Evidently acepta.

    Las categoricas pasan de ``int8`` a ``int64``: Evidently 0.7.23 no acepta ``int8`` como
    etiqueta de una categoria (falla el resumen de datos con "type numpy.int8 not supported as
    Label"). Las numericas pasan a ``float64``.
    """
    tipos = {columna: "int64" for columna in CATEGORICAS}
    tipos |= {columna: "float64" for columna in NUMERICAS}
    return df[VARIABLES].astype(tipos).reset_index(drop=True)


def _mayor_cambio_de_proporcion(referencia: pd.Series, actual: pd.Series) -> str:
    """Categoria cuya proporcion cambio mas, p. ej. ``9: 8.4% -> 33.1%``."""
    prop_ref = referencia.value_counts(normalize=True)
    prop_act = actual.value_counts(normalize=True)
    diferencia = prop_act.sub(prop_ref, fill_value=0.0)
    categoria = diferencia.abs().idxmax()
    return f"{categoria}: {prop_ref.get(categoria, 0.0):.1%} -> {prop_act.get(categoria, 0.0):.1%}"


def estadisticas(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Una fila por variable con lo necesario para explicar un drift en palabras.

    Numericas: media y mediana en cada conjunto, cambio relativo de la media y fraccion de
    ceros. Categoricas: la categoria cuya proporcion cambio mas (la media de ``mes`` no dice
    nada).
    """
    filas = []
    for columna in VARIABLES:
        ref = referencia[columna].astype(float)
        act = actual[columna].astype(float)
        fila = {
            "variable": columna,
            "tipo": "categorica" if columna in CATEGORICAS else "numerica",
        }
        if columna in CATEGORICAS:
            fila["mayor_cambio"] = _mayor_cambio_de_proporcion(
                referencia[columna], actual[columna]
            )
        else:
            media_ref, media_act = ref.mean(), act.mean()
            fila |= {
                "media_referencia": media_ref,
                "media_actual": media_act,
                "cambio_media": (media_act - media_ref) / media_ref if media_ref else np.nan,
                "mediana_referencia": ref.median(),
                "mediana_actual": act.median(),
                "ceros_referencia": float((ref == 0).mean()),
                "ceros_actual": float((act == 0).mean()),
            }
        filas.append(fila)
    return pd.DataFrame(filas)
