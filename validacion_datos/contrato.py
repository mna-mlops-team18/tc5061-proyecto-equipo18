"""Contrato de la validacion de datos: que se valida, con que columnas y como se parte.

Solo usa pandas y ``config``; NO importa DeepChecks. Asi el ambiente principal (Python 3.14)
puede importarlo y comprobar que sigue igual que ``features.py`` (ver
``tests/test_contrato_validacion.py``). No importa ``features.py`` porque arrastra mlflow, typer
y dataset, que no estan en el ambiente de validacion (Python 3.12, ver README).

Debe correrse desde la raiz del repo: el paquete no se instala en el ambiente de validacion.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tc5061_proyecto_equipo18 import config

MUESTRA_POR_DEFECTO = 200_000
PARTICION_ENTRENAMIENTO = "entrenamiento"
PARTICION_PRUEBA = "prueba"

# Copia de features.py (COLUMNAS_P1/P2 y OBJETIVO_P1/P2); el test de guarda las compara.
COLUMNAS_P1 = [
    "hora_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "capacidad", "latitud", "longitud",
    "salidas_lag_24h", "salidas_lag_48h", "salidas_lag_168h", "salidas_lag_336h",
    "llegadas_lag_24h", "salidas_media_misma_hora_7d", "salidas_media_24h_previa",
    "salidas_media_7d",
]  # fmt: skip
OBJETIVO_P1 = "salidas"
COLUMNAS_P2 = [
    "bloque_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "capacidad", "latitud", "longitud",
    "salidas_lag_1b", "llegadas_lag_1b", "neto_lag_1b", "neto_lag_dia", "neto_lag_semana",
    "neto_media_mismo_bloque_7d", "neto_rel_media_mismo_bloque_7d", "neto_rel_lag_dia",
    "frac_vaciado_mismo_bloque_7d", "frac_llenado_mismo_bloque_7d", "salidas_previas_24h",
]  # fmt: skip
OBJETIVO_P2 = "clase_riesgo"

# Variables categoricas declaradas a mano (DeepChecks las adivinaria si no). estacion_id no es
# entrada del modelo; se declara categorica para que NewCategoryTrainTest muestre las
# estaciones que aparecen solo despues del corte.
CATEGORICAS_P1 = [
    "hora_dia", "dia_semana", "mes", "es_fin_semana", "es_festivo", "estacion_id",
]  # fmt: skip
CATEGORICAS_P2 = [
    "bloque_dia", "dia_semana", "mes", "es_fin_semana", "es_festivo", "estacion_id",
]  # fmt: skip

# P3: perfiles horarios sal_* / lle_* (laboral y fin de semana) mas el resumen por estacion,
# igual que modeling/p3_clustering.columnas_variables
VARIABLES_RESUMEN_P3 = ["log_viajes_dia", "ratio_fin_semana", "balance_neto"]
PERFILES_P3 = [
    f"{tipo}_{dia}_h{hora:02d}"
    for tipo in ("sal", "lle")
    for dia in ("lab", "fds")
    for hora in range(24)
]


@dataclass(frozen=True)
class Especificacion:
    """Todo lo que DeepChecks necesita saber de un conjunto de datos, declarado a mano.

    ``categoricas`` se declara siempre: sin ello DeepChecks adivina que columnas lo son, y una
    suposicion no es un contrato.
    """

    pregunta: str
    ruta: Path
    caracteristicas: tuple[str, ...]
    categoricas: tuple[str, ...]
    etiqueta: str | None  # None: sin etiqueta (clustering)
    tarea: str | None  # "regression", "multiclass" o None
    columna_tiempo: str | None  # None: la particion viene de ``columna_particion``
    columna_particion: str | None
    llave: tuple[str, ...]  # combinacion que debe ser unica en todo el archivo


ESPECIFICACIONES = {
    "p1": Especificacion(
        pregunta="p1",
        ruta=config.DATASET_P1,
        caracteristicas=(*COLUMNAS_P1, "estacion_id"),
        categoricas=tuple(CATEGORICAS_P1),
        etiqueta=OBJETIVO_P1,
        tarea="regression",
        columna_tiempo="hora",
        columna_particion=None,
        llave=("estacion_id", "hora"),
    ),
    "p2": Especificacion(
        pregunta="p2",
        ruta=config.DATASET_P2,
        caracteristicas=(*COLUMNAS_P2, "estacion_id"),
        categoricas=tuple(CATEGORICAS_P2),
        etiqueta=OBJETIVO_P2,
        tarea="multiclass",
        columna_tiempo="bloque_inicio",
        columna_particion=None,
        llave=("estacion_id", "bloque_inicio"),
    ),
    "p3": Especificacion(
        pregunta="p3",
        ruta=config.DATASET_P3,
        caracteristicas=(*PERFILES_P3, *VARIABLES_RESUMEN_P3),
        categoricas=(),
        etiqueta=None,
        tarea=None,
        columna_tiempo=None,
        columna_particion="periodo",
        llave=("estacion_id", "periodo"),
    ),
}


def asignar_particion(
    tiempos: pd.Series,
    fin_entrenamiento: str = config.FIN_ENTRENAMIENTO,
    fin_validacion: str = config.FIN_VALIDACION,
) -> pd.Series:
    """Copia de ``features.asignar_particion`` (cronologica, nunca aleatoria)."""
    corte_entrenamiento = pd.Timestamp(fin_entrenamiento) + pd.Timedelta(days=1)
    corte_validacion = pd.Timestamp(fin_validacion) + pd.Timedelta(days=1)
    etiquetas = np.select(
        [tiempos < corte_entrenamiento, tiempos < corte_validacion],
        ["entrenamiento", "validacion"],
        default="prueba",
    )
    return pd.Series(etiquetas, index=tiempos.index, name="particion")


def estado_datos() -> str:
    """``"faltan"``, ``"desactualizados"`` u ``"ok"``.

    Faltar datos es un estado de preparacion, no un defecto; que no coincidan con el manifiesto
    si es un problema (misma regla que ``features.cargar_metadatos_features``).
    """
    requeridos = [esp.ruta for esp in ESPECIFICACIONES.values()]
    requeridos += [config.FEATURES_META, config.MANIFIESTO_DATOS]
    if not all(ruta.exists() for ruta in requeridos):
        return "faltan"
    meta = json.loads(config.FEATURES_META.read_text(encoding="utf-8"))
    if meta["data_version"] != version_datos():
        return "desactualizados"
    return "ok"


def version_datos() -> str:
    """``data_version`` del manifiesto (para el resumen)."""
    manifiesto = json.loads(config.MANIFIESTO_DATOS.read_text(encoding="utf-8"))
    return manifiesto["version"]


def revisar_llave(pregunta: str) -> tuple[int, int]:
    """``(nulos, repetidas)`` de la llave en TODO el archivo, no en la muestra.

    DataDuplicates de DeepChecks corre sobre la muestra y puede ocultar duplicados reales;
    esta revision es barata porque solo lee las columnas de la llave. ``nulos`` cuenta las
    celdas vacias de la llave y ``repetidas`` las filas cuya llave ya aparecio antes.
    """
    esp = ESPECIFICACIONES[pregunta]
    llave = pd.read_parquet(esp.ruta, columns=list(esp.llave))
    return int(llave.isna().sum().sum()), int(llave.duplicated().sum())


def cargar(pregunta: str, muestra: int = MUESTRA_POR_DEFECTO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve ``(entrenamiento, prueba)`` de una pregunta, con las columnas del contrato.

    Lee solo las columnas necesarias y muestrea cada particion por separado con
    ``config.SEMILLA`` (P3 es pequeno y no se muestrea). El tamano completo de cada particion
    queda en ``df.attrs["filas_particion"]`` para anotarlo en el reporte.
    """
    esp = ESPECIFICACIONES[pregunta]
    columnas = list(esp.caracteristicas)
    for extra in (esp.etiqueta, esp.columna_tiempo, esp.columna_particion):
        if extra:
            columnas.append(extra)
    df = pd.read_parquet(esp.ruta, columns=columnas)

    if esp.columna_tiempo:
        particion = asignar_particion(df[esp.columna_tiempo])
    else:
        particion = df[esp.columna_particion]

    partes = []
    for nombre in (PARTICION_ENTRENAMIENTO, PARTICION_PRUEBA):
        parte = df[particion == nombre]
        total = len(parte)
        if esp.columna_tiempo and total > muestra:
            parte = parte.sample(n=muestra, random_state=config.SEMILLA)
        if esp.columna_tiempo:
            parte = parte.sort_values(esp.columna_tiempo)
        if esp.columna_particion:
            parte = parte.drop(columns=esp.columna_particion)
        parte = parte.reset_index(drop=True)
        parte.attrs["filas_particion"] = total
        partes.append(parte)
    return partes[0], partes[1]
