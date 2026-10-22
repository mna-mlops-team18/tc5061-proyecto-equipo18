"""Feature engineering para las tres preguntas del proyecto MiBici.

Uso (despues de la ingesta), como punto de entrada ``features`` del MLflow Project::

    mlflow run . -e features --env-manager local
    python -m tc5061_proyecto_equipo18.features      # equivalente, sin registrar en MLflow

Flujo::

    viajes limpios (Parquet mensual) + estaciones
        -> filtrar_viajes_validos
        -> contar_por_estacion_hora           malla estacion-hora completa, con ceros
        -> construir_dataset_p1 / p2 / p3     variables y objetivo de cada pregunta

Regla contra la fuga de informacion: toda variable de una fila con tiempo ``t`` se calcula
solo con datos disponibles en el momento de la prediccion. Los rezagos y las medias moviles
se desplazan al menos un horizonte completo, la informacion del mismo periodo que se predice
nunca se usa como variable, y los perfiles del clustering se calculan por periodo, de modo
que el modelo se ajusta solo con el periodo de entrenamiento.
"""

import json
import math
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.dataset import Manifiesto, meses_en_rango
from tc5061_proyecto_equipo18.seguimiento import registrar_etapa_datos

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

PARTICIONES = ("entrenamiento", "validacion", "prueba")


# =============================================================================
# Carga y limpieza
# =============================================================================
def cargar_viajes(
    mes_inicio: str = config.MES_INICIO,
    mes_fin: str = config.MES_FIN,
    columnas: tuple[str, ...] = (
        "viaje_id",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
        "duracion_seg",
    ),
) -> pd.DataFrame:
    """Lee los Parquet mensuales del rango y quita viajes repetidos entre meses.

    Solo lee las columnas necesarias para reducir memoria. Un viaje que cruza la medianoche
    del ultimo dia del mes puede aparecer en dos archivos; se conserva una sola vez.
    """
    archivos = [
        config.ruta_viajes_mes(mes)
        for mes in meses_en_rango(mes_inicio, mes_fin)
        if config.RUTAS_VIAJES.get(mes)
    ]
    faltantes = [archivo.name for archivo in archivos if not archivo.exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan meses procesados: {faltantes}. "
            "Ejecuta: python -m tc5061_proyecto_equipo18.dataset ingest"
        )
    viajes = pd.concat(
        [pd.read_parquet(archivo, columns=list(columnas)) for archivo in archivos],
        ignore_index=True,
    )
    if "viaje_id" in viajes.columns:
        viajes = viajes.drop_duplicates(subset="viaje_id", keep="first", ignore_index=True)
    return viajes


def cargar_estaciones() -> pd.DataFrame:
    """Lee la nomenclatura normalizada (id, nombre, coordenadas, capacidad, municipio)."""
    if not config.ESTACIONES_PARQUET.exists():
        raise FileNotFoundError(
            "Ejecuta primero: python -m tc5061_proyecto_equipo18.dataset ingest"
        )
    return pd.read_parquet(config.ESTACIONES_PARQUET)


def filtrar_viajes_validos(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    duracion_min_seg: int = config.DURACION_MIN_SEG,
    duracion_max_seg: int = config.DURACION_MAX_SEG,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Quita viajes que no representan demanda real y reporta cuantos quito por causa.

    Pertinencia (P1, P2 y P3):
    - Duracion menor a ``duracion_min_seg``: desanclajes fallidos o pruebas de la bicicleta;
      inflarian la demanda de la estacion sin que nadie haya viajado.
    - Duracion mayor a ``duracion_max_seg``: bicicletas mal ancladas o registros anomalos;
      su hora de llegada no refleja una devolucion real.
    - Estacion de origen o destino fuera de la nomenclatura: no hay capacidad ni ubicacion
      para usarlas en las variables, la etiqueta de riesgo ni el mapa.
    """
    conocidas = set(estaciones["estacion_id"].tolist())
    muy_corto = viajes["duracion_seg"] < duracion_min_seg
    muy_largo = viajes["duracion_seg"] > duracion_max_seg
    desconocida = ~viajes["origen_id"].isin(conocidas) | ~viajes["destino_id"].isin(conocidas)
    conservar = ~(muy_corto | muy_largo | desconocida)
    resumen = {
        "viajes_entrada": len(viajes),
        "descartados_duracion_corta": int(muy_corto.sum()),
        "descartados_duracion_larga": int((muy_largo & ~muy_corto).sum()),
        "descartados_estacion_desconocida": int((desconocida & ~muy_corto & ~muy_largo).sum()),
        "viajes_validos": int(conservar.sum()),
    }
    return viajes.loc[conservar].reset_index(drop=True), resumen


# =============================================================================
# Agregacion base: estacion x hora
# =============================================================================
def contar_por_estacion_hora(
    viajes: pd.DataFrame, mes_inicio: str = config.MES_INICIO, mes_fin: str = config.MES_FIN
) -> pd.DataFrame:
    """Cuenta salidas y llegadas por estacion y hora, con la malla horaria completa.

    - ``salidas``: viajes que inician en la estacion en esa hora (demanda, P1).
    - ``llegadas``: viajes que terminan en la estacion en esa hora (devoluciones).

    La malla incluye las horas sin viajes (ceros reales) desde el primer hasta el ultimo dia
    con actividad de cada estacion dentro del rango. Asi cada estacion es una serie regular
    de una fila por hora, condicion para que los rezagos de ``agregar_rezagos`` sean
    exactos (un desplazamiento de k filas = k horas).
    """
    inicio = pd.Period(mes_inicio, "M").start_time
    fin = pd.Period(mes_fin, "M").end_time.floor("h")

    salidas = viajes.groupby([viajes["origen_id"], viajes["inicio_viaje"].dt.floor("h")]).size()
    llegadas = viajes.groupby([viajes["destino_id"], viajes["fin_viaje"].dt.floor("h")]).size()
    salidas.index = salidas.index.set_names(["estacion_id", "hora"])
    llegadas.index = llegadas.index.set_names(["estacion_id", "hora"])
    conteos = pd.concat({"salidas": salidas, "llegadas": llegadas}, axis=1).fillna(0).reset_index()
    conteos = conteos[(conteos["hora"] >= inicio) & (conteos["hora"] <= fin)]

    ventanas = conteos.groupby("estacion_id")["hora"].agg(["min", "max"])
    mallas = [
        pd.DataFrame(
            {
                "estacion_id": estacion,
                "hora": pd.date_range(
                    desde.floor("D"), min(hasta.floor("D") + pd.Timedelta(hours=23), fin), freq="h"
                ),
            }
        )
        for estacion, (desde, hasta) in ventanas.iterrows()
    ]
    malla = pd.concat(mallas, ignore_index=True)
    malla["hora"] = malla["hora"].astype("datetime64[s]")
    conteos["hora"] = conteos["hora"].astype("datetime64[s]")
    resultado = malla.merge(conteos, on=["estacion_id", "hora"], how="left")
    resultado[["salidas", "llegadas"]] = resultado[["salidas", "llegadas"]].fillna(0)
    return resultado.astype(
        {"estacion_id": "int32", "salidas": "int32", "llegadas": "int32"}
    ).sort_values(["estacion_id", "hora"], ignore_index=True)


# =============================================================================
# Variables de calendario y de estacion
# =============================================================================
def _pascua(anio: int) -> pd.Timestamp:
    """Domingo de Pascua (algoritmo gregoriano anonimo)."""
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    mes = (h + m - 7 * n + 114) // 31
    dia = (h + m - 7 * n + 114) % 31 + 1
    return pd.Timestamp(year=anio, month=mes, day=dia)


def _n_lunes(anio: int, mes: int, n: int) -> pd.Timestamp:
    """n-esimo lunes de un mes."""
    primero = pd.Timestamp(year=anio, month=mes, day=1)
    return primero + pd.Timedelta(days=(7 - primero.dayofweek) % 7 + 7 * (n - 1))


def dias_festivos_mx(anios: list[int]) -> set[pd.Timestamp]:
    """Dias de descanso obligatorio en Mexico (LFT, art. 74) mas jueves y viernes santos.

    Pertinencia (P1, P2, P3): en dias festivos la demanda se parece mas a un fin de semana
    que a un dia laboral; sin esta variable el modelo confundiria un lunes festivo con un
    lunes normal. Jueves y viernes santos no son obligatorios por ley, pero en Guadalajara
    se comportan como festivos.
    """
    festivos = set()
    for anio in anios:
        festivos |= {
            pd.Timestamp(year=anio, month=1, day=1),
            _n_lunes(anio, 2, 1),  # Dia de la Constitucion
            _n_lunes(anio, 3, 3),  # Natalicio de Benito Juarez
            pd.Timestamp(year=anio, month=5, day=1),
            pd.Timestamp(year=anio, month=9, day=16),
            _n_lunes(anio, 11, 3),  # Dia de la Revolucion
            pd.Timestamp(year=anio, month=12, day=25),
            _pascua(anio) - pd.Timedelta(days=3),  # jueves santo
            _pascua(anio) - pd.Timedelta(days=2),  # viernes santo
        }
        if (anio - 2024) % 6 == 0:  # transmision del Poder Ejecutivo Federal
            festivos.add(pd.Timestamp(year=anio, month=10, day=1))
    return festivos


def agregar_calendario(df: pd.DataFrame, columna_tiempo: str) -> pd.DataFrame:
    """Agrega hora del dia, dia de la semana, fin de semana, festivo y mes.

    Pertinencia: la demanda de MiBici tiene ciclos diarios (picos de entrada y salida de
    trabajo y escuela) y semanales (laboral vs. fin de semana). Son variables conocidas de
    antemano para cualquier fecha futura, asi que no generan fuga de informacion.
    """
    tiempo = df[columna_tiempo]
    festivos = dias_festivos_mx(sorted(tiempo.dt.year.unique().tolist()))
    resultado = df.assign(
        hora_dia=tiempo.dt.hour.astype("int8"),
        dia_semana=tiempo.dt.dayofweek.astype("int8"),
        mes=tiempo.dt.month.astype("int8"),
    )
    resultado["es_fin_semana"] = (resultado["dia_semana"] >= 5).astype("int8")
    resultado["es_festivo"] = tiempo.dt.normalize().isin(festivos).astype("int8")
    return resultado


def agregar_atributos_estacion(df: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega capacidad y coordenadas de cada estacion.

    Pertinencia: la capacidad acota cuantas bicicletas puede prestar o recibir una estacion
    (central para P2) y la ubicacion resume el entorno urbano (oficinas, vivienda, centro).
    """
    atributos = estaciones[["estacion_id", "capacidad", "latitud", "longitud"]]
    return df.merge(atributos, on="estacion_id", how="left", validate="many_to_one")


# =============================================================================
# Rezagos y medias moviles (siempre hacia el pasado)
# =============================================================================
def validar_malla_regular(df: pd.DataFrame, columna_tiempo: str, paso: pd.Timedelta) -> None:
    """Comprueba que cada estacion sea una serie ordenada sin huecos de ``paso``.

    Los rezagos por posicion solo equivalen a rezagos en el tiempo si se cumple esto.

    Pertinencia (P1 y P2): un hueco en la serie haria que "hace 24 filas" no fuera "hace 24
    horas" y las variables de rezago mezclarian dias distintos sin avisar.
    """
    diferencias = df.groupby("estacion_id", sort=False)[columna_tiempo].diff().dropna()
    if not (diferencias == paso).all():
        raise ValueError(
            f"La serie no es regular cada {paso}; ordena por estacion y tiempo y completa la malla."
        )


def agregar_rezagos(
    df: pd.DataFrame, columna: str, pasos: tuple[int, ...], sufijo: str, grupo: str = "estacion_id"
) -> pd.DataFrame:
    """Agrega ``columna`` desplazada ``k`` pasos hacia el pasado dentro de cada estacion.

    En la fila con tiempo ``t``, ``{columna}_lag_{k}{sufijo}`` vale lo observado en ``t - k``.
    Requiere una malla regular ordenada (ver ``validar_malla_regular``).

    Pertinencia (P1 y P2): la demanda y el flujo neto se repiten por dia y por semana, asi
    que lo observado hace ``k`` pasos es el mejor indicio simple de lo que viene; usar solo
    valores pasados evita la fuga de informacion.
    """
    agrupado = df.groupby(grupo, sort=False)[columna]
    nuevas = {f"{columna}_lag_{k}{sufijo}": agrupado.shift(k).astype("float32") for k in pasos}
    return df.assign(**nuevas)


def media_movil_previa(
    df: pd.DataFrame, columna: str, ventana: int, desfase: int, grupos: list[str]
) -> pd.Series:
    """Media de los ``ventana`` valores que terminan ``desfase`` pasos antes de cada fila.

    Con ``desfase >= 1`` el valor de la propia fila nunca entra en el promedio. Si ``grupos``
    incluye la hora del dia, los pasos son dias a la misma hora (estacionalidad diaria).

    Pertinencia (P1 y P2): un promedio de varios dias es menos ruidoso que un solo rezago y
    resume el nivel reciente de la estacion (cierres temporales, temporada, tendencia).
    """
    return (
        df.groupby(grupos, sort=False)[columna]
        .transform(lambda serie: serie.shift(desfase).rolling(ventana, min_periods=ventana).mean())
        .astype("float32")
    )


# =============================================================================
# P1 - Regresion: salidas por estacion y hora, un dia antes
# =============================================================================
COLUMNAS_P1 = [
    "hora_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "capacidad", "latitud", "longitud",
    "salidas_lag_24h", "salidas_lag_48h", "salidas_lag_168h", "salidas_lag_336h",
    "llegadas_lag_24h", "salidas_media_misma_hora_7d", "salidas_media_24h_previa",
    "salidas_media_7d",
]  # fmt: skip
OBJETIVO_P1 = "salidas"


def construir_dataset_p1(
    conteos: pd.DataFrame,
    estaciones: pd.DataFrame,
    horizonte_h: int = config.P1_HORIZONTE_H,
    rezagos_h: tuple[int, ...] = config.P1_REZAGOS_H,
) -> pd.DataFrame:
    """Tabla para P1: una fila por estacion y hora con el objetivo ``salidas``.

    Pertinencia: el operador planea la flota del dia siguiente al cierre del dia anterior,
    asi que todas las variables usan datos con al menos ``horizonte_h`` horas de antiguedad:
    - rezagos de 1, 2, 7 y 14 dias a la misma hora (patron diario y semanal),
    - llegadas de ayer a la misma hora (bicicletas que suelen quedar disponibles),
    - medias de la misma hora en los ultimos 7 dias, del dia previo y de la semana previa
      (nivel reciente de la estacion),
    - calendario y atributos de la estacion.
    Las llegadas de la misma hora no se usan: no se conocen al momento de predecir.
    Se modelan las estaciones vigentes de la nomenclatura de sep-2026 (``main`` pasa solo
    esas y aqui se exige capacidad > 0); asi la capacidad nunca delata un retiro futuro.
    """
    if min(rezagos_h) < horizonte_h:
        raise ValueError("Un rezago menor que el horizonte usaria informacion del futuro")
    df = conteos[["estacion_id", "hora", "salidas", "llegadas"]].sort_values(
        ["estacion_id", "hora"], ignore_index=True
    )
    validar_malla_regular(df, "hora", pd.Timedelta(hours=1))
    df = agregar_calendario(df, "hora")
    df = agregar_rezagos(df, "salidas", tuple(rezagos_h), sufijo="h")
    df = agregar_rezagos(df, "llegadas", (horizonte_h,), sufijo="h")
    dias_desfase = math.ceil(horizonte_h / 24)
    df["salidas_media_misma_hora_7d"] = media_movil_previa(
        df, "salidas", ventana=7, desfase=dias_desfase, grupos=["estacion_id", "hora_dia"]
    )
    df["salidas_media_24h_previa"] = media_movil_previa(
        df, "salidas", ventana=24, desfase=horizonte_h, grupos=["estacion_id"]
    )
    df["salidas_media_7d"] = media_movil_previa(
        df, "salidas", ventana=168, desfase=horizonte_h, grupos=["estacion_id"]
    )
    df = agregar_atributos_estacion(df, estaciones)
    df = df[df["capacidad"] > 0]
    df = df.dropna(subset=COLUMNAS_P1).reset_index(drop=True)
    return df[["estacion_id", "hora", *COLUMNAS_P1, OBJETIVO_P1]]


# =============================================================================
# P2 - Clasificacion: riesgo de vaciarse o llenarse en el siguiente bloque
# =============================================================================
COLUMNAS_P2 = [
    "bloque_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "capacidad", "latitud", "longitud",
    "salidas_lag_1b", "llegadas_lag_1b", "neto_lag_1b", "neto_lag_dia", "neto_lag_semana",
    "neto_media_mismo_bloque_7d", "neto_rel_media_mismo_bloque_7d", "neto_rel_lag_dia",
    "frac_vaciado_mismo_bloque_7d", "frac_llenado_mismo_bloque_7d", "salidas_previas_24h",
]  # fmt: skip
OBJETIVO_P2 = "clase_riesgo"


def agregar_bloques(
    conteos: pd.DataFrame, horas_bloque: int = config.P2_HORAS_BLOQUE
) -> pd.DataFrame:
    """Suma salidas y llegadas por estacion en bloques de ``horas_bloque`` horas.

    ``neto = salidas - llegadas``: positivo si la estacion pierde bicicletas en el bloque.
    Pertinencia (P2): el equipo de rebalanceo programa sus rutas por turnos de algunas horas,
    no hora por hora; el bloque es la unidad de decision.
    """
    if 24 % horas_bloque:
        raise ValueError("horas_bloque debe dividir 24")
    bloques = (
        conteos.assign(bloque_inicio=conteos["hora"].dt.floor(f"{horas_bloque}h"))
        .groupby(["estacion_id", "bloque_inicio"], sort=True)[["salidas", "llegadas"]]
        .sum()
        .reset_index()
    )
    bloques["neto"] = bloques["salidas"] - bloques["llegadas"]
    return bloques


def etiquetar_riesgo(
    neto: pd.Series, capacidad: pd.Series, umbral: float = config.P2_UMBRAL_RIESGO
) -> pd.Series:
    """Aproxima el riesgo de desabasto con viajes y capacidad (no hay inventario publicado).

    Supuesto: tras el rebalanceo, una estacion arranca el bloque cerca de la mitad de su
    capacidad. Si en el bloque pierde al menos ``umbral x capacidad`` bicicletas netas, se
    vaciaria (clase 1, falta de bicicletas); si gana esa cantidad, se llenaria (clase 2,
    falta de anclajes). En otro caso, clase 0.

    Limitacion: los viajes observados estan censurados (en una estacion vacia nadie puede
    sacar bicicleta), por lo que la etiqueta tiende a subestimar el riesgo real.

    Pertinencia (P2): es la variable objetivo de la clasificacion. Distingue la falta de
    bicicletas (hay que llevar) de la falta de anclajes (hay que retirar), que son decisiones
    distintas para el equipo de rebalanceo.
    """
    limite = umbral * capacidad
    clase = np.select([neto >= limite, -neto >= limite], [1, 2], default=0)
    return pd.Series(clase, index=neto.index, name=OBJETIVO_P2).astype("int8")


def construir_dataset_p2(
    conteos: pd.DataFrame,
    estaciones: pd.DataFrame,
    horas_bloque: int = config.P2_HORAS_BLOQUE,
    umbral: float = config.P2_UMBRAL_RIESGO,
) -> pd.DataFrame:
    """Tabla para P2: una fila por estacion y bloque con ``clase_riesgo`` (0, 1 o 2).

    La prediccion se hace al inicio del bloque, asi que las variables usan solo bloques ya
    cerrados: el bloque anterior, el mismo bloque de ayer y de hace una semana, y la
    frecuencia con la que ese bloque estuvo en riesgo en los ultimos 7 dias. El flujo del
    bloque que se predice no se usa como variable. Las versiones relativas (``*_rel_*``)
    dividen entre la capacidad, porque la etiqueta depende de esa proporcion.
    """
    pasos_dia = 24 // horas_bloque
    bloques = agregar_bloques(conteos, horas_bloque)
    validar_malla_regular(bloques, "bloque_inicio", pd.Timedelta(hours=horas_bloque))
    bloques = agregar_atributos_estacion(bloques, estaciones)
    bloques = bloques[bloques["capacidad"] > 0].reset_index(drop=True)
    bloques[OBJETIVO_P2] = etiquetar_riesgo(bloques["neto"], bloques["capacidad"], umbral)

    bloques = agregar_calendario(bloques, "bloque_inicio").rename(
        columns={"hora_dia": "bloque_dia"}
    )
    bloques = agregar_rezagos(bloques, "salidas", (1,), sufijo="b")
    bloques = agregar_rezagos(bloques, "llegadas", (1,), sufijo="b")
    bloques = agregar_rezagos(bloques, "neto", (1,), sufijo="b")
    agrupado = bloques.groupby("estacion_id", sort=False)["neto"]
    bloques["neto_lag_dia"] = agrupado.shift(pasos_dia).astype("float32")
    bloques["neto_lag_semana"] = agrupado.shift(7 * pasos_dia).astype("float32")

    mismo_bloque = ["estacion_id", "bloque_dia"]
    bloques["neto_media_mismo_bloque_7d"] = media_movil_previa(bloques, "neto", 7, 1, mismo_bloque)
    bloques["neto_rel_media_mismo_bloque_7d"] = (
        bloques["neto_media_mismo_bloque_7d"] / bloques["capacidad"]
    ).astype("float32")
    bloques["neto_rel_lag_dia"] = (bloques["neto_lag_dia"] / bloques["capacidad"]).astype(
        "float32"
    )
    bloques["es_vaciado"] = (bloques[OBJETIVO_P2] == 1).astype("int8")
    bloques["es_llenado"] = (bloques[OBJETIVO_P2] == 2).astype("int8")
    bloques["frac_vaciado_mismo_bloque_7d"] = media_movil_previa(
        bloques, "es_vaciado", 7, 1, mismo_bloque
    )
    bloques["frac_llenado_mismo_bloque_7d"] = media_movil_previa(
        bloques, "es_llenado", 7, 1, mismo_bloque
    )
    bloques["salidas_previas_24h"] = (
        media_movil_previa(bloques, "salidas", pasos_dia, 1, ["estacion_id"]) * pasos_dia
    )

    bloques = bloques.dropna(subset=COLUMNAS_P2).reset_index(drop=True)
    return bloques[["estacion_id", "bloque_inicio", *COLUMNAS_P2, OBJETIVO_P2]]


# =============================================================================
# P3 - Clustering: perfiles de uso por estacion
# =============================================================================
def perfiles_estacion(
    conteos: pd.DataFrame,
    desde: pd.Timestamp,
    hasta: pd.Timestamp,
    min_viajes: int = config.P3_MIN_VIAJES,
) -> pd.DataFrame:
    """Perfil de uso de cada estacion en ``[desde, hasta)``.

    Variables (una fila por estacion):
    - ``sal_lab_hHH`` / ``lle_lab_hHH``: fraccion de las salidas y llegadas de dias laborales
      que ocurre en cada hora (24 + 24 valores que suman 1 cada bloque). Distingue estaciones
      "residenciales" (salen en la manana) de "laborales" (llegan en la manana).
    - ``sal_fds_hHH`` / ``lle_fds_hHH``: lo mismo en fines de semana y festivos (uso recreativo).
    - ``log_viajes_dia``: volumen diario (escala log) para separar estaciones muy y poco usadas.
    - ``ratio_fin_semana``: viajes diarios en fin de semana / en dias laborales.
    - ``balance_neto``: (llegadas - salidas) / total; estaciones que se llenan o se vacian.
    Se excluyen estaciones con menos de ``min_viajes`` viajes en el periodo (perfil ruidoso).

    Pertinencia (P3): los grupos deben reunir estaciones que se usan igual, sin importar
    cuantos viajes tienen; por eso el perfil horario va en proporciones y el volumen va
    aparte. La ubicacion no entra: los grupos se forman por uso y luego se ubican en el mapa.
    """
    periodo = conteos[(conteos["hora"] >= desde) & (conteos["hora"] < hasta)]
    periodo = agregar_calendario(periodo, "hora")
    no_laboral = (periodo["es_fin_semana"] == 1) | (periodo["es_festivo"] == 1)
    periodo = periodo.assign(
        tipo_dia=np.where(no_laboral, "fds", "lab"), dia=periodo["hora"].dt.normalize()
    )

    por_hora = periodo.groupby(["estacion_id", "tipo_dia", "hora_dia"])[
        ["salidas", "llegadas"]
    ].sum()
    totales = (
        por_hora.groupby(level=["estacion_id", "tipo_dia"]).transform("sum").replace(0, np.nan)
    )
    fracciones = (por_hora / totales).fillna(0).unstack(["tipo_dia", "hora_dia"], fill_value=0)
    fracciones.columns = [
        f"{'sal' if medida == 'salidas' else 'lle'}_{tipo}_h{hora:02d}"
        for medida, tipo, hora in fracciones.columns
    ]
    columnas_perfil = [
        f"{medida}_{tipo}_h{hora:02d}"
        for medida in ("sal", "lle") for tipo in ("lab", "fds") for hora in range(24)
    ]  # fmt: skip
    fracciones = fracciones.reindex(columns=columnas_perfil, fill_value=0.0)

    diario = periodo.groupby(["estacion_id", "tipo_dia", "dia"])[["salidas", "llegadas"]].sum()
    viajes_dia = (
        (diario["salidas"] + diario["llegadas"]).groupby(level=["estacion_id", "tipo_dia"]).mean()
    )
    viajes_dia = viajes_dia.unstack("tipo_dia").reindex(columns=["lab", "fds"]).fillna(0)
    totales_estacion = periodo.groupby("estacion_id")[["salidas", "llegadas"]].sum()
    total = totales_estacion["salidas"] + totales_estacion["llegadas"]

    resumen = pd.DataFrame(
        {
            "viajes_periodo": total,
            "log_viajes_dia": np.log1p(
                diario[["salidas", "llegadas"]].sum(axis=1).groupby(level="estacion_id").mean()
            ),
            "ratio_fin_semana": (viajes_dia["fds"] / viajes_dia["lab"].replace(0, np.nan)).fillna(
                0
            ),
            "balance_neto": (totales_estacion["llegadas"] - totales_estacion["salidas"])
            / total.replace(0, np.nan),
        }
    ).fillna(0)
    perfiles = resumen.join(fracciones, how="inner")
    perfiles = perfiles[perfiles["viajes_periodo"] >= min_viajes]
    return perfiles.reset_index().astype({"estacion_id": "int32"}).copy()


def construir_dataset_p3(
    conteos: pd.DataFrame,
    estaciones: pd.DataFrame,
    fin_entrenamiento: str = config.FIN_ENTRENAMIENTO,
    fin_validacion: str = config.FIN_VALIDACION,
    min_viajes: int = config.P3_MIN_VIAJES,
) -> pd.DataFrame:
    """Perfiles de P3 calculados por separado para entrenamiento, validacion y prueba.

    El modelo de grupos se ajusta solo con los perfiles de entrenamiento; los de validacion y
    prueba sirven para medir si una estacion conserva su grupo en periodos posteriores
    (estabilidad), sin que el ajuste vea esos datos.
    """
    limites = {
        "entrenamiento": (
            conteos["hora"].min(),
            pd.Timestamp(fin_entrenamiento) + pd.Timedelta(days=1),
        ),
        "validacion": (
            pd.Timestamp(fin_entrenamiento) + pd.Timedelta(days=1),
            pd.Timestamp(fin_validacion) + pd.Timedelta(days=1),
        ),
        "prueba": (
            pd.Timestamp(fin_validacion) + pd.Timedelta(days=1),
            conteos["hora"].max() + pd.Timedelta(hours=1),
        ),
    }
    partes = [
        perfiles_estacion(conteos, desde, hasta, min_viajes).assign(periodo=nombre)
        for nombre, (desde, hasta) in limites.items()
    ]
    perfiles = pd.concat(partes, ignore_index=True)
    atributos = estaciones[
        ["estacion_id", "codigo", "nombre", "municipio", "capacidad", "latitud", "longitud"]
    ]
    perfiles = perfiles.merge(atributos, on="estacion_id", how="left", validate="many_to_one")
    # Solo estaciones vigentes (capacidad > 0), como en P1 y P2: P3 sirve para planear la red actual
    return perfiles[perfiles["capacidad"] > 0].reset_index(drop=True)


# =============================================================================
# Particion temporal
# =============================================================================
def asignar_particion(
    tiempos: pd.Series,
    fin_entrenamiento: str = config.FIN_ENTRENAMIENTO,
    fin_validacion: str = config.FIN_VALIDACION,
) -> pd.Series:
    """Etiqueta cada fila como entrenamiento, validacion o prueba segun su fecha.

    La particion es cronologica (nunca aleatoria): el modelo se entrena con el pasado y se
    evalua con meses posteriores, como ocurriria en operacion.

    Pertinencia (P1, P2 y P3): con una particion aleatoria, horas vecinas del mismo dia
    quedarian en entrenamiento y prueba a la vez y el error medido seria optimista.
    """
    corte_entrenamiento = pd.Timestamp(fin_entrenamiento) + pd.Timedelta(days=1)
    corte_validacion = pd.Timestamp(fin_validacion) + pd.Timedelta(days=1)
    etiquetas = np.select(
        [tiempos < corte_entrenamiento, tiempos < corte_validacion],
        ["entrenamiento", "validacion"],
        default="prueba",
    )
    return pd.Series(etiquetas, index=tiempos.index, name="particion")


def parametros_features() -> dict:
    """Parametros que determinan los conjuntos de datos (se guardan y se registran en MLflow)."""
    return {
        "mes_inicio": config.MES_INICIO,
        "mes_fin": config.MES_FIN,
        "duracion_min_seg": config.DURACION_MIN_SEG,
        "duracion_max_seg": config.DURACION_MAX_SEG,
        "p1_horizonte_h": config.P1_HORIZONTE_H,
        "p1_rezagos_h": list(config.P1_REZAGOS_H),
        "p2_horas_bloque": config.P2_HORAS_BLOQUE,
        "p2_umbral_riesgo": config.P2_UMBRAL_RIESGO,
        "p3_min_viajes": config.P3_MIN_VIAJES,
        "fin_entrenamiento": config.FIN_ENTRENAMIENTO,
        "fin_validacion": config.FIN_VALIDACION,
    }


def cargar_metadatos_features() -> dict:
    """Lee los metadatos de la ultima construccion de features y valida que esten vigentes."""
    if not config.FEATURES_META.exists():
        raise FileNotFoundError("Ejecuta primero: python -m tc5061_proyecto_equipo18.features")
    meta = json.loads(config.FEATURES_META.read_text(encoding="utf-8"))
    version_actual = Manifiesto.cargar().version
    if meta["data_version"] != version_actual:
        raise RuntimeError(
            f"Las features se construyeron con la version de datos {meta['data_version']} y el "
            f"manifiesto actual es {version_actual}. Vuelve a correr features."
        )
    if meta["parametros"] != parametros_features():
        raise RuntimeError(
            "config.py cambio desde la ultima construccion de features; vuelve a correrla."
        )
    return meta


@app.command()
def main(
    salida_meta: Path = config.FEATURES_META,
):
    """Construye los conjuntos de datos de P1, P2 y P3 a partir de los datos ingeridos."""
    manifiesto = Manifiesto.cargar()
    if (manifiesto.mes_inicio, manifiesto.mes_fin) != (config.MES_INICIO, config.MES_FIN):
        # Deben coincidir: el rango que se registra en MLflow sale del manifiesto
        raise typer.BadParameter(
            f"El manifiesto cubre {manifiesto.mes_inicio}..{manifiesto.mes_fin} y config.py pide "
            f"{config.MES_INICIO}..{config.MES_FIN}. Corre: python -m "
            "tc5061_proyecto_equipo18.dataset ingest --actualizar-manifiesto"
        )
    logger.info(f"Construyendo features con datos version {manifiesto.version}")

    estaciones = cargar_estaciones()
    viajes, filtros = filtrar_viajes_validos(cargar_viajes(), estaciones)
    # Se modelan solo las estaciones vigentes: sin prefijo "ex-" y con capacidad > 0
    vigentes = estaciones[~estaciones["retirada"]]
    filtros["estaciones_vigentes"] = len(vigentes)
    logger.info(f"Filtros de limpieza: {filtros}")

    conteos = contar_por_estacion_hora(viajes)
    del viajes
    config.CONTEOS_HORA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    conteos.to_parquet(config.CONTEOS_HORA_PARQUET, index=False, compression="zstd")
    logger.info(
        f"Malla estacion-hora: {len(conteos):,} filas, {conteos['estacion_id'].nunique()} estaciones"
    )

    p1 = construir_dataset_p1(conteos, vigentes)
    p1.to_parquet(config.DATASET_P1, index=False, compression="zstd")
    logger.info(f"P1: {len(p1):,} filas")

    p2 = construir_dataset_p2(conteos, vigentes)
    p2.to_parquet(config.DATASET_P2, index=False, compression="zstd")
    distribucion = p2[OBJETIVO_P2].value_counts(normalize=True).sort_index().round(4).to_dict()
    logger.info(f"P2: {len(p2):,} filas; distribucion de clases {distribucion}")

    p3 = construir_dataset_p3(conteos, vigentes)
    p3.to_parquet(config.DATASET_P3, index=False, compression="zstd")
    logger.info(f"P3: {p3.groupby('periodo').size().to_dict()} estaciones por periodo")

    meta = {
        "data_version": manifiesto.version,
        "parametros": parametros_features(),
        "filtros": filtros,
        "filas": {
            "conteos_estacion_hora": len(conteos),
            "p1": len(p1),
            "p2": len(p2),
            "p3": {periodo: int(n) for periodo, n in p3.groupby("periodo").size().items()},
        },
        "p2_distribucion_clases": {str(k): float(v) for k, v in distribucion.items()},
    }
    salida_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.success(
        f"Features listas (datos {manifiesto.version}). Metadatos en {salida_meta.name}"
    )
    # Con `mlflow run . -e features`, deja los metadatos y los tamanios en ese run
    registrar_etapa_datos(
        "datos-features",
        manifiesto.version,
        parametros={clave: str(valor) for clave, valor in parametros_features().items()},
        metricas={
            "viajes_validos": filtros["viajes_validos"],
            "estaciones_vigentes": filtros["estaciones_vigentes"],
            "filas_p1": len(p1),
            "filas_p2": len(p2),
            **{f"perfiles_p3_{periodo}": n for periodo, n in meta["filas"]["p3"].items()},
        },
        diccionarios={"features_meta.json": meta},
    )


if __name__ == "__main__":
    app()
