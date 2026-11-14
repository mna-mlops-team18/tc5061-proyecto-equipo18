"""Reporte de drift de P1 con Evidently: datos de entrenamiento contra datos de produccion.

Corre en el ambiente ``.venv-monitoreo`` (Python 3.13), desde la raiz del repositorio::

    make drift                                          # Linux y macOS
    .venv-monitoreo\\Scripts\\python -m monitoreo.drift  # Windows (PowerShell)

Por que un ambiente aparte: Evidently 0.7.23 (la version actual) usa ``pydantic.v1`` por dentro
y Pydantic avisa que esa parte no funciona en Python 3.14, el Python del proyecto; Evidently
solo se prueba hasta 3.13. Como el reporte de drift solo lee los Parquet (no necesita MLflow ni
los modelos), vive aparte, igual que la validacion con DeepChecks. Ver README > Monitoreo.

Que hace, para cada comparacion de ``config.MONITOREO_COMPARACIONES``:

1. Arma la referencia y el conjunto actual (``monitoreo/conjuntos.py``) y toma una muestra de
   ``config.MONITOREO_MUESTRA`` filas de cada uno, con la semilla del proyecto.
2. Corre un ``Report`` de Evidently con ``DataDriftPreset`` (drift por variable) y
   ``DataSummaryPreset`` (calidad de los datos), con las variables declaradas a mano
   (numericas y categoricas) y el metodo y umbral de ``config.py``.
3. Guarda el reporte HTML de Evidently y sus metricas en JSON.

Al final escribe ``reports/monitoreo/drift_puntajes.csv``, ``reports/monitoreo/resumen_drift.md``
(tablas para el reporte) y dos figuras en ``reports/figures/monitoreo/``.
"""

import argparse
import json
from pathlib import Path
import unicodedata

import evidently
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitoreo import conjuntos
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.plots import SERIES, TINTA_SECUNDARIA, _titulo, guardar_figura
from validacion_datos import contrato

ORDEN_COMPARACIONES = list(config.MONITOREO_COMPARACIONES)
ETIQUETAS = {
    "principal": "Entrenamiento vs. jul-sep 2026",
    "sin_nuevas": "Igual, sin estaciones nuevas",
    "misma_temporada": "Jul-sep 2025 vs. jul-sep 2026 (sin nuevas)",
}
# Metricas de calidad del conjunto actual que se copian al resumen (DataSummaryPreset)
CALIDAD = {
    "DatasetMissingValueCount": "Valores faltantes",
    "EmptyRowsCount": "Filas vacias",
    "EmptyColumnsCount": "Columnas vacias",
    "ConstantColumnsCount": "Columnas constantes",
    "AlmostConstantColumnsCount": "Columnas casi constantes",
    "DuplicatedColumnsCount": "Columnas duplicadas",
    "DuplicatedRowCount": "Filas con valores repetidos",
}
PRUEBAS_DE_DRIFT = {"ValueDrift", "DriftedColumnsCount"}
MAX_PRUEBAS_EN_RESUMEN = 30


# =============================================================================
# Evidently
# =============================================================================
def definicion_datos() -> DataDefinition:
    """Tipos declarados a mano: Evidently no tiene que adivinar que columnas son categoricas."""
    return DataDefinition(
        numerical_columns=list(conjuntos.NUMERICAS),
        categorical_columns=list(conjuntos.CATEGORICAS),
    )


def construir_reporte(
    umbral: float = config.MONITOREO_UMBRAL, drift_share: float = config.MONITOREO_DRIFT_SHARE
) -> Report:
    """Drift por variable (con el metodo y umbral de config.py) y resumen de calidad.

    ``include_tests=True`` agrega las pruebas automaticas de Evidently (por ejemplo, que la media
    del conjunto actual quede a +-10 % de la referencia); salen en la pestana de pruebas del HTML
    y las que fallan se listan en el resumen.
    """
    return Report(
        [
            DataDriftPreset(
                columns=list(conjuntos.VARIABLES),
                num_method=config.MONITOREO_METODO_NUMERICAS,
                cat_method=config.MONITOREO_METODO_CATEGORICAS,
                threshold=umbral,
                drift_share=drift_share,
            ),
            DataSummaryPreset(columns=list(conjuntos.VARIABLES)),
        ],
        include_tests=True,
    )


def correr(comparacion: conjuntos.Comparacion):
    """Corre el reporte de Evidently sobre una comparacion y devuelve el ``Snapshot``."""
    definicion = definicion_datos()
    actual = Dataset.from_pandas(
        conjuntos.para_evidently(comparacion.actual), data_definition=definicion
    )
    referencia = Dataset.from_pandas(
        conjuntos.para_evidently(comparacion.referencia), data_definition=definicion
    )
    metadatos = {clave: str(valor) for clave, valor in comparacion.info.items()}
    return construir_reporte().run(
        actual,
        referencia,
        metadata=metadatos,
        tags=["p1", "drift", comparacion.nombre],
        name=f"drift-p1-{comparacion.nombre}",
    )


def _tipo(metrica: dict) -> str:
    """``evidently:metric_v2:ValueDrift`` -> ``ValueDrift``."""
    return str(metrica.get("config", {}).get("type", "")).rsplit(":", 1)[-1]


def es_p_valor(metodo: str) -> bool:
    """Las pruebas estadisticas (K-S, chi-cuadrada, Z) dan un p-valor: hay drift si es BAJO.
    Las distancias (Wasserstein, Jensen-Shannon) hay drift si son ALTAS."""
    return "p_value" in metodo


def puntajes(resultado: dict, comparacion: str) -> pd.DataFrame:
    """Una fila por variable: metodo, umbral, puntaje de drift y si deriva."""
    filas = []
    for metrica in resultado["metrics"]:
        if _tipo(metrica) != "ValueDrift":
            continue
        ajuste = metrica["config"]
        metodo = str(ajuste["method"])
        valor, umbral = float(metrica["value"]), float(ajuste["threshold"])
        filas.append(
            {
                "comparacion": comparacion,
                "variable": ajuste["column"],
                "metodo": metodo,
                "umbral": umbral,
                "puntaje": valor,
                "deriva": valor < umbral if es_p_valor(metodo) else valor >= umbral,
            }
        )
    if not filas:
        raise RuntimeError(
            "El reporte de Evidently no trae metricas ValueDrift. Se probo con evidently 0.7.23; "
            "revisa la version instalada (requirements-monitoreo.txt)."
        )
    return pd.DataFrame(filas)


def conteo_columnas_con_drift(resultado: dict) -> dict[str, float]:
    """``{"count": ..., "share": ...}`` de la metrica ``DriftedColumnsCount``."""
    for metrica in resultado["metrics"]:
        if _tipo(metrica) == "DriftedColumnsCount":
            return {clave: float(valor) for clave, valor in metrica["value"].items()}
    raise RuntimeError("El reporte de Evidently no trae DriftedColumnsCount")


def calidad(resultado: dict) -> dict[str, float]:
    """Metricas de calidad del conjunto actual (las de ``CALIDAD``)."""
    valores = {}
    for metrica in resultado["metrics"]:
        nombre = _tipo(metrica)
        if nombre in CALIDAD and "column" not in metrica["config"]:
            valor = metrica["value"]
            valores[CALIDAD[nombre]] = float(valor["count"] if isinstance(valor, dict) else valor)
    return valores


def pruebas_fallidas(resultado: dict) -> list[dict[str, str]]:
    """Pruebas automaticas de Evidently que no pasaron, salvo las de drift (ya estan en la tabla
    de drift por variable). Casi todas comparan un estadistico del conjunto actual (media,
    cuantiles, minimo, maximo) contra la referencia con una tolerancia de +-10 %."""
    fallidas = []
    for prueba in resultado.get("tests", []):
        estado = str(getattr(prueba["status"], "value", prueba["status"]))
        tipo = str(prueba.get("metric_config", {}).get("params", {}).get("type", ""))
        if estado != "SUCCESS" and tipo.rsplit(":", 1)[-1] not in PRUEBAS_DE_DRIFT:
            fallidas.append(
                {"estado": estado, "prueba": prueba["name"], "detalle": prueba["description"]}
            )
    return fallidas


def guardar_reporte(snapshot, nombre: str, carpeta: Path = config.MONITOREO_DIR) -> dict:
    """Escribe el HTML (Git lo ignora) y el JSON de metricas; devuelve las metricas."""
    carpeta.mkdir(parents=True, exist_ok=True)
    # save_html de Evidently solo escribe si recibe un str (con un Path no hace nada)
    snapshot.save_html(str(carpeta / f"drift_{nombre}.html"))
    (carpeta / f"drift_{nombre}.json").write_text(snapshot.json(), encoding="utf-8")
    return json.loads(snapshot.json())


# =============================================================================
# Interpretacion: causa probable de cada drift
# =============================================================================
def causa_probable(fila: pd.Series) -> str:
    """Lee la "escalera" de comparaciones de una variable (ver config.MONITOREO_COMPARACIONES).

    - deriva en ``principal`` y no en ``sin_nuevas``: la explican las estaciones nuevas;
    - deriva en ``sin_nuevas`` y no en ``misma_temporada``: es la temporada (estacionalidad);
    - deriva en ``misma_temporada``: cambio de un anio a otro en las estaciones que ya existian
      (puede quedar oculto en ``principal`` si la temporada lo compensa).

    Puede haber mas de una causa. Es una guia para interpretar, no una prueba: confirma cada caso
    con las graficas y el reporte HTML.
    """
    if not {"principal", "sin_nuevas", "misma_temporada"} <= set(fila.index):
        return "-"
    causas = []
    if fila["principal"] and not fila["sin_nuevas"]:
        causas.append("estaciones nuevas")
    if fila["sin_nuevas"] and not fila["misma_temporada"]:
        causas.append("temporada")
    if fila["misma_temporada"]:
        causas.append("cambio de un anio en la red existente")
    return " + ".join(causas) if causas else "estable"


def tabla_variables(tabla_puntajes: pd.DataFrame, estadistica: pd.DataFrame) -> pd.DataFrame:
    """Puntaje por comparacion, causa probable y cambio de cada variable (en la principal)."""
    puntaje = tabla_puntajes.pivot(index="variable", columns="comparacion", values="puntaje")
    deriva = tabla_puntajes.pivot(index="variable", columns="comparacion", values="deriva")
    orden = [nombre for nombre in ORDEN_COMPARACIONES if nombre in puntaje.columns]
    tabla = puntaje[orden].copy()
    tabla["causa_probable"] = deriva[orden].apply(causa_probable, axis=1)
    tabla = tabla.join(deriva[orden].astype(bool).add_prefix("deriva_"))
    metodo = tabla_puntajes.drop_duplicates("variable").set_index("variable")["metodo"]
    tabla.insert(0, "metodo", metodo)
    tabla = tabla.join(estadistica.set_index("variable"))
    ordenar_por = "principal" if "principal" in tabla.columns else orden[0]
    return tabla.sort_values(ordenar_por, ascending=False)


# =============================================================================
# Figuras
# =============================================================================
def grafica_puntajes(
    tabla: pd.DataFrame, umbral: float = config.MONITOREO_UMBRAL, tope: float = 1.0
):
    """Barras horizontales del puntaje de drift por variable y comparacion, con el umbral.

    La distancia de Wasserstein normalizada no tiene maximo (se divide entre la desviacion
    estandar de la referencia); para que una variable no aplaste a las demas, las barras se
    recortan en ``tope`` y se anota el valor real de las recortadas.
    """
    orden = [nombre for nombre in ORDEN_COMPARACIONES if nombre in tabla.columns]
    datos = tabla[orden].iloc[::-1]  # la de mayor drift arriba
    alto_barra = 0.8 / len(orden)
    figura, ax = plt.subplots(figsize=(8, 0.42 * len(datos) + 1.6))
    posiciones = np.arange(len(datos))
    for i, nombre in enumerate(orden):
        y = posiciones + ((len(orden) - 1) / 2 - i) * alto_barra
        valores = datos[nombre].to_numpy(dtype=float)
        ax.barh(y, np.minimum(valores, tope), height=alto_barra * 0.85, color=SERIES[i],
                label=ETIQUETAS.get(nombre, nombre))  # fmt: skip
        for yi, valor in zip(y, valores, strict=True):
            if valor > tope:
                ax.annotate(f"{valor:.2f}", (tope, yi), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=7, color=TINTA_SECUNDARIA)  # fmt: skip
    ax.axvline(umbral, color=TINTA_SECUNDARIA, linestyle="--", linewidth=1)
    ax.annotate(f"umbral {umbral:g}", (umbral, len(datos) - 0.45), xytext=(4, 0),
                textcoords="offset points", color=TINTA_SECUNDARIA, fontsize=8)  # fmt: skip
    ax.set_xlim(0, tope * 1.12)
    ax.set_yticks(posiciones, datos.index)
    ax.set_xlabel(
        f"Puntaje de drift (Wasserstein normalizada o Jensen-Shannon; recorte en {tope:g})"
    )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _titulo(figura, "Drift por variable (Evidently)", "P1: referencia contra periodo actual")
    return figura


def grafica_distribuciones(comparacion: conjuntos.Comparacion, variables: list[str]):
    """Referencia contra actual para las variables con mas drift (2 x 2 paneles)."""
    columnas = 2
    filas = int(np.ceil(len(variables) / columnas))
    figura, ejes = plt.subplots(filas, columnas, figsize=(8, 2.6 * filas + 0.8), squeeze=False)
    for ax, variable in zip(ejes.flat, variables, strict=False):
        ref = comparacion.referencia[variable].astype(float)
        act = comparacion.actual[variable].astype(float)
        if variable in conjuntos.CATEGORICAS:
            prop_ref = ref.value_counts(normalize=True)
            prop_act = act.value_counts(normalize=True)
            categorias = sorted(set(prop_ref.index) | set(prop_act.index))
            x = np.arange(len(categorias))
            ax.bar(x - 0.2, prop_ref.reindex(categorias, fill_value=0), 0.38, color=SERIES[0],
                   label="Referencia")  # fmt: skip
            ax.bar(x + 0.2, prop_act.reindex(categorias, fill_value=0), 0.38, color=SERIES[1],
                   label="Actual")  # fmt: skip
            paso = max(1, len(categorias) // 12)
            ax.set_xticks(x[::paso], [f"{c:g}" for c in categorias][::paso])
            ax.set_ylabel("Proporcion de filas")
        else:
            juntos = np.concatenate([ref.to_numpy(), act.to_numpy()])
            bajo, alto = np.nanquantile(juntos, [0.01, 0.99])
            if alto <= bajo:
                bajo, alto = bajo - 0.5, alto + 0.5
            bordes = np.linspace(bajo, alto, 31)
            ax.hist(ref.clip(bajo, alto), bins=bordes, density=True, histtype="step",
                    color=SERIES[0], linewidth=2, label="Referencia")  # fmt: skip
            ax.hist(act.clip(bajo, alto), bins=bordes, density=True, histtype="step",
                    color=SERIES[1], linewidth=2, label="Actual")  # fmt: skip
            ax.set_ylabel("Densidad")
        ax.set_title(variable, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in list(ejes.flat)[len(variables) :]:
        ax.set_visible(False)
    _titulo(
        figura,
        "Variables con mas drift: referencia vs. actual",
        "Recorte en los percentiles 1 y 99",
    )
    figura.legend(*ejes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=2)
    return figura


# =============================================================================
# Resumen en Markdown
# =============================================================================
def ascii_simple(texto: str) -> str:
    """Texto de Evidently en ASCII (convencion del repo): "+-" en lugar del simbolo."""
    texto = str(texto).replace("\u00b1", "+-").replace("|", "/")
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _formato(valor, decimales: int = 3) -> str:
    if isinstance(valor, float | np.floating):
        return "-" if np.isnan(valor) else f"{valor:.{decimales}f}"
    return str(valor)


def _cambio(fila: pd.Series) -> str:
    if fila.get("tipo") == "categorica":
        return f"mayor cambio de proporcion en {fila['mayor_cambio']}"
    return (
        f"media {fila['media_referencia']:.2f} -> {fila['media_actual']:.2f} "
        f"({fila['cambio_media']:+.0%}); ceros {fila['ceros_referencia']:.0%} -> "
        f"{fila['ceros_actual']:.0%}"
    )


def escribir_resumen(
    infos: list[dict],
    conteos: dict[str, dict],
    tabla: pd.DataFrame,
    calidad_actual: dict[str, float],
    fallidas: list[dict],
    ruta: Path = config.RESUMEN_DRIFT_MD,
) -> Path:
    """Resumen versionado: periodos, tamanos, drift por variable, calidad y pruebas fallidas."""
    orden = [nombre for nombre in ORDEN_COMPARACIONES if nombre in tabla.columns]
    lineas = [
        "# Monitoreo de P1: drift con Evidently",
        "",
        "Generado con `python -m monitoreo.drift` (ambiente `.venv-monitoreo`). No editar a mano.",
        "",
        f"- **Version de datos (`data_version`):** `{contrato.version_datos()}`",
        (
            f"- **Evidently:** {evidently.__version__}. Numericas: "
            f"`{config.MONITOREO_METODO_NUMERICAS}` (distancia de Wasserstein normalizada); "
            "categoricas: "
            f"`{config.MONITOREO_METODO_CATEGORICAS}` (distancia de Jensen-Shannon). Una variable "
            f"deriva si su distancia es >= {config.MONITOREO_UMBRAL}; hay drift del conjunto si "
            f"deriva al menos el {config.MONITOREO_DRIFT_SHARE:.0%} de las variables."
        ),
        (
            f"- **Muestra:** hasta {config.MONITOREO_MUESTRA:,} filas por conjunto (semilla "
            f"{config.SEMILLA}). Reportes HTML en `reports/monitoreo/drift_<comparacion>.html` "
            "(no se versionan)."
        ),
        "",
        "## Conjuntos comparados",
        "",
        (
            "| Comparacion | Referencia | Actual | Solo conocidas | Filas ref. | Filas actual | "
            "Estaciones comparadas (ref. / actual) | Estaciones nuevas en el periodo actual "
            "(fraccion de sus filas) | Variables con drift |"
        ),
        "|---|---|---|---|---:|---:|---|---|---|",
    ]
    for info in infos:
        conteo = conteos[info["nombre"]]
        conjunto = "si" if conteo["share"] >= config.MONITOREO_DRIFT_SHARE else "no"
        lineas.append(
            f"| {info['nombre']} | {info['referencia']} | {info['actual']} | "
            f"{'si' if info['solo_conocidas'] else 'no'} | {info['filas_referencia']:,} | "
            f"{info['filas_actual']:,} | {info['estaciones_referencia']} / "
            f"{info['estaciones_actual']} | {info['estaciones_nuevas_periodo']} "
            f"({info['fraccion_filas_nuevas_periodo']:.1%}) | {conteo['count']:.0f} de "
            f"{len(tabla)} ({conteo['share']:.0%}); drift del conjunto: {conjunto} |"
        )
    lineas += [
        "",
        "## Drift por variable",
        "",
        (
            "Puntaje de Evidently en cada comparacion (en **negritas** si deriva). La causa "
            "probable lee la escalera de comparaciones: si deriva en `principal` pero no en "
            "`sin_nuevas`, la explican las estaciones nuevas; si deriva en `sin_nuevas` pero "
            "no en `misma_temporada`, es la temporada; si deriva en `misma_temporada`, cambio "
            "de un anio a otro la red que ya existia. Es una guia: se confirma con las "
            "graficas. El cambio se mide en la comparacion principal."
        ),
        "",
        "| Variable | Metodo | " + " | ".join(orden) + " | Causa probable | Cambio |",
        "|---|---|" + "---:|" * len(orden) + "---|---|",
    ]
    for variable, fila in tabla.iterrows():
        celdas = []
        for nombre in orden:
            texto = _formato(fila[nombre])
            celdas.append(f"**{texto}**" if fila[f"deriva_{nombre}"] else texto)
        lineas.append(
            f"| `{variable}` | {fila['metodo']} | " + " | ".join(celdas) + f" | "
            f"{fila['causa_probable']} | {_cambio(fila)} |"
        )
    lineas += [
        "",
        "## Calidad de los datos (conjunto actual, comparacion principal)",
        "",
        "| Revision (DataSummaryPreset) | Valor |",
        "|---|---:|",
        *[f"| {nombre} | {valor:,.0f} |" for nombre, valor in calidad_actual.items()],
        "",
        (
            "Las filas con valores repetidos no son registros duplicados: la llave "
            "`(estacion_id, hora)` es unica (lo revisa la validacion de datos). Son horas "
            "distintas con exactamente las mismas variables; lo esperable es que sean horas sin "
            "viajes de una misma estacion (rezagos y medias en cero)."
        ),
        "",
        (
            "Pruebas automaticas de Evidently (sin las de drift) que no pasaron en la comparacion "
            f"principal: {len(fallidas)}. Casi todas comparan un estadistico del conjunto actual "
            "contra la referencia con +-10 % de tolerancia, asi que repiten el drift con otras "
            "palabras."
        ),
        "",
    ]
    if fallidas:
        lineas += ["| Estado | Prueba | Detalle |", "|---|---|---|"]
        lineas += [
            f"| {prueba['estado']} | {ascii_simple(prueba['prueba'])} | "
            f"{ascii_simple(prueba['detalle'])} |"
            for prueba in fallidas[:MAX_PRUEBAS_EN_RESUMEN]
        ]
        if len(fallidas) > MAX_PRUEBAS_EN_RESUMEN:
            lineas.append(
                f"\n... y {len(fallidas) - MAX_PRUEBAS_EN_RESUMEN} mas en "
                "`reports/monitoreo/drift_principal.html` (pestana de pruebas)."
            )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


# =============================================================================
# Orquestacion
# =============================================================================
def ejecutar(
    p1: pd.DataFrame,
    comparaciones: list[str] | None = None,
    muestra: int = config.MONITOREO_MUESTRA,
    carpeta: Path = config.MONITOREO_DIR,
    figuras: Path = config.MONITOREO_FIGURAS_DIR,
    resumen: Path = config.RESUMEN_DRIFT_MD,
) -> pd.DataFrame:
    """Corre las comparaciones pedidas y escribe todas las salidas; devuelve la tabla final."""
    comparaciones = comparaciones or ORDEN_COMPARACIONES
    conocidas = conjuntos.estaciones_conocidas(p1)
    todas, infos, conteos, armadas = [], [], {}, {}
    calidad_actual, fallidas = {}, []
    for nombre in comparaciones:
        comparacion = conjuntos.armar_comparacion(
            p1, nombre, config.MONITOREO_COMPARACIONES[nombre], conocidas, muestra
        )
        logger.info(f"{nombre}: {comparacion.info}")
        resultado = guardar_reporte(correr(comparacion), nombre, carpeta)
        todas.append(puntajes(resultado, nombre))
        conteos[nombre] = conteo_columnas_con_drift(resultado)
        infos.append(comparacion.info)
        armadas[nombre] = comparacion
        if nombre == comparaciones[0]:
            calidad_actual, fallidas = calidad(resultado), pruebas_fallidas(resultado)
        logger.success(f"{nombre}: {conteos[nombre]['count']:.0f} variables con drift")

    tabla_puntajes = pd.concat(todas, ignore_index=True)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / config.PUNTAJES_DRIFT_CSV.name
    tabla_puntajes.to_csv(ruta_csv, index=False, float_format="%.6f")

    primera = armadas[comparaciones[0]]
    tabla = tabla_variables(
        tabla_puntajes, conjuntos.estadisticas(primera.referencia, primera.actual)
    )
    escribir_resumen(infos, conteos, tabla, calidad_actual, fallidas, resumen)
    guardar_figura(grafica_puntajes(tabla), figuras / "drift_puntajes.png")
    principales = tabla.index[: min(4, len(tabla))].tolist()
    guardar_figura(
        grafica_distribuciones(primera, principales), figuras / "drift_distribuciones.png"
    )
    logger.success(f"Resumen en {resumen}; puntajes en {ruta_csv}; figuras en {figuras}")
    return tabla


def main(argumentos: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reporte de drift de P1 con Evidently.")
    parser.add_argument("--muestra", type=int, default=config.MONITOREO_MUESTRA)
    parser.add_argument(
        "--comparaciones",
        default=",".join(ORDEN_COMPARACIONES),
        help="Lista separada por comas; la primera da la calidad y las figuras.",
    )
    opciones = parser.parse_args(argumentos)
    pedidas = [nombre.strip() for nombre in opciones.comparaciones.split(",") if nombre.strip()]
    desconocidas = sorted(set(pedidas) - set(config.MONITOREO_COMPARACIONES))
    if desconocidas:
        parser.error(f"Comparaciones desconocidas {desconocidas}; opciones {ORDEN_COMPARACIONES}")

    estado = contrato.estado_datos()
    if estado != "ok":
        raise SystemExit(
            f"Los datos estan {estado}: corre `make data && make features` en el ambiente "
            "principal antes del monitoreo."
        )
    ejecutar(conjuntos.cargar_p1(), pedidas, opciones.muestra)


if __name__ == "__main__":
    main()
