"""Graficas del proyecto: exploracion (EDA) y artefactos de los experimentos.

Todas las funciones devuelven una figura de matplotlib y ``guardar_figura`` la escribe en
disco. Convenciones, iguales en todas las graficas:

- Paleta validada para daltonismo: una serie en azul; dos, azul y naranja; tres, mas aqua.
- Con mas de tres grupos se usan paneles pequenos (un panel por grupo resaltado sobre el
  resto en gris), en lugar de agregar colores dificiles de distinguir.
- Cuadricula fina y discreta, una sola escala por grafica y leyenda cuando hay dos o mas
  series. El texto nunca usa el color de la serie.

Uso para la exploracion::

    python -m tc5061_proyecto_equipo18.plots   # escribe reports/figures/eda/*.png
"""

import json
from pathlib import Path
import sys

import matplotlib

if "ipykernel" not in sys.modules:
    # Fuera de Jupyter: sin ventanas, igual en scripts, servidores y Windows.
    # En un notebook se respeta el backend en linea para que las figuras se vean.
    matplotlib.use("Agg")

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
import typer

from tc5061_proyecto_equipo18 import config

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# Paleta (instancia de referencia validada; slots en orden fijo)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # azul, naranja, aqua
CATEGORICA_8 = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
GRIS_CONTEXTO = "#d9d8d2"
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": TINTA_TENUE,
        "ytick.color": TINTA_TENUE,
        "xtick.labelcolor": TINTA_SECUNDARIA,
        "ytick.labelcolor": TINTA_SECUNDARIA,
        "font.size": 9,
        "legend.frameon": False,
        "legend.labelcolor": TINTA_SECUNDARIA,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def guardar_figura(figura: Figure, ruta: Path) -> Path:
    """Guarda la figura como PNG (150 ppp) y libera la memoria."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return ruta


def _titulo(figura: Figure, titulo: str, subtitulo: str | None = None) -> None:
    """Titulo y subtitulo alineados a la izquierda, a distancia fija en pulgadas; ajusta margenes."""
    alto = figura.get_figheight()
    figura.text(0.01, 1 - 0.08 / alto, titulo, ha="left", va="top", fontsize=12,
                fontweight="bold", color=TINTA)  # fmt: skip
    reservado = 0.42
    if subtitulo:
        figura.text(0.01, 1 - 0.34 / alto, subtitulo, ha="left", va="top", fontsize=9,
                    color=TINTA_SECUNDARIA)  # fmt: skip
        reservado = 0.62
    figura.tight_layout(rect=(0, 0, 1, 1 - reservado / alto))


def _cuadricula(n: int, columnas: int = 4) -> tuple[int, int]:
    columnas = min(columnas, n)
    return int(np.ceil(n / columnas)), columnas


# =============================================================================
# Exploracion (EDA)
# =============================================================================
def grafica_viajes_por_mes(viajes: pd.DataFrame) -> Figure:
    """Viajes por mes: volumen y estacionalidad del rango elegido."""
    por_mes = viajes.groupby(viajes["inicio_viaje"].dt.to_period("M")).size()
    figura, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(por_mes.index.astype(str), por_mes.to_numpy() / 1000, width=0.6, color=SERIES[0])
    ax.set_ylabel("Miles de viajes")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="x", visible=False)
    _titulo(figura, "Viajes por mes", f"{len(viajes):,} viajes en el rango")
    return figura


def grafica_perfil_horario(conteos: pd.DataFrame) -> Figure:
    """Salidas promedio por estacion y hora: dias laborales vs. fines de semana."""
    datos = conteos.assign(
        hora_dia=conteos["hora"].dt.hour, fin_semana=conteos["hora"].dt.dayofweek >= 5
    )
    perfil = datos.groupby(["fin_semana", "hora_dia"])["salidas"].mean().unstack("fin_semana")
    figura, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(perfil.index, perfil[False], color=SERIES[0], label="Lunes a viernes")
    ax.plot(perfil.index, perfil[True], color=SERIES[1], label="Sabado y domingo")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Salidas por estacion")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _titulo(figura, "Perfil horario de la demanda", "Promedio de salidas por estacion y hora")
    return figura


def grafica_mapa_estaciones(estaciones: pd.DataFrame, volumen: pd.Series | None = None) -> Figure:
    """Ubicacion de las estaciones; el tamanio del punto es proporcional a su volumen."""
    activas = estaciones[~estaciones["retirada"]] if "retirada" in estaciones else estaciones
    tamanos = 12.0
    if volumen is not None:
        valores = activas["estacion_id"].map(volumen).fillna(0).to_numpy(dtype=float)
        tamanos = 8 + 120 * valores / max(valores.max(), 1)
    figura, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(activas["longitud"], activas["latitud"], s=tamanos, color=SERIES[0], alpha=0.75,
               edgecolors=SUPERFICIE, linewidths=0.8)  # fmt: skip
    ax.set_aspect(1 / np.cos(np.deg2rad(activas["latitud"].mean())))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    subtitulo = "Tamanio proporcional al numero de viajes" if volumen is not None else None
    _titulo(figura, f"Estaciones de MiBici ({len(activas)} activas)", subtitulo)
    return figura


def grafica_flujo_neto_relativo(
    p2: pd.DataFrame, umbral: float = config.P2_UMBRAL_RIESGO
) -> Figure:
    """Distribucion del flujo neto por bloque relativo a la capacidad (justifica el umbral de P2)."""
    relativo = (p2["neto_lag_dia"] / p2["capacidad"]).clip(-1.5, 1.5)
    figura, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(relativo, bins=np.linspace(-1.5, 1.5, 61), color=SERIES[0], rwidth=0.85)
    for lado in (-umbral, umbral):
        ax.axvline(lado, color=TINTA_SECUNDARIA, linewidth=1)
    ax.text(
        umbral,
        ax.get_ylim()[1] * 0.9,
        f"  riesgo de vaciarse (>= {umbral:.0%})",
        color=TINTA_SECUNDARIA,
    )
    ax.text(
        -umbral, ax.get_ylim()[1] * 0.9, "riesgo de llenarse  ", ha="right", color=TINTA_SECUNDARIA
    )
    ax.set_yscale("log")
    ax.set_xlabel("(salidas - llegadas) / capacidad en un bloque")
    ax.set_ylabel("Bloques (escala log)")
    _titulo(figura, "Flujo neto por bloque relativo a la capacidad")
    return figura


# =============================================================================
# P1 - Regresion
# =============================================================================
def grafica_real_vs_predicho(
    datos: pd.DataFrame,
    columna_tiempo: str,
    real: str,
    predicho: str,
    titulo: str,
    n_estaciones: int = 3,
) -> Figure:
    """Serie real vs. predicha de las estaciones con mas demanda, una semana del conjunto."""
    principales = datos.groupby("estacion_id")[real].sum().nlargest(n_estaciones).index
    inicio = datos[columna_tiempo].min().normalize()
    semana = datos[
        (datos[columna_tiempo] >= inicio) & (datos[columna_tiempo] < inicio + pd.Timedelta(days=7))
    ]
    figura, ejes = plt.subplots(n_estaciones, 1, figsize=(10, 2.3 * n_estaciones), sharex=True)
    for ax, estacion in zip(np.atleast_1d(ejes), principales):
        serie = semana[semana["estacion_id"] == estacion].sort_values(columna_tiempo)
        ax.plot(serie[columna_tiempo], serie[real], color=SERIES[0], label="Real")
        ax.plot(serie[columna_tiempo], serie[predicho], color=SERIES[1], label="Predicho")
        ax.set_title(
            f"Estacion {estacion}", fontsize=9, color=TINTA_SECUNDARIA, fontweight="normal"
        )
        ax.set_ylim(bottom=0)
    np.atleast_1d(ejes)[0].legend(loc="upper right", ncols=2)
    _titulo(figura, titulo, "Salidas por hora, primera semana del conjunto")
    return figura


def grafica_residuales(
    real: np.ndarray, predicho: np.ndarray, titulo: str, n_niveles: int = 20
) -> Figure:
    """Residuales (real - predicho): sesgo por nivel de prediccion y su distribucion.

    A la izquierda, el residuo medio y la banda de percentiles 25 a 75 en grupos de igual
    tamanio segun el valor predicho (un modelo sin sesgo queda centrado en 0). A la derecha, el
    histograma de residuos, recortado a los percentiles 0.5 y 99.5 para que se lea la forma.
    """
    datos = pd.DataFrame({"predicho": np.asarray(predicho, float)})
    datos["residuo"] = np.asarray(real, float) - datos["predicho"]
    niveles = pd.qcut(datos["predicho"], q=n_niveles, duplicates="drop")
    resumen = datos.groupby(niveles, observed=True).agg(
        predicho=("predicho", "mean"),
        media=("residuo", "mean"),
        p25=("residuo", lambda r: r.quantile(0.25)),
        p75=("residuo", lambda r: r.quantile(0.75)),
    )
    figura, (izq, der) = plt.subplots(1, 2, figsize=(11, 3.8))
    izq.fill_between(resumen["predicho"], resumen["p25"], resumen["p75"], color=SERIES[0],
                     alpha=0.2, linewidth=0, label="Percentiles 25 a 75")  # fmt: skip
    izq.plot(resumen["predicho"], resumen["media"], color=SERIES[0], marker="o", markersize=4,
             label="Residuo medio")  # fmt: skip
    izq.axhline(0, color=TINTA_SECUNDARIA, linewidth=1)
    izq.set_xlabel("Valor predicho (promedio del grupo)")
    izq.set_ylabel("Real - predicho")
    izq.legend(loc="upper left")
    limites = datos["residuo"].quantile([0.005, 0.995]).to_numpy()
    der.hist(datos["residuo"].clip(*limites), bins=60, color=SERIES[0], rwidth=0.85)
    der.axvline(0, color=TINTA_SECUNDARIA, linewidth=1)
    der.set_xlabel("Real - predicho")
    der.set_ylabel("Filas")
    der.grid(axis="x", visible=False)
    _titulo(figura, titulo, "Residuo = real - predicho; positivo = el modelo se quedo corto")
    return figura


def grafica_error_por_hora(datos: pd.DataFrame, real: str, predicho: str, titulo: str) -> Figure:
    """Error absoluto medio por hora del dia (donde falla el modelo)."""
    error = (datos[real] - datos[predicho]).abs().groupby(datos["hora_dia"]).mean()
    figura, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(error.index, error.to_numpy(), width=0.6, color=SERIES[0])
    pico = error.idxmax()
    ax.annotate(f"{error.max():.2f}", (pico, error.max()), textcoords="offset points", xytext=(0, 4),
                ha="center", color=TINTA_SECUNDARIA)  # fmt: skip
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("MAE (viajes)")
    ax.grid(axis="x", visible=False)
    _titulo(figura, titulo)
    return figura


def grafica_importancias(importancias: pd.Series, titulo: str, n: int = 15) -> Figure:
    """Importancia por permutacion de las variables (caida de la metrica al barajarlas)."""
    principales = importancias.sort_values().tail(n)
    figura, ax = plt.subplots(figsize=(7, 0.32 * len(principales) + 1.2))
    ax.barh(principales.index, principales.to_numpy(), height=0.6, color=SERIES[0])
    ax.set_xlabel("Caida de la metrica al permutar la variable")
    ax.grid(axis="y", visible=False)
    _titulo(figura, titulo)
    return figura


# =============================================================================
# P2 - Clasificacion
# =============================================================================
def grafica_matriz_confusion(
    y_real: np.ndarray, y_predicho: np.ndarray, etiquetas: dict[int, str], titulo: str
) -> Figure:
    """Matriz de confusion normalizada por clase real (recall en la diagonal) con conteos."""
    clases = list(etiquetas)
    matriz = confusion_matrix(y_real, y_predicho, labels=clases)
    proporcion = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    figura, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(proporcion, cmap=RAMPA_AZUL, vmin=0, vmax=1)
    ax.grid(False)
    nombres = [etiquetas[c].replace("_", " ") for c in clases]
    ax.set_xticks(range(len(clases)), nombres, rotation=20, ha="right")
    ax.set_yticks(range(len(clases)), nombres)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    for i in range(len(clases)):
        for j in range(len(clases)):
            color_texto = "white" if proporcion[i, j] > 0.55 else TINTA
            ax.text(j, i, f"{proporcion[i, j]:.0%}\n({matriz[i, j]:,})", ha="center", va="center",
                    fontsize=8, color=color_texto)  # fmt: skip
    _titulo(figura, titulo, "Porcentaje por fila (clase real) y conteo")
    return figura


def grafica_curvas_pr(
    y_real: np.ndarray, probabilidades: np.ndarray, etiquetas: dict[int, str], titulo: str
) -> Figure:
    """Curvas precision-recall de las clases de riesgo (uno contra el resto)."""
    figura, ax = plt.subplots(figsize=(6, 4.4))
    for color, clase in zip(SERIES, [c for c in etiquetas if c != 0]):
        precision, recall, _ = precision_recall_curve(y_real == clase, probabilidades[:, clase])
        prevalencia = float(np.mean(y_real == clase))
        ax.plot(recall, precision, color=color, label=f"{etiquetas[clase].replace('_', ' ')}")
        ax.axhline(prevalencia, color=color, linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (riesgos detectados)")
    ax.set_ylabel("Precision (alertas correctas)")
    ax.legend(loc="upper right")
    _titulo(
        figura, titulo, "Lineas horizontales: prevalencia de cada clase (clasificador al azar)"
    )
    return figura


def grafica_riesgo_por_bloque(
    datos: pd.DataFrame, real: str, predicho: str, etiquetas: dict[int, str], titulo: str
) -> Figure:
    """Tasa de bloques en riesgo por hora de inicio del bloque: real vs. predicha."""
    clases = [c for c in etiquetas if c != 0]
    figura, ejes = plt.subplots(1, len(clases), figsize=(10, 3.4), sharey=True)
    for ax, clase in zip(np.atleast_1d(ejes), clases):
        tasa_real = (datos[real] == clase).groupby(datos["bloque_dia"]).mean() * 100
        tasa_pred = (datos[predicho] == clase).groupby(datos["bloque_dia"]).mean() * 100
        ax.plot(
            tasa_real.index,
            tasa_real.to_numpy(),
            color=SERIES[0],
            marker="o",
            markersize=5,
            label="Real",
        )
        ax.plot(
            tasa_pred.index,
            tasa_pred.to_numpy(),
            color=SERIES[1],
            marker="o",
            markersize=5,
            label="Predicha",
        )
        ax.set_title(
            etiquetas[clase].replace("_", " "),
            fontsize=9,
            color=TINTA_SECUNDARIA,
            fontweight="normal",
        )
        ax.set_xticks(sorted(datos["bloque_dia"].unique()))
        ax.set_xlabel("Hora de inicio del bloque")
        ax.set_ylim(bottom=0)
    np.atleast_1d(ejes)[0].set_ylabel("% de bloques")
    np.atleast_1d(ejes)[0].legend(loc="upper left")
    _titulo(figura, titulo)
    return figura


# =============================================================================
# Busquedas de hiperparametros (corrida padre)
# =============================================================================
def grafica_busqueda(
    resultados: pd.DataFrame, hiperparametros: list[str], metrica: str, sentido: str, titulo: str
) -> Figure:
    """Metrica de seleccion de cada combinacion de una busqueda.

    Con un solo hiperparametro numerico se dibuja una linea por conjunto (entrenamiento,
    validacion y prueba, los que existan) y una linea punteada en la mejor combinacion. Con
    varios, una barra por combinacion con la mejor resaltada.
    """
    base = metrica.split("_", 1)[1]
    series = [
        f"{conjunto}_{base}"
        for conjunto in ("entrenamiento", "validacion", "prueba")
        if f"{conjunto}_{base}" in resultados
    ]
    mejor = resultados[metrica].idxmin() if sentido == "min" else resultados[metrica].idxmax()
    primero = hiperparametros[0]
    if len(hiperparametros) == 1 and pd.api.types.is_numeric_dtype(resultados[primero]):
        orden = resultados.sort_values(primero)
        figura, ax = plt.subplots(figsize=(7, 3.8))
        for color, serie in zip(SERIES, series):
            ax.plot(orden[primero], orden[serie], color=color, marker="o", markersize=6,
                    markeredgecolor=SUPERFICIE, markeredgewidth=2,
                    label=serie.split("_")[0])  # fmt: skip
        ax.axvline(resultados.loc[mejor, primero], color=TINTA_SECUNDARIA, linewidth=1,
                   linestyle="--")  # fmt: skip
        valores = orden[primero].to_numpy(dtype=float)
        if valores.min() > 0 and valores.max() / valores.min() >= 10:
            ax.set_xscale("log")
        ax.set_xticks(valores, [f"{v:g}" for v in valores])
        ax.minorticks_off()
        ax.set_xlabel(primero)
        ax.set_ylabel(base)
        ax.legend(loc="best")
        subtitulo = f"Linea punteada: mejor combinacion por {metrica}"
    else:
        # Puntos (no barras): las diferencias entre combinaciones suelen ser pequenas y el eje
        # se ajusta a ellas; cada punto lleva su valor
        etiquetas = resultados[hiperparametros].apply(
            lambda fila: ", ".join(f"{clave}={fila[clave]:g}" if isinstance(fila[clave], float)
                                   else f"{clave}={fila[clave]}" for clave in hiperparametros),
            axis=1,
        )  # fmt: skip
        colores = [SERIES[1] if indice == mejor else SERIES[0] for indice in resultados.index]
        figura, ax = plt.subplots(figsize=(8, 0.45 * len(resultados) + 1.4))
        posiciones = np.arange(len(resultados))
        ax.scatter(resultados[metrica], posiciones, color=colores, s=60, zorder=3)
        for posicion, valor in zip(posiciones, resultados[metrica]):
            ax.annotate(f"{valor:.4f}", (valor, posicion), textcoords="offset points",
                        xytext=(8, -3), color=TINTA_SECUNDARIA)  # fmt: skip
        ax.set_yticks(posiciones, etiquetas)
        ax.invert_yaxis()
        ax.margins(x=0.25)
        ax.set_xlabel(metrica)
        ax.grid(axis="y", visible=False)
        subtitulo = f"En naranja, la mejor combinacion por {metrica}"
    _titulo(figura, titulo, subtitulo)
    return figura


# =============================================================================
# P3 - Clustering
# =============================================================================
def grafica_mapa_grupos(
    perfiles: pd.DataFrame, titulo: str, columna_grupo: str = "grupo"
) -> Figure:
    """Mapa en paneles: cada panel resalta un grupo sobre el resto de las estaciones en gris."""
    grupos = sorted(perfiles[columna_grupo].unique())
    filas, columnas = _cuadricula(len(grupos))
    figura, ejes = plt.subplots(
        filas, columnas, figsize=(3.2 * columnas, 3.4 * filas), squeeze=False
    )
    aspecto = 1 / np.cos(np.deg2rad(perfiles["latitud"].mean()))
    tamano = 6 + 60 * perfiles["viajes_periodo"] / perfiles["viajes_periodo"].max()
    for ax, grupo in zip(ejes.flat, grupos):
        es_grupo = perfiles[columna_grupo] == grupo
        ax.scatter(
            perfiles.loc[~es_grupo, "longitud"],
            perfiles.loc[~es_grupo, "latitud"],
            s=8,
            color=GRIS_CONTEXTO,
        )
        ax.scatter(perfiles.loc[es_grupo, "longitud"], perfiles.loc[es_grupo, "latitud"], s=tamano[es_grupo],
                   color=SERIES[0], edgecolors=SUPERFICIE, linewidths=0.8)  # fmt: skip
        ax.set_title(f"Grupo {grupo} - {int(es_grupo.sum())} estaciones", fontsize=9,
                     color=TINTA_SECUNDARIA, fontweight="normal")  # fmt: skip
        ax.set_aspect(aspecto)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for ax in list(ejes.flat)[len(grupos) :]:
        ax.set_visible(False)
    _titulo(figura, titulo, "Tamanio proporcional al volumen de viajes")
    return figura


def grafica_perfiles_grupo(
    perfiles: pd.DataFrame, titulo: str, columna_grupo: str = "grupo"
) -> Figure:
    """Perfil horario promedio de cada grupo en dias laborales: salidas vs. llegadas."""
    grupos = sorted(perfiles[columna_grupo].unique())
    filas, columnas = _cuadricula(len(grupos))
    figura, ejes = plt.subplots(
        filas, columnas, figsize=(3.4 * columnas, 2.6 * filas), sharey=True, squeeze=False
    )
    horas = np.arange(24)
    salidas = [f"sal_lab_h{h:02d}" for h in horas]
    llegadas = [f"lle_lab_h{h:02d}" for h in horas]
    for ax, grupo in zip(ejes.flat, grupos):
        datos = perfiles[perfiles[columna_grupo] == grupo]
        ax.plot(horas, datos[salidas].mean().to_numpy() * 100, color=SERIES[0], label="Salidas")
        ax.plot(horas, datos[llegadas].mean().to_numpy() * 100, color=SERIES[1], label="Llegadas")
        ax.set_title(f"Grupo {grupo} - {len(datos)} estaciones", fontsize=9, color=TINTA_SECUNDARIA,
                     fontweight="normal")  # fmt: skip
        ax.set_xticks([0, 6, 12, 18, 23])
    for ax in list(ejes.flat)[len(grupos) :]:
        ax.set_visible(False)
    ejes.flat[0].set_ylabel("% del dia")
    ejes.flat[0].legend(loc="upper left")
    _titulo(figura, titulo, "Dias laborales; porcentaje de los viajes del dia en cada hora")
    return figura


def mapa_html_grupos(perfiles: pd.DataFrame, titulo: str, columna_grupo: str = "grupo") -> str:
    """Mapa interactivo (Leaflet + OpenStreetMap) con una capa por grupo para activar o apagar.

    Las capas permiten ver un grupo a la vez y cada punto muestra su grupo en la ventana
    emergente, asi que la identidad no depende solo del color.
    """
    puntos = [
        {
            "lat": round(float(fila.latitud), 6),
            "lon": round(float(fila.longitud), 6),
            "grupo": int(getattr(fila, columna_grupo)),
            "texto": f"Grupo {int(getattr(fila, columna_grupo))} - {fila.codigo} - {fila.nombre}"
            f" - {int(fila.viajes_periodo):,} viajes",
        }
        for fila in perfiles.itertuples()
    ]
    colores = {
        int(g): CATEGORICA_8[i % len(CATEGORICA_8)]
        for i, g in enumerate(sorted(perfiles[columna_grupo].unique()))
    }
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>{titulo}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#mapa{{height:100%;margin:0}} body{{font-family:system-ui,sans-serif}}</style>
</head><body><div id="mapa"></div><script>
const puntos = {json.dumps(puntos).replace("</", "<\\/")};
const colores = {json.dumps(colores)};
const mapa = L.map('mapa');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 18, attribution: '&copy; OpenStreetMap'}}).addTo(mapa);
const capas = {{}};
for (const p of puntos) {{
  const clave = 'Grupo ' + p.grupo;
  capas[clave] = capas[clave] || L.layerGroup().addTo(mapa);
  L.circleMarker([p.lat, p.lon], {{radius: 6, color: '#fcfcfb', weight: 2,
    fillColor: colores[p.grupo], fillOpacity: 0.9}})
    .bindPopup(() => {{ const d = document.createElement('div'); d.textContent = p.texto; return d; }})
    .addTo(capas[clave]);
}}
L.control.layers(null, capas, {{collapsed: false}}).addTo(mapa);
mapa.fitBounds(puntos.map(p => [p.lat, p.lon]), {{padding: [20, 20]}});
</script></body></html>
"""


# =============================================================================
# Comando: figuras de exploracion para el reporte
# =============================================================================
@app.command()
def main(salida: Path = config.EDA_FIGURAS_DIR):
    """Genera las figuras de exploracion (requiere ingesta y features)."""
    from tc5061_proyecto_equipo18.features import cargar_estaciones, cargar_viajes

    estaciones = cargar_estaciones()
    viajes = cargar_viajes(columnas=("viaje_id", "inicio_viaje", "origen_id"))
    conteos = pd.read_parquet(config.CONTEOS_HORA_PARQUET)
    volumen = viajes.groupby("origen_id").size()
    guardar_figura(grafica_viajes_por_mes(viajes), salida / "viajes_por_mes.png")
    guardar_figura(grafica_perfil_horario(conteos), salida / "perfil_horario.png")
    guardar_figura(grafica_mapa_estaciones(estaciones, volumen), salida / "mapa_estaciones.png")
    if config.DATASET_P2.exists():
        p2 = pd.read_parquet(config.DATASET_P2, columns=["neto_lag_dia", "capacidad"])
        guardar_figura(grafica_flujo_neto_relativo(p2), salida / "flujo_neto_relativo.png")
    logger.success(f"Figuras de exploracion en {salida.relative_to(config.PROJ_ROOT)}")


if __name__ == "__main__":
    app()
