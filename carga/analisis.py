"""Resumen y graficas de la prueba de carga: latencia, peticiones por segundo, errores y recursos.

Entradas (las deja ``carga.ejecutar``):

- ``peticiones.csv``: una fila por peticion (``tiempo, endpoint, ms, fallo, usuarios, filas``),
  escrita por el locustfile. ``filas`` = predicciones entregadas (0 si fallo o es ``/salud``).
- ``recursos.csv``: muestras de ``docker stats`` por contenedor (ver ``carga.recursos``).

Salidas:

- ``escalones.csv``: por escalon de usuarios, peticiones y predicciones por segundo, percentiles
  de latencia en tiempo real, p95 de los lotes, tasa de error y CPU y memoria de la API; dice si
  cumple el objetivo de servicio (``config.CARGA_SLO_*``).
- ``resumen_carga.json``: totales, por endpoint, escalones, capacidad y entorno de la corrida (lo
  lee el analisis FinOps).
- ``resumen_carga.md`` y figuras en ``reports/figures/carga/``.

Los percentiles salen de los tiempos de cada peticion, no de los promedios que Locust reporta por
ventana, y se calculan sin los primeros ``config.CARGA_CALENTAMIENTO_SEG`` de cada escalon. Los
percentiles "en tiempo real" excluyen los lotes (``... [lote]``), que no tienen a nadie esperando
la respuesta y tardan mas por traer cientos de filas.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tc5061_proyecto_equipo18 import config

API = "mibici-api"
LOTE = "[lote]"
PERCENTILES = (50, 95, 99)

# Paleta (validada para daltonismo): un color por serie, en orden fijo; texto en tinta neutra
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
FONDO = "#fcfcfb"
COLOR_CONTENEDOR = {"mibici-api": COLORES[0], "mibici-mlflow": COLORES[1]}


# =============================================================================
# Lectura
# =============================================================================
def leer_peticiones(ruta: Path) -> pd.DataFrame:
    """``peticiones.csv`` con ``segundo`` (desde el inicio de la prueba)."""
    peticiones = pd.read_csv(ruta)
    if peticiones.empty:
        raise ValueError(f"{ruta} no tiene peticiones: la prueba no llego a correr")
    peticiones["fallo"] = peticiones["fallo"].astype(bool)
    peticiones["lote"] = peticiones["endpoint"].str.endswith(LOTE)
    peticiones["segundo"] = peticiones["tiempo"] - peticiones["tiempo"].min()
    return peticiones


def leer_recursos(ruta: Path, inicio: float) -> pd.DataFrame:
    """``recursos.csv`` con ``segundo`` relativo al ``inicio`` de la prueba (puede estar vacio)."""
    if not Path(ruta).exists():
        return pd.DataFrame(columns=["tiempo", "contenedor", "cpu_pct", "memoria_mib", "segundo"])
    recursos = pd.read_csv(ruta)
    recursos["segundo"] = recursos["tiempo"] - inicio
    return recursos


# =============================================================================
# Calculo
# =============================================================================
def percentiles(ms: pd.Series) -> dict:
    if ms.empty:
        return {f"p{p}_ms": float("nan") for p in PERCENTILES}
    valores = np.percentile(ms, PERCENTILES)
    return {f"p{p}_ms": round(float(v), 1) for p, v in zip(PERCENTILES, valores, strict=True)}


def por_segundo(peticiones: pd.DataFrame) -> pd.DataFrame:
    """Por cada segundo de la prueba: usuarios, peticiones, errores y percentiles de latencia."""
    con_segundo = peticiones.assign(s=peticiones["segundo"].astype(int))
    agrupado = con_segundo.groupby("s")
    serie = pd.DataFrame(
        {
            "usuarios": agrupado["usuarios"].max(),
            "rps": agrupado.size(),
            "predicciones_por_seg": agrupado["filas"].sum(),
            "errores_por_seg": agrupado["fallo"].sum(),
        }
    )
    tiempo_real = con_segundo[~con_segundo["lote"]].groupby("s")["ms"]
    for p in PERCENTILES:
        serie[f"p{p}_ms"] = tiempo_real.quantile(p / 100)
    serie["tasa_error"] = serie["errores_por_seg"] / serie["rps"]
    serie = serie.reset_index(names="segundo")
    # El ultimo segundo queda incompleto (la prueba termina a la mitad): no se grafica
    return serie.iloc[:-1] if len(serie) > 1 else serie


def cumple_slo(p95_ms: float, tasa_error: float) -> bool:
    return bool(p95_ms <= config.CARGA_SLO_P95_MS and tasa_error <= config.CARGA_SLO_ERRORES)


def por_escalon(
    peticiones: pd.DataFrame,
    recursos: pd.DataFrame,
    calentamiento: float = config.CARGA_CALENTAMIENTO_SEG,
    contenedor: str = API,
) -> pd.DataFrame:
    """Un renglon por escalon de usuarios, sin sus primeros ``calentamiento`` segundos."""
    filas = []
    for usuarios, grupo in peticiones.groupby("usuarios", sort=True):
        if usuarios <= 0:
            continue
        inicio, fin = grupo["segundo"].min() + calentamiento, grupo["segundo"].max()
        estable = grupo[grupo["segundo"] >= inicio]
        duracion = fin - inicio
        if estable.empty or duracion <= 0:
            continue
        tasa_error = float(estable["fallo"].mean())
        lotes = estable[estable["lote"]]
        fila = {
            "usuarios": int(usuarios),
            "segundos": round(float(duracion), 1),
            "peticiones": len(estable),
            "rps": round(len(estable) / duracion, 2),
            "predicciones_por_seg": round(float(estable["filas"].sum()) / duracion, 1),
            **percentiles(estable.loc[~estable["lote"], "ms"]),
            "p95_lote_ms": percentiles(lotes["ms"])["p95_ms"],
            "tasa_error": round(tasa_error, 5),
        }
        api = recursos[
            (recursos["contenedor"] == contenedor)
            & recursos["segundo"].between(inicio, fin, inclusive="both")
        ]
        fila["cpu_api_pct"] = round(float(api["cpu_pct"].mean()), 1) if len(api) else np.nan
        fila["cpu_api_max_pct"] = round(float(api["cpu_pct"].max()), 1) if len(api) else np.nan
        fila["memoria_api_max_mib"] = (
            round(float(api["memoria_mib"].max()), 1) if len(api) else np.nan
        )
        fila["cumple_slo"] = cumple_slo(fila["p95_ms"], tasa_error)
        filas.append(fila)
    return pd.DataFrame(filas)


def por_endpoint(peticiones: pd.DataFrame) -> list[dict]:
    """Totales por endpoint en toda la prueba."""
    duracion = max(peticiones["segundo"].max(), 1e-9)
    resultado = []
    for nombre, grupo in peticiones.groupby("endpoint", sort=True):
        resultado.append(
            {
                "endpoint": nombre,
                "peticiones": len(grupo),
                "errores": int(grupo["fallo"].sum()),
                "tasa_error": round(float(grupo["fallo"].mean()), 5),
                "rps": round(len(grupo) / duracion, 2),
                "filas_por_peticion": round(float(grupo.loc[~grupo["fallo"], "filas"].mean()), 1)
                if (~grupo["fallo"]).any()
                else 0.0,
                **percentiles(grupo["ms"]),
            }
        )
    return resultado


def capacidad(escalones: pd.DataFrame) -> dict | None:
    """El escalon con mas predicciones por segundo que cumple el objetivo de servicio (None si
    ninguno cumple)."""
    validos = escalones[escalones["cumple_slo"]] if len(escalones) else escalones
    if validos.empty:
        return None
    mejor = validos.loc[validos["predicciones_por_seg"].idxmax()]
    return {
        "usuarios": int(mejor["usuarios"]),
        "rps": float(mejor["rps"]),
        "predicciones_por_seg": float(mejor["predicciones_por_seg"]),
        "p95_ms": float(mejor["p95_ms"]),
        "tasa_error": float(mejor["tasa_error"]),
        "cpu_api_pct": None if pd.isna(mejor["cpu_api_pct"]) else float(mejor["cpu_api_pct"]),
    }


def recursos_por_contenedor(recursos: pd.DataFrame) -> dict:
    """CPU promedio y maximo y memoria maxima de cada contenedor durante la prueba."""
    resumen = {}
    for nombre, grupo in recursos.groupby("contenedor", sort=True):
        resumen[nombre] = {
            "muestras": len(grupo),
            "cpu_promedio_pct": round(float(grupo["cpu_pct"].mean()), 1),
            "cpu_max_pct": round(float(grupo["cpu_pct"].max()), 1),
            "memoria_max_mib": round(float(grupo["memoria_mib"].max()), 1),
            "limite_memoria_mib": round(float(grupo["limite_mib"].max()), 1),
        }
    return resumen


def sin_saturar(
    peticiones: pd.DataFrame, calentamiento: float = config.CARGA_CALENTAMIENTO_SEG
) -> pd.DataFrame:
    """Peticiones del escalon con menos usuarios, sin su calentamiento: el tiempo de servicio de
    cada endpoint sin cola (lo usa FinOps para estimar el cobro por peticion)."""
    activos = peticiones[peticiones["usuarios"] > 0]
    if activos.empty:
        return activos
    grupo = activos[activos["usuarios"] == activos["usuarios"].min()]
    return grupo[grupo["segundo"] >= grupo["segundo"].min() + calentamiento]


def resumir(peticiones: pd.DataFrame, recursos: pd.DataFrame, entorno: dict | None = None):
    """(resumen para JSON, escalones, serie por segundo)."""
    escalones = por_escalon(peticiones, recursos)
    serie = por_segundo(peticiones)
    duracion = float(peticiones["segundo"].max())
    resumen = {
        "duracion_seg": round(duracion, 1),
        "peticiones": len(peticiones),
        "errores": int(peticiones["fallo"].sum()),
        "tasa_error": round(float(peticiones["fallo"].mean()), 5),
        "rps_promedio": round(len(peticiones) / max(duracion, 1e-9), 2),
        "predicciones": int(peticiones["filas"].sum()),
        **percentiles(peticiones.loc[~peticiones["lote"], "ms"]),
        "slo": {"p95_ms": config.CARGA_SLO_P95_MS, "tasa_error": config.CARGA_SLO_ERRORES},
        "capacidad": capacidad(escalones),
        "por_endpoint": por_endpoint(peticiones),
        "por_endpoint_sin_saturar": por_endpoint(sin_saturar(peticiones)),
        "escalones": escalones.to_dict(orient="records"),
        "recursos": recursos_por_contenedor(recursos),
        "entorno": entorno or {},
    }
    return resumen, escalones, serie


# =============================================================================
# Graficas
# =============================================================================
def estilo(ax, titulo: str, unidad: str):
    ax.set_facecolor(FONDO)
    ax.set_title(titulo, loc="left", fontsize=10, color=TINTA)
    ax.set_ylabel(unidad, fontsize=8, color=TINTA_SECUNDARIA)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8, length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)


def _figura(paneles: int, alto: float = 2.0):
    figura, ejes = plt.subplots(
        paneles, 1, figsize=(9, alto * paneles + 0.6), sharex=True, facecolor=FONDO
    )
    return figura, np.atleast_1d(ejes)


def _panel_usuarios(ax, serie: pd.DataFrame):
    ax.step(serie["segundo"], serie["usuarios"], where="post", color=TINTA_SECUNDARIA, lw=1.5)
    estilo(ax, "Usuarios concurrentes", "usuarios")


def grafica_en_el_tiempo(serie: pd.DataFrame, ruta: Path):
    """Usuarios, peticiones por segundo, latencia (p50/p95/p99) y tasa de error en el tiempo."""
    figura, (usuarios, rps, latencia, errores) = _figura(4)
    _panel_usuarios(usuarios, serie)

    rps.plot(serie["segundo"], serie["rps"], color=COLORES[0], lw=1.5)
    rps.set_ylim(bottom=0)
    estilo(rps, "Peticiones atendidas por segundo (RPS)", "peticiones/s")

    for color, p in zip(COLORES, PERCENTILES, strict=False):
        latencia.plot(serie["segundo"], serie[f"p{p}_ms"], color=color, lw=1.5, label=f"p{p}")
    latencia.axhline(config.CARGA_SLO_P95_MS, color=TINTA_SECUNDARIA, lw=1, ls="--")
    latencia.annotate(
        f"objetivo p95 = {config.CARGA_SLO_P95_MS} ms",
        (serie["segundo"].max(), config.CARGA_SLO_P95_MS),
        xytext=(-2, 3),
        ha="right",
        textcoords="offset points",
        fontsize=7,
        color=TINTA_SECUNDARIA,
    )
    latencia.legend(frameon=False, fontsize=8, ncol=3, loc="center left")
    estilo(latencia, "Latencia de las peticiones en tiempo real (sin lotes)", "ms")

    errores.plot(serie["segundo"], 100 * serie["tasa_error"], color=COLORES[3], lw=1.5)
    errores.set_ylim(0, max(2.0, 110 * serie["tasa_error"].max()))  # escala visible sin errores
    estilo(errores, "Tasa de error", "% de peticiones")
    errores.set_xlabel("segundos desde el inicio de la prueba", fontsize=8, color=TINTA_SECUNDARIA)

    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def grafica_escalones(escalones: pd.DataFrame, ruta: Path):
    """Capacidad: RPS, predicciones por segundo y p95 por escalon, con el objetivo de servicio."""
    figura, (rps, predicciones, p95) = plt.subplots(1, 3, figsize=(11, 3.2), facecolor=FONDO)
    x = escalones["usuarios"]
    rps.plot(x, escalones["rps"], color=COLORES[0], lw=1.5, marker="o", ms=5)
    estilo(rps, "Peticiones por segundo", "peticiones/s")
    predicciones.plot(
        x, escalones["predicciones_por_seg"], color=COLORES[2], lw=1.5, marker="o", ms=5
    )
    estilo(predicciones, "Predicciones (filas) por segundo", "predicciones/s")
    p95.plot(x, escalones["p95_ms"], color=COLORES[1], lw=1.5, marker="o", ms=5)
    p95.axhline(config.CARGA_SLO_P95_MS, color=TINTA_SECUNDARIA, lw=1, ls="--")
    estilo(p95, f"p95 en tiempo real (objetivo {config.CARGA_SLO_P95_MS} ms)", "ms")
    for ax in (rps, predicciones, p95):
        ax.set_ylim(bottom=0)
        ax.set_xlabel("usuarios concurrentes", fontsize=8, color=TINTA_SECUNDARIA)
        ax.set_xticks(x)
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


def grafica_recursos(recursos: pd.DataFrame, serie: pd.DataFrame, ruta: Path):
    """CPU y memoria de cada contenedor durante la prueba, con los usuarios como referencia."""
    figura, (usuarios, cpu, memoria) = _figura(3)
    _panel_usuarios(usuarios, serie)
    for nombre, grupo in recursos.groupby("contenedor", sort=True):
        color = COLOR_CONTENEDOR.get(nombre, COLORES[2])
        cpu.plot(grupo["segundo"], grupo["cpu_pct"], color=color, lw=1.5, label=nombre)
        memoria.plot(grupo["segundo"], grupo["memoria_mib"], color=color, lw=1.5, label=nombre)
    cpu.axhline(100, color=TINTA_SECUNDARIA, lw=1, ls="--")
    cpu.annotate(
        "1 vCPU", (0, 100), xytext=(2, 3), textcoords="offset points", fontsize=7,
        color=TINTA_SECUNDARIA,
    )  # fmt: skip
    estilo(cpu, "CPU (docker stats)", "% de 1 vCPU")
    estilo(memoria, "Memoria (docker stats)", "MiB")
    for ax in (cpu, memoria):
        if recursos["contenedor"].nunique() > 1:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    memoria.set_ylim(bottom=0)
    memoria.set_xlabel("segundos desde el inicio de la prueba", fontsize=8, color=TINTA_SECUNDARIA)
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)


# =============================================================================
# Reporte
# =============================================================================
def tabla(filas: list[dict], columnas: list[tuple[str, str]]) -> str:
    def celda(valor):
        if isinstance(valor, bool):
            return "si" if valor else "no"
        if isinstance(valor, int):
            return f"{valor:,}"
        if isinstance(valor, float):
            return (
                "-" if np.isnan(valor) else f"{valor:,.4g}" if abs(valor) < 1 else f"{valor:,.1f}"
            )
        return str(valor)

    encabezado = "| " + " | ".join(titulo for _, titulo in columnas) + " |"
    separador = "|" + "|".join("---" for _ in columnas) + "|"
    cuerpo = ["| " + " | ".join(celda(f[c]) for c, _ in columnas) + " |" for f in filas]
    return "\n".join([encabezado, separador, *cuerpo])


def markdown(resumen: dict, figuras: dict[str, Path], destino: Path) -> str:
    """Resumen de la corrida en Markdown, con rutas relativas a ``destino``."""
    entorno = resumen["entorno"]
    capacidad_ = resumen["capacidad"]
    relativa = {clave: Path(ruta).relative_to(destino.parent, walk_up=True).as_posix()
                for clave, ruta in figuras.items()}  # fmt: skip
    lineas = [
        "# Prueba de carga de la API (Locust)",
        "",
        "Generado por `make carga` (`python -m carga.ejecutar`). No editar a mano.",
        "",
    ]
    if entorno.get("modelos_de_prueba"):
        lineas += [
            "> **Corrida con modelos de prueba** (`scripts/modelos_de_prueba.py`), no con los",
            "> champions: sirve para probar el procedimiento, no para dimensionar el servicio.",
            "",
        ]
    lineas += [
        f"- Fecha: {entorno.get('fecha', '-')}; API: `{entorno.get('url', '-')}`",
        f"- Modelos: {entorno.get('modelos_texto', '-')}",
        (
            f"- Contenedor de la API: {entorno.get('limites_api', '-')}; "
            f"filas por peticion: {entorno.get('filas_por_peticion', '-')}"
        ),
        (
            f"- Duracion: {resumen['duracion_seg']:.0f} s; peticiones: {resumen['peticiones']:,} "
            f"({resumen['predicciones']:,} predicciones); errores: {resumen['errores']:,} "
            f"({100 * resumen['tasa_error']:.2f} %)"
        ),
        (
            f"- Latencia en tiempo real: p50 {resumen['p50_ms']} ms, p95 {resumen['p95_ms']} ms, "
            f"p99 {resumen['p99_ms']} ms"
        ),
        (
            f"- Objetivo de servicio: p95 <= {resumen['slo']['p95_ms']} ms y errores <= "
            f"{100 * resumen['slo']['tasa_error']:.0f} %"
        ),
    ]
    if capacidad_:
        lineas.append(
            f"- **Capacidad** (escalon con mas predicciones por segundo que cumple el objetivo): "
            f"**{capacidad_['predicciones_por_seg']:,.0f} predicciones/s** "
            f"({capacidad_['rps']:.1f} peticiones/s) con {capacidad_['usuarios']} usuarios "
            f"(p95 {capacidad_['p95_ms']:.0f} ms)"
        )
    else:
        lineas.append("- **Capacidad:** ningun escalon cumplio el objetivo de servicio")
    lineas += [
        "",
        "## Por escalon de usuarios",
        "",
        "p50, p95 y p99: peticiones en tiempo real (P2 y /salud); los lotes van aparte.",
        "",
        tabla(
            resumen["escalones"],
            [
                ("usuarios", "Usuarios"),
                ("rps", "RPS"),
                ("predicciones_por_seg", "Predicciones/s"),
                ("p50_ms", "p50 (ms)"),
                ("p95_ms", "p95 (ms)"),
                ("p99_ms", "p99 (ms)"),
                ("p95_lote_ms", "p95 lotes (ms)"),
                ("tasa_error", "Tasa de error"),
                ("cpu_api_pct", "CPU API (%)"),
                ("memoria_api_max_mib", "Memoria API max (MiB)"),
                ("cumple_slo", "Cumple"),
            ],
        ),
        "",
        "## Por endpoint (toda la prueba)",
        "",
        tabla(
            resumen["por_endpoint"],
            [
                ("endpoint", "Endpoint"),
                ("peticiones", "Peticiones"),
                ("filas_por_peticion", "Filas por peticion"),
                ("rps", "RPS"),
                ("p50_ms", "p50 (ms)"),
                ("p95_ms", "p95 (ms)"),
                ("p99_ms", "p99 (ms)"),
                ("tasa_error", "Tasa de error"),
            ],
        ),
        "",
        "## Recursos (docker stats)",
        "",
        tabla(
            [{"contenedor": k, **v} for k, v in resumen["recursos"].items()],
            [
                ("contenedor", "Contenedor"),
                ("cpu_promedio_pct", "CPU promedio (%)"),
                ("cpu_max_pct", "CPU max (%)"),
                ("memoria_max_mib", "Memoria max (MiB)"),
                ("limite_memoria_mib", "Limite (MiB)"),
            ],
        )
        if resumen["recursos"]
        else "Sin muestras de docker stats (ningun contenedor de `config.CARGA_CONTENEDORES`).",
        "",
        "## Graficas",
        "",
    ]
    titulos = {
        "en_el_tiempo": "Usuarios, RPS, latencia y errores en el tiempo",
        "escalones": "Capacidad por escalon",
        "recursos": "CPU y memoria de los contenedores",
    }
    for clave, ruta in relativa.items():
        lineas += [f"![{titulos.get(clave, clave)}]({ruta})", ""]
    return "\n".join(lineas)


def sin_nan(valor):
    """NaN -> None (JSON valido) y tipos de numpy -> tipos de Python, recursivamente."""
    if isinstance(valor, dict):
        return {clave: sin_nan(v) for clave, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [sin_nan(v) for v in valor]
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float) and np.isnan(valor):
        return None
    return valor


def generar(
    peticiones_csv: Path,
    recursos_csv: Path,
    entorno: dict | None = None,
    destino_dir: Path = config.CARGA_DIR,
    figuras_dir: Path = config.CARGA_FIGURAS_DIR,
) -> dict:
    """Lee los CSV de la corrida y escribe escalones, recursos, resumen y graficas."""
    destino_dir, figuras_dir = Path(destino_dir), Path(figuras_dir)
    destino_dir.mkdir(parents=True, exist_ok=True)
    figuras_dir.mkdir(parents=True, exist_ok=True)

    peticiones = leer_peticiones(peticiones_csv)
    recursos = leer_recursos(recursos_csv, inicio=float(peticiones["tiempo"].min()))
    resumen, escalones, serie = resumir(peticiones, recursos, entorno)

    escalones.to_csv(destino_dir / config.CARGA_ESCALONES_CSV.name, index=False)
    recursos.drop(columns="segundo").to_csv(
        destino_dir / config.CARGA_RECURSOS_CSV.name, index=False
    )
    figuras = {"en_el_tiempo": figuras_dir / "carga_en_el_tiempo.png"}
    grafica_en_el_tiempo(serie, figuras["en_el_tiempo"])
    if len(escalones):
        figuras["escalones"] = figuras_dir / "capacidad_por_escalon.png"
        grafica_escalones(escalones, figuras["escalones"])
    if len(recursos):
        figuras["recursos"] = figuras_dir / "recursos_contenedores.png"
        grafica_recursos(recursos, serie, figuras["recursos"])
    resumen["figuras"] = {
        clave: Path(ruta).relative_to(config.PROJ_ROOT, walk_up=True).as_posix()
        for clave, ruta in figuras.items()
    }

    destino_json = destino_dir / config.CARGA_RESUMEN_JSON.name
    destino_json.write_text(
        json.dumps(sin_nan(resumen), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    destino_md = destino_dir / "resumen_carga.md"
    destino_md.write_text(markdown(resumen, figuras, destino_md) + "\n", encoding="utf-8")
    return resumen
