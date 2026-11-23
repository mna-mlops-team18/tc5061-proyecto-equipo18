"""Analisis FinOps: costo por inferencia de la API a partir de la prueba de carga.

Lee ``reports/carga/resumen_carga.json`` (lo deja ``make carga``) y estima:

1. **Costo de la instancia** de 1 vCPU y 1 GiB (el limite de la API en ``docker-compose.yml``) con
   los precios de referencia de ``config.FINOPS_PRECIOS``.
2. **Costo por inferencia a plena capacidad:** precio por hora entre las predicciones por hora que
   la instancia sostiene cumpliendo el objetivo de servicio. Es el piso: solo se alcanza si la
   instancia esta ocupada todo el tiempo.
3. **Costo con la demanda esperada de MiBici** (``config.FINOPS_*``), con la instancia siempre
   encendida o con cobro por peticion (escala a cero). Con cobro por peticion, cada peticion paga
   su tiempo de servicio sin cola (p50 del escalon mas bajo), redondeado a 100 ms, sin suponer que
   varias peticiones comparten el tiempo facturable (estimacion pesimista).
4. **Evidencia para optimizar:** milisegundos por prediccion segun el tamano de la peticion,
   segundos de CPU por prediccion (``docker stats``), utilizacion y memoria usada contra la
   asignada.

Escribe ``reports/carga/finops.json`` y la grafica de costo mensual
(``reports/figures/carga/costo_mensual.png``), insumos del reporte de la semana 10 (en PDF, fuera
del repositorio).

Uso, despues de ``make carga``::

    make finops
"""

import json
import math
from pathlib import Path

from loguru import logger
import typer

from carga import analisis
from tc5061_proyecto_equipo18 import config

app = typer.Typer(add_completion=False)
SIEMPRE_ENCENDIDA = ("cloud_run_instancia", "fargate")
POR_PETICION = "cloud_run_peticion"


# =============================================================================
# Precios y demanda
# =============================================================================
def precio_por_seg(precio: dict) -> float:
    """USD por segundo de una instancia con los vCPU y GiB del precio de referencia."""
    return precio["vcpu"] * precio["vcpu_seg"] + precio["gib"] * precio["gib_seg"]


def demanda_mensual(
    estaciones: int = config.FINOPS_ESTACIONES,
    despachadores: int = config.FINOPS_DESPACHADORES,
    consultas_por_min: float = config.FINOPS_CONSULTAS_POR_MIN,
    horas_operacion: float = config.FINOPS_HORAS_OPERACION,
    dias: int = config.FINOPS_DIAS_MES,
    filas: dict = config.CARGA_FILAS,
) -> dict:
    """Peticiones y predicciones al mes por pregunta, segun la estrategia de despliegue."""
    filas_p1 = estaciones * 24  # estaciones x horas del dia siguiente
    peticiones_p2 = despachadores * consultas_por_min * 60 * horas_operacion * dias
    demanda = {
        "p1": {
            "peticiones": math.ceil(filas_p1 / filas["p1"]) * dias,
            "predicciones": filas_p1 * dias,
            "modo": "lote nocturno",
        },
        "p2": {
            "peticiones": round(peticiones_p2),
            "predicciones": round(peticiones_p2 * filas["p2"]),
            "modo": "tiempo real",
        },
        "p3": {"peticiones": 1, "predicciones": estaciones, "modo": "lote mensual"},
    }
    demanda["total"] = {
        "peticiones": sum(d["peticiones"] for d in demanda.values()),
        "predicciones": sum(d["predicciones"] for d in demanda.values()),
    }
    # Pico en tiempo real: todos los despachadores consultando a la vez en horario de operacion
    demanda["pico_predicciones_por_seg"] = despachadores * consultas_por_min / 60 * filas["p2"]
    return demanda


# =============================================================================
# Medidas de la prueba de carga
# =============================================================================
def servicio_sin_cola(resumen: dict) -> dict[str, dict]:
    """Por endpoint: p50 (ms) y filas por peticion en el escalon sin saturar."""
    return {
        e["endpoint"]: {"p50_ms": e["p50_ms"], "filas": e["filas_por_peticion"]}
        for e in resumen.get("por_endpoint_sin_saturar") or resumen["por_endpoint"]
    }


def ms_por_prediccion(servicio: dict[str, dict]) -> dict[str, float]:
    """Milisegundos de servicio por prediccion segun el endpoint (y por tanto, el tamano)."""
    return {
        endpoint: round(datos["p50_ms"] / datos["filas"], 4)
        for endpoint, datos in servicio.items()
        if datos["filas"] and datos["p50_ms"] is not None
    }


def cpu_seg_por_prediccion(escalones: list[dict], cpu_max: float = 80.0) -> float | None:
    """Segundos de vCPU por prediccion en los escalones que cumplen sin saturar la CPU (mediana)."""
    valores = sorted(
        e["cpu_api_pct"] / 100 / e["predicciones_por_seg"]
        for e in escalones
        if e.get("cumple_slo")
        and e.get("cpu_api_pct") is not None
        and e["cpu_api_pct"] < cpu_max
        and e["predicciones_por_seg"] > 0
    )
    return valores[len(valores) // 2] if valores else None


def memoria_sugerida(memoria_max_mib: float | None, margen: float = 1.5) -> int | None:
    """Memoria a asignar: lo usado con 50 % de margen, en multiplos de 256 MiB (minimo 512)."""
    if not memoria_max_mib:
        return None
    return max(512, math.ceil(memoria_max_mib * margen / 256) * 256)


def tiempo_facturable(ms: float, redondeo: float = config.FINOPS_REDONDEO_SEG) -> float:
    """Segundos que se cobran por una peticion de ``ms`` con cobro por peticion."""
    return max(1, math.ceil(ms / 1000 / redondeo - 1e-9)) * redondeo


def endpoint_de(pregunta: str, servicio: dict) -> str:
    """El endpoint medido de cada pregunta (los lotes llevan el sufijo ``[lote]``)."""
    for nombre in (f"/predicciones/{pregunta}", f"/predicciones/{pregunta} {analisis.LOTE}"):
        if nombre in servicio:
            return nombre
    raise KeyError(f"La prueba de carga no tiene peticiones de {pregunta}")


# =============================================================================
# Costos
# =============================================================================
def costos(resumen: dict, demanda: dict, precios: dict = config.FINOPS_PRECIOS) -> dict:
    """Costo por hora, por millon de predicciones a capacidad y al mes con la demanda."""
    capacidad = resumen["capacidad"]
    if capacidad is None:
        raise ValueError("Ningun escalon cumplio el objetivo de servicio: no hay capacidad medida")
    pred_por_seg = capacidad["predicciones_por_seg"]
    servicio = servicio_sin_cola(resumen)
    segundos_mes = config.FINOPS_DIAS_MES * 24 * 3600
    resultado = {}
    for clave, precio in precios.items():
        por_seg = precio_por_seg(precio)
        fila = {"nombre": precio["nombre"], "usd_hora": round(por_seg * 3600, 5)}
        if clave == POR_PETICION:
            # Cada tipo de peticion paga su tiempo de servicio redondeado y la cuota por peticion
            mensual = 0.0
            for pregunta in ("p1", "p2", "p3"):
                ms = servicio[endpoint_de(pregunta, servicio)]["p50_ms"]
                peticiones = demanda[pregunta]["peticiones"]
                mensual += peticiones * tiempo_facturable(ms) * por_seg
            mensual += demanda["total"]["peticiones"] / 1e6 * precio["por_millon_peticiones"]
            ms_p2 = servicio[endpoint_de("p2", servicio)]
            fila["usd_millon_predicciones_capacidad"] = round(
                (
                    tiempo_facturable(ms_p2["p50_ms"]) * por_seg
                    + precio["por_millon_peticiones"] / 1e6
                )
                / ms_p2["filas"]
                * 1e6,
                4,
            )
        else:
            mensual = por_seg * segundos_mes  # una instancia encendida todo el mes
            fila["usd_millon_predicciones_capacidad"] = round(por_seg / pred_por_seg * 1e6, 4)
        fila["usd_mes"] = round(mensual, 2)
        fila["usd_millon_predicciones_demanda"] = round(
            mensual / demanda["total"]["predicciones"] * 1e6, 4
        )
        resultado[clave] = fila
    return resultado


def analizar(resumen: dict, demanda: dict | None = None) -> dict:
    """Todo el analisis FinOps a partir del resumen de la prueba de carga."""
    demanda = demanda or demanda_mensual()
    capacidad = resumen["capacidad"]
    servicio = servicio_sin_cola(resumen)
    api = resumen.get("recursos", {}).get(analisis.API, {})
    return {
        "capacidad": capacidad,
        "demanda": demanda,
        "utilizacion_pico": round(
            demanda["pico_predicciones_por_seg"] / capacidad["predicciones_por_seg"], 6
        ),
        "costos": costos(resumen, demanda),
        "servicio_sin_cola": servicio,
        "ms_por_prediccion": ms_por_prediccion(servicio),
        "cpu_seg_por_prediccion": cpu_seg_por_prediccion(resumen["escalones"]),
        "memoria_api_max_mib": api.get("memoria_max_mib"),
        "memoria_api_limite_mib": api.get("limite_memoria_mib"),
        "memoria_sugerida_mib": memoria_sugerida(api.get("memoria_max_mib")),
        # La CPU satura si algun escalon la tuvo al 90 % o mas: entonces la capacidad es el limite
        # real de la instancia; si no, la instancia aguanta mas que el ultimo escalon
        "cpu_saturada": any((e.get("cpu_api_pct") or 0) >= 90 for e in resumen["escalones"]),
        "modelos_de_prueba": bool(resumen.get("entorno", {}).get("modelos_de_prueba")),
    }


# =============================================================================
# Grafica
# =============================================================================
def grafica_costo_mensual(finops: dict, ruta: Path):
    """Costo mensual de cada opcion con la demanda esperada (barras horizontales)."""
    filas = sorted(finops["costos"].values(), key=lambda f: f["usd_mes"])
    figura, ax = analisis.plt.subplots(
        figsize=(9, 0.6 * len(filas) + 1.2), facecolor=analisis.FONDO
    )
    posiciones = range(len(filas))
    ax.barh(posiciones, [f["usd_mes"] for f in filas], color=analisis.COLORES[0], height=0.5)
    for y, fila in zip(posiciones, filas, strict=True):
        ax.annotate(
            f"{fila['usd_mes']:,.2f} USD",
            (fila["usd_mes"], y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=analisis.TINTA,
        )
    ax.set_yticks(list(posiciones), [f["nombre"] for f in filas], fontsize=8)
    ax.set_xlim(0, max(f["usd_mes"] for f in filas) * 1.25)
    analisis.estilo(ax, "Costo mensual con la demanda esperada de MiBici", "")
    ax.grid(axis="x", color=analisis.REJILLA, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)  # la rejilla detras de las barras
    ax.set_xlabel("USD al mes", fontsize=8, color=analisis.TINTA_SECUNDARIA)
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    analisis.plt.close(figura)


@app.command()
def main(resumen_json: Path = config.CARGA_RESUMEN_JSON):
    """Costo por inferencia a partir de ``resumen_json`` (salida de ``make carga``)."""
    if not resumen_json.exists():
        logger.error(f"No existe {resumen_json}: corre primero `make carga`")
        raise typer.Exit(1)
    resumen = json.loads(resumen_json.read_text(encoding="utf-8"))
    finops = analizar(resumen)
    config.FINOPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.FINOPS_JSON.write_text(
        json.dumps(analisis.sin_nan(finops), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    config.CARGA_FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    figura = config.CARGA_FIGURAS_DIR / "costo_mensual.png"
    grafica_costo_mensual(finops, figura)
    for fila in finops["costos"].values():
        logger.info(
            f"{fila['nombre']}: {fila['usd_mes']:,.2f} USD/mes; "
            f"{fila['usd_millon_predicciones_capacidad']:.4f} USD por millon a capacidad"
        )
    if finops["modelos_de_prueba"]:
        logger.warning("Calculado con los modelos de prueba: no sirve para el reporte final")
    logger.info(
        f"Utilizacion en el pico: {100 * finops['utilizacion_pico']:.3f} % de la capacidad; "
        f"memoria sugerida: {finops['memoria_sugerida_mib']} MiB"
    )
    logger.success(f"Analisis en {config.FINOPS_JSON} y grafica en {figura}")


if __name__ == "__main__":
    app()
