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

Escribe ``reports/carga/finops.json``, la grafica de costo mensual y el reporte completo de la
semana 10 (``reports/reporte_carga_finops.md``), que integra la prueba de carga y este analisis.

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
# Grafica y reporte
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


def texto_escalones(escalones) -> str:
    """``[[10, 40], [50, 40]]`` -> ``'10 y 50 usuarios, 40 s cada uno'``."""
    if not escalones:
        return "-"
    usuarios = [str(u) for u, _ in escalones]
    segundos = {s for _, s in escalones}
    lista = ", ".join(usuarios[:-1]) + " y " + usuarios[-1] if len(usuarios) > 1 else usuarios[0]
    duracion = f"{segundos.pop()} s cada uno" if len(segundos) == 1 else "duracion variable"
    return f"{lista} usuarios, {duracion}"


def _ruta(ruta: str | Path, desde: Path) -> str:
    return Path(config.PROJ_ROOT / ruta).relative_to(desde.parent, walk_up=True).as_posix()


def reporte(resumen: dict, finops: dict, figura_costos: Path, destino: Path) -> str:
    """Reporte de la semana 10: prueba de carga, recursos y FinOps, en Markdown."""
    entorno = resumen["entorno"]
    capacidad = finops["capacidad"]
    demanda = finops["demanda"]
    costos_ = finops["costos"]
    encendida = costos_["cloud_run_instancia"]
    por_peticion = costos_[POR_PETICION]
    figuras = {clave: _ruta(ruta, destino) for clave, ruta in resumen.get("figuras", {}).items()}
    ms_pred = finops["ms_por_prediccion"]
    p2, lote_p1 = "/predicciones/p2", f"/predicciones/p1 {analisis.LOTE}"
    lineas = [
        "# Prueba de carga y analisis FinOps de la API (semana 10)",
        "",
        "Generado por `make carga` y `make finops` (`carga/`). No editar a mano: se regenera con",
        "cada corrida.",
        "",
    ]
    if finops["modelos_de_prueba"]:
        lineas += [
            "> **Corrida con modelos de prueba**, no con los champions. Las cifras sirven para",
            "> probar el procedimiento; para el reporte final, repetir con `make export-model`,",
            "> `make up`, `make carga` y `make finops`.",
            "",
        ]
    lineas += [
        "## 1. Que se midio",
        "",
        f"- Fecha: {entorno.get('fecha', '-')}. Modelos: {entorno.get('modelos_texto', '-')}.",
        f"- Instancia: contenedor de la API con {entorno.get('limites_api', '-')}.",
        (
            f"- Maquina: {entorno.get('maquina', {}).get('sistema', '-')}, "
            f"{entorno.get('maquina', {}).get('cpus', '-')} CPU. Locust corre en la misma maquina."
        ),
        "- Escenario (README > Prueba de carga): despachadores que piden el riesgo de P2 de 20",
        "  estaciones en tiempo real, mas un proceso que manda lotes de P1 (1000 filas) y P3",
        f"  (470 estaciones). Escalones: {texto_escalones(entorno.get('escalones'))}.",
        f"- Objetivo de servicio: p95 en tiempo real <= {resumen['slo']['p95_ms']} ms y errores",
        f"  <= {100 * resumen['slo']['tasa_error']:.0f} %.",
        "",
        "## 2. Resultados de la prueba de carga",
        "",
        (
            f"- {resumen['peticiones']:,} peticiones y {resumen['predicciones']:,} predicciones en "
            f"{resumen['duracion_seg']:.0f} s; {resumen['errores']:,} errores "
            f"({100 * resumen['tasa_error']:.2f} %)."
        ),
        (
            f"- **Capacidad de la instancia:** {capacidad['predicciones_por_seg']:,.0f} predicciones/s "
            f"({capacidad['rps']:.1f} peticiones/s) con {capacidad['usuarios']} usuarios, p95 de "
            f"{capacidad['p95_ms']:.0f} ms."
        ),
        "",
        analisis.tabla(
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
                ("memoria_api_max_mib", "Memoria API (MiB)"),
                ("cumple_slo", "Cumple"),
            ],
        ),
        "",
    ]
    if "en_el_tiempo" in figuras:
        lineas += [f"![Usuarios, RPS, latencia y errores]({figuras['en_el_tiempo']})", ""]
    if "escalones" in figuras:
        lineas += [f"![Capacidad por escalon]({figuras['escalones']})", ""]
    lineas += [
        "Por endpoint, en toda la prueba:",
        "",
        analisis.tabla(
            resumen["por_endpoint"],
            [
                ("endpoint", "Endpoint"),
                ("peticiones", "Peticiones"),
                ("filas_por_peticion", "Filas"),
                ("p50_ms", "p50 (ms)"),
                ("p95_ms", "p95 (ms)"),
                ("p99_ms", "p99 (ms)"),
                ("tasa_error", "Tasa de error"),
            ],
        ),
        "",
        "## 3. Uso de recursos (docker stats)",
        "",
    ]
    if resumen["recursos"]:
        lineas += [
            analisis.tabla(
                [{"contenedor": k, **v} for k, v in resumen["recursos"].items()],
                [
                    ("contenedor", "Contenedor"),
                    ("cpu_promedio_pct", "CPU promedio (%)"),
                    ("cpu_max_pct", "CPU max (%)"),
                    ("memoria_max_mib", "Memoria max (MiB)"),
                    ("limite_memoria_mib", "Limite (MiB)"),
                ],
            ),
            "",
        ]
        if "recursos" in figuras:
            lineas += [f"![CPU y memoria]({figuras['recursos']})", ""]
    else:
        lineas += ["Sin muestras de docker stats.", ""]
    cpu_pred = finops["cpu_seg_por_prediccion"]
    lineas += [
        (
            "- La CPU es el recurso que limita: la latencia se dispara cuando la API llega a su "
            "vCPU."
            if finops["cpu_saturada"]
            else "- La CPU no llego a saturarse: la instancia aguanta mas que el ultimo escalon. "
            "Para medir su limite, agregar escalones (`CARGA_ESCALONES`)."
        ),
        (
            f"- Memoria maxima de la API: {finops['memoria_api_max_mib'] or '-'} MiB de "
            f"{finops['memoria_api_limite_mib'] or '-'} MiB asignados."
        ),
        "- CPU por prediccion (escalones que cumplen, con CPU < 80 %): "
        + (f"{1000 * cpu_pred:.3f} ms de vCPU." if cpu_pred else "sin dato."),
        "",
        "## 4. FinOps: costo por inferencia",
        "",
        "Precios de lista en USD (`config.FINOPS_PRECIOS`, consultados el 2026-11-22, sin capa",
        'gratuita). Una inferencia = una prediccion (una fila). "A plena capacidad" = la',
        "instancia ocupada todo el tiempo; con cobro por peticion, cada peticion de P2 (20 filas)",
        "paga su tiempo de servicio redondeado a 100 ms mas la cuota por peticion.",
        "",
        (
            "| Opcion | USD/hora | USD por millon de predicciones a plena capacidad | USD al mes con "
            "la demanda esperada | USD por millon de predicciones con la demanda |"
        ),
        "|---|---|---|---|---|",
    ]
    for fila in costos_.values():
        lineas.append(
            f"| {fila['nombre']} | {fila['usd_hora']:.4f} | "
            f"{fila['usd_millon_predicciones_capacidad']:.4f} | {fila['usd_mes']:,.2f} | "
            f"{fila['usd_millon_predicciones_demanda']:.4f} |"
        )
    lineas += [
        "",
        "**Demanda esperada de MiBici** (supuestos en `config.FINOPS_*`, segun la estrategia de",
        "despliegue):",
        "",
        "| Pregunta | Modo | Peticiones al mes | Predicciones al mes |",
        "|---|---|---|---|",
    ]
    for pregunta in ("p1", "p2", "p3"):
        d = demanda[pregunta]
        lineas.append(
            f"| {pregunta.upper()} | {d['modo']} | {d['peticiones']:,} | {d['predicciones']:,} |"
        )
    lineas += [
        f"| Total | | {demanda['total']['peticiones']:,} | {demanda['total']['predicciones']:,} |",
        "",
        f"![Costo mensual por opcion]({_ruta(figura_costos, destino)})",
        "",
        "### Lectura",
        "",
        (
            f"- **A plena capacidad, una prediccion cuesta "
            f"{encendida['usd_millon_predicciones_capacidad']:.4f} USD por millon** en Cloud Run "
            "siempre encendida. Es el piso: solo se paga eso si la instancia esta ocupada todo el "
            "tiempo."
        ),
        (
            f"- **Con la demanda esperada, la instancia casi nunca trabaja.** El pico en tiempo real "
            f"es de {demanda['pico_predicciones_por_seg']:,.1f} predicciones/s, el "
            f"{100 * finops['utilizacion_pico']:.3f} % de la capacidad. Por eso, siempre encendida, "
            f"el costo efectivo sube a {encendida['usd_millon_predicciones_demanda']:.4f} USD por "
            "millon: se paga sobre todo tiempo ocioso."
        ),
        (
            f"- **Cobrar por peticion (escalar a cero) baja el mes de {encendida['usd_mes']:,.2f} a "
            f"{por_peticion['usd_mes']:,.2f} USD** con la misma demanda, aun suponiendo que ninguna "
            "peticion comparte tiempo facturable."
        ),
        "",
        "## 5. Como optimizarlo",
        "",
        "1. **Escalar a cero con cobro por peticion** (Cloud Run con `min-instances=0`, o un",
        "   servicio equivalente). Es el mayor ahorro con esta demanda (seccion 4). El costo es el",
        "   arranque en frio: la imagen carga los tres modelos al iniciar (unos segundos, ver",
        "   `start_period` del healthcheck). Para P2, que tiene un despachador esperando, se puede",
        "   dejar `min-instances=1` solo en el horario de operacion y escalar a cero de noche.",
        "2. **Agrupar filas por peticion.** El tiempo de servicio por prediccion baja mucho con",
        "   peticiones mas grandes: "
        + (
            f"{ms_pred[p2]:.3f} ms por prediccion en P2 (20 filas) contra "
            f"{ms_pred[lote_p1]:.4f} ms en los lotes de P1 (1000 filas)."
            if p2 in ms_pred and lote_p1 in ms_pred
            else "ver `ms_por_prediccion` en `finops.json`."
        ),
        "   P1 y P3 ya van por lotes; en P2 conviene que el tablero pida todas las estaciones de la",
        "   zona en una sola peticion, no una por estacion (`CARGA_FILAS_P2=1 make carga` mide el",
        "   caso sin agrupar).",
        (
            "3. **Ajustar el tamano de la instancia.** La API usa como maximo "
            f"{finops['memoria_api_max_mib'] or '-'} MiB de {finops['memoria_api_limite_mib'] or '-'}"
            f" MiB asignados: con {finops['memoria_sugerida_mib'] or '-'} MiB (50 % de margen) baja"
        ),
        "   la parte de memoria del precio. En Fargate, 1 vCPU exige al menos 2 GB; si la latencia",
        "   lo permite, 0.5 vCPU con 1 GB cuesta menos de la mitad.",
        "4. **Precalcular P1 y P3.** Son lotes con resultado fijo para el dia o el mes: se pueden",
        "   correr como un trabajo programado (Cloud Run Jobs, AWS Batch) que guarde el resultado,",
        "   sin un servicio encendido. La API queda solo para P2.",
        "5. **Sacar MLflow del camino de servicio.** La API sirve los modelos copiados en la imagen",
        "   (opcion B): el servidor de MLflow no tiene que estar encendido para predecir, solo para",
        "   registrar y exportar modelos.",
        "6. **Mas workers solo si la CPU lo justifica.** Un worker de uvicorn por vCPU; en una",
        "   instancia de 1 vCPU, mas workers suben la memoria sin aumentar la capacidad.",
        "7. **Descuentos de la nube.** Instancias spot o compromisos de uso de 1 a 3 anos bajan el",
        "   precio por hora si la instancia debe quedarse encendida; no cambian la baja utilizacion.",
        "",
        "## 6. Limitaciones",
        "",
        "- Locust y la API corren en la misma maquina y compiten por CPU; en la nube el generador",
        "  de carga iria aparte.",
        "- `docker stats` mide el contenedor en Docker Desktop, no una instancia de la nube; la vCPU",
        "  de la nube puede rendir distinto.",
        "- La demanda es un supuesto razonable, no un dato medido de MiBici.",
        "- El cobro por peticion supone que las peticiones no comparten tiempo facturable; con",
        "  concurrencia real seria menor.",
        "- Los precios son de lista y cambian; no incluyen red, almacenamiento de la imagen ni",
        "  capa gratuita.",
    ]
    return "\n".join(lineas)


@app.command()
def main(resumen_json: Path = config.CARGA_RESUMEN_JSON):
    """Costo por inferencia y reporte de la semana 10 a partir de ``resumen_json``."""
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
    destino = config.REPORTE_CARGA_FINOPS_MD
    destino.write_text(reporte(resumen, finops, figura, destino) + "\n", encoding="utf-8")
    for fila in finops["costos"].values():
        logger.info(
            f"{fila['nombre']}: {fila['usd_mes']:,.2f} USD/mes; "
            f"{fila['usd_millon_predicciones_capacidad']:.4f} USD por millon a capacidad"
        )
    if finops["modelos_de_prueba"]:
        logger.warning("Calculado con los modelos de prueba: no sirve para el reporte final")
    logger.success(f"Reporte en {destino}")


if __name__ == "__main__":
    app()
