"""Corre la prueba de carga completa: Locust contra la API y ``docker stats`` al mismo tiempo.

Pasos:

1. Revisa que la API responda ``GET /salud`` con 200 y anota que modelos sirve (``GET /modelo``)
   y el tamano de su contenedor (``docker inspect``).
2. Arranca el muestreo de CPU y memoria de ``config.CARGA_CONTENEDORES`` (``carga.recursos``).
3. Corre Locust sin interfaz con el escenario de ``carga/locustfile.py`` (escalones de usuarios
   de ``config.CARGA_ESCALONES``). Locust deja su reporte HTML y sus CSV en
   ``reports/carga/crudos/``.
4. Detiene el muestreo y genera el resumen, los escalones y las graficas (``carga.analisis``).

Uso, con el sistema arriba (``make up``, o ``make run-api`` para solo la API)::

    make carga
    python -m carga.ejecutar --url http://127.0.0.1:8000
    CARGA_ESCALONES=5:15,20:15 make carga     # corrida corta para probar
"""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess  # nosec B404  # corre locust y docker con argumentos fijos, sin shell
import sys
import time
import urllib.error
import urllib.request

from loguru import logger
import typer

from carga import analisis, recursos
from tc5061_proyecto_equipo18 import config

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
app = typer.Typer(add_completion=False)


def consultar(url: str, timeout: float = 10) -> tuple[int, object]:
    """(codigo HTTP, JSON) de un GET; codigo 0 si la API no responde."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL no soportada: {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as respuesta:  # nosec B310  # http(s)
            return respuesta.status, json.loads(respuesta.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, None


def describir_modelos(modelos: list[dict]) -> tuple[str, bool]:
    """Texto con nombre y version de cada modelo, y si son los modelos de prueba del CI."""
    texto = ", ".join(
        f"{m.get('pregunta', '?').upper()} {m.get('nombre', '?')} v{m.get('version', '?')}"
        for m in modelos
    )
    de_prueba = any(str(m.get("run_id", "")).startswith("run-de-prueba") for m in modelos)
    return texto or "-", de_prueba


def inspeccionar(contenedor: str, ejecutable: str) -> dict:
    """Limite de CPU y memoria y workers de uvicorn del contenedor (vacio si no se puede leer)."""
    formato = (
        "{{.HostConfig.NanoCpus}}|{{.HostConfig.Memory}}|{{range .Config.Env}}{{println .}}{{end}}"
    )
    try:
        salida = subprocess.run(  # nosec B603  # argumentos fijos
            [ejecutable, "inspect", "--format", formato, contenedor],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    nano_cpus, memoria, variables = salida.split("|", 2)
    workers = next(
        (v.split("=", 1)[1] for v in variables.splitlines() if v.startswith("WEB_CONCURRENCY=")),
        "1",
    )
    cpus = int(nano_cpus or 0) / 1e9
    memoria_mib = int(memoria or 0) / 2**20
    return {
        "cpus": cpus or None,  # 0 = sin limite
        "memoria_mib": memoria_mib or None,
        "workers": int(workers or 1),
    }


def texto_limites(limites: dict) -> str:
    if not limites:
        return "no se pudo inspeccionar"
    cpus = f"{limites['cpus']:g} vCPU" if limites.get("cpus") else "CPU sin limite"
    memoria = (
        f"{limites['memoria_mib']:.0f} MiB" if limites.get("memoria_mib") else "memoria sin limite"
    )
    return f"{cpus}, {memoria}, {limites.get('workers', 1)} worker(s) de uvicorn"


def entorno_de(url: str, ejecutable: str) -> dict:
    """Lo que hace falta para interpretar la corrida: modelos, contenedor y maquina."""
    _, modelos = consultar(f"{url}/modelo")
    texto, de_prueba = describir_modelos(modelos if isinstance(modelos, list) else [])
    limites = inspeccionar("mibici-api", ejecutable)
    return {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
        "modelos": modelos if isinstance(modelos, list) else [],
        "modelos_texto": texto,
        "modelos_de_prueba": de_prueba,
        "limites_api": texto_limites(limites),
        "contenedor_api": limites,
        "filas_por_peticion": config.CARGA_FILAS,
        "escalones": config.CARGA_ESCALONES,
        "espera_seg": list(config.CARGA_ESPERA_SEG),
        "pesos": config.CARGA_PESOS,
        "espera_lotes_seg": list(config.CARGA_ESPERA_LOTES_SEG),
        "pesos_lotes": config.CARGA_PESOS_LOTES,
        "maquina": {
            "sistema": platform.platform(),
            "procesador": platform.machine(),
            "cpus": os.cpu_count(),
        },
    }


def correr_locust(url: str, crudos: Path, peticiones_csv: Path) -> int:
    """Corre Locust sin interfaz; regresa su codigo de salida (1 si hubo peticiones fallidas)."""
    comando = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        url,
        "--csv",
        str(crudos / "locust"),
        "--html",
        str(crudos / "locust.html"),
        "--only-summary",
    ]
    variables = {**os.environ, "CARGA_PETICIONES_CSV": str(peticiones_csv)}
    return subprocess.run(comando, env=variables, check=False).returncode  # nosec B603


@app.command()
def main(
    url: str = typer.Option(config.CARGA_URL, help="URL base de la API."),
    motor: str = typer.Option(
        os.getenv("CONTAINER", "docker"), help="docker o podman (para stats e inspect)."
    ),
    reposo_seg: float = typer.Option(5, help="Segundos de muestreo antes y despues de la carga."),
):
    url = url.rstrip("/")
    codigo, _ = consultar(f"{url}/salud")
    if codigo != 200:
        logger.error(
            f"La API en {url} no esta lista (GET /salud -> {codigo or 'sin respuesta'}). "
            "Levantala con `make up` (o `make run-api`) y espera a que este sana."
        )
        raise typer.Exit(1)

    entorno = entorno_de(url, motor)
    logger.info(f"Modelos: {entorno['modelos_texto']}")
    logger.info(f"Contenedor de la API: {entorno['limites_api']}")
    if entorno["modelos_de_prueba"]:
        logger.warning(
            "La API sirve los modelos de prueba: el resultado no sirve para dimensionar"
        )
    duracion = sum(segundos for _, segundos in config.CARGA_ESCALONES)
    logger.info(f"Escalones {config.CARGA_ESCALONES}: ~{duracion / 60:.1f} min")

    crudos = config.CARGA_CRUDOS_DIR
    crudos.mkdir(parents=True, exist_ok=True)
    peticiones_csv = crudos / "peticiones.csv"
    muestreo = recursos.Muestreo(
        config.CARGA_CONTENEDORES, crudos / "docker_stats.csv", ejecutable=motor
    ).iniciar()
    try:
        time.sleep(reposo_seg)  # linea base en reposo
        salida = correr_locust(url, crudos, peticiones_csv)
        time.sleep(reposo_seg)  # recuperacion despues de la carga
    finally:
        muestras = muestreo.detener()
    logger.info(f"Locust termino (codigo {salida}); {muestras} muestras de docker stats")

    resumen = analisis.generar(peticiones_csv, crudos / "docker_stats.csv", entorno)
    capacidad = resumen["capacidad"]
    logger.success(
        f"{resumen['peticiones']:,} peticiones, {100 * resumen['tasa_error']:.2f} % de error, "
        f"p95 en tiempo real {resumen['p95_ms']} ms. Capacidad: "
        + (
            f"{capacidad['predicciones_por_seg']:,.0f} predicciones/s "
            f"({capacidad['rps']:.1f} peticiones/s)"
            if capacidad
            else "ningun escalon cumplio"
        )
    )
    logger.success(f"Resumen en {config.CARGA_DIR / 'resumen_carga.md'}")


if __name__ == "__main__":
    app()
