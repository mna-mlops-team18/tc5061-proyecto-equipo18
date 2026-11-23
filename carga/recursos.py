"""CPU y memoria de los contenedores durante la prueba de carga, con ``docker stats``.

Toma una muestra de ``docker stats --no-stream`` tras otra (cada una tarda ~1-2 s, lo que Docker
necesita para medir el CPU) y la agrega a un CSV con una fila por contenedor y muestra:

``tiempo, contenedor, cpu_pct, memoria_mib, limite_mib, memoria_pct, red_entrada_mb,
red_salida_mb``

``cpu_pct`` es relativo a un nucleo: 100 % = 1 vCPU ocupada, el limite de la API en
``docker-compose.yml``. Con Podman: ``CONTAINER=podman`` (``podman stats`` acepta el mismo
formato).

Solo se miden los contenedores que estan corriendo; si ninguno corre, no hay muestras y el CSV
queda solo con el encabezado.
"""

import csv
from pathlib import Path
import re
import shutil
import subprocess  # nosec B404  # corre docker/podman con argumentos fijos, sin shell
import threading
import time

from loguru import logger

COLUMNAS = [
    "tiempo",
    "contenedor",
    "cpu_pct",
    "memoria_mib",
    "limite_mib",
    "memoria_pct",
    "red_entrada_mb",
    "red_salida_mb",
]
FORMATO = "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}"

# Unidades de docker stats: binarias (KiB, MiB, GiB) en memoria, decimales (kB, MB, GB) en red
_UNIDADES = {
    "b": 1,
    "kb": 1e3,
    "mb": 1e6,
    "gb": 1e9,
    "tb": 1e12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}
_TAMANO = re.compile(r"^\s*([0-9.]+)\s*([a-zA-Z]+)\s*$")


def a_bytes(texto: str) -> float:
    """``'123.4MiB'`` -> bytes. Acepta las unidades de ``docker stats`` y ``podman stats``."""
    coincide = _TAMANO.match(texto)
    if not coincide:
        raise ValueError(f"Tamano no reconocido: {texto!r}")
    numero, unidad = coincide.groups()
    try:
        return float(numero) * _UNIDADES[unidad.lower()]
    except KeyError:
        raise ValueError(f"Unidad no reconocida en {texto!r}") from None


def porcentaje(texto: str) -> float:
    """``'12.34%'`` -> 12.34 (``'--'``, que da Podman sin dato, -> NaN)."""
    texto = texto.strip().rstrip("%")
    return float("nan") if texto in ("", "--") else float(texto)


def par(texto: str) -> tuple[float, float]:
    """``'1.2kB / 3.4MB'`` -> (1200.0, 3400000.0) en bytes."""
    izquierda, derecha = texto.split("/")
    return a_bytes(izquierda), a_bytes(derecha)


def leer_linea(linea: str, tiempo: float) -> dict:
    """Una linea de ``docker stats`` con ``FORMATO`` -> fila del CSV."""
    nombre, cpu, memoria, memoria_pct, red = linea.split("\t")
    usada, limite = par(memoria)
    entrada, salida = par(red)
    return {
        "tiempo": round(tiempo, 3),
        "contenedor": nombre.strip(),
        "cpu_pct": porcentaje(cpu),
        "memoria_mib": round(usada / 2**20, 2),
        "limite_mib": round(limite / 2**20, 2),
        "memoria_pct": porcentaje(memoria_pct),
        "red_entrada_mb": round(entrada / 1e6, 3),
        "red_salida_mb": round(salida / 1e6, 3),
    }


def motor(nombre: str = "docker") -> str:
    """Ruta del ejecutable de contenedores, o error claro si no esta instalado."""
    ruta = shutil.which(nombre)
    if ruta is None:
        raise RuntimeError(f"No se encontro `{nombre}`: instala Docker o usa CONTAINER=podman")
    return ruta


def corriendo(contenedores: list[str], ejecutable: str) -> list[str]:
    """Los ``contenedores`` que estan corriendo ahora."""
    salida = subprocess.run(  # nosec B603  # argumentos fijos
        [ejecutable, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [nombre for nombre in contenedores if nombre in salida]


def muestra(contenedores: list[str], ejecutable: str) -> list[dict]:
    """Una muestra de ``docker stats`` de los contenedores indicados."""
    if not contenedores:
        return []
    salida = subprocess.run(  # nosec B603  # argumentos fijos
        [ejecutable, "stats", "--no-stream", "--format", FORMATO, *contenedores],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tiempo = time.time()
    return [leer_linea(linea, tiempo) for linea in salida.splitlines() if linea.strip()]


class Muestreo:
    """Muestrea en un hilo aparte hasta ``detener()``; escribe cada muestra en ``ruta``."""

    def __init__(self, contenedores, ruta: Path, ejecutable: str = "docker", pausa: float = 0.0):
        self.ejecutable = motor(ejecutable)
        self.contenedores = corriendo(list(contenedores), self.ejecutable)
        self.ruta = Path(ruta)
        self.pausa = pausa
        self.muestras = 0
        self._alto = threading.Event()
        self._hilo = threading.Thread(target=self._ciclo, name="docker-stats", daemon=True)
        faltan = sorted(set(contenedores) - set(self.contenedores))
        if faltan:
            logger.warning(f"No estan corriendo y no se mediran: {', '.join(faltan)}")

    def _ciclo(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
            escritor.writeheader()
            while not self._alto.is_set() and self.contenedores:
                try:
                    filas = muestra(self.contenedores, self.ejecutable)
                except subprocess.CalledProcessError as error:
                    logger.warning(f"docker stats fallo: {error.stderr.strip()}")
                    filas = []
                escritor.writerows(filas)
                archivo.flush()
                self.muestras += len(filas)
                self._alto.wait(self.pausa)

    def iniciar(self) -> "Muestreo":
        self._hilo.start()
        return self

    def detener(self) -> int:
        """Termina el muestreo (espera la muestra en curso) y regresa cuantas filas escribio."""
        self._alto.set()
        self._hilo.join(timeout=30)
        return self.muestras
