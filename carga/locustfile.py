"""Escenario de la prueba de carga de la API de MiBici con Locust.

Sigue la estrategia de despliegue del README (P2 en tiempo real; P1 y P3 por lotes):

- ``Despachador``: casi todos los usuarios simulados. Es el tablero del equipo de rebalanceo:
  pide el riesgo de P2 de las estaciones que esta viendo (``config.CARGA_FILAS["p2"]`` por
  peticion) y de vez en cuando ``/salud``, con una pausa de ``config.CARGA_ESPERA_SEG``.
- ``ProcesoPorLotes``: exactamente un usuario, que manda lotes de P1 (1000 filas, el maximo de la
  API) y de P3 (todas las estaciones) al mismo tiempo que el trafico en tiempo real. Sus
  peticiones se llaman ``... [lote]`` para separarlas en el resumen.

Las filas parten de los ejemplos de ``api/ejemplos/`` con estacion, hora o bloque y dia al azar
(``carga.peticiones``). Una respuesta cuenta como error si no es 200 o si no trae una prediccion
por fila.

La carga sube por escalones (``config.CARGA_ESCALONES``), lo que permite ver donde se satura una
instancia de tamano conocido (el contenedor con 1 vCPU y 1 GiB de ``docker-compose.yml``).

Normalmente se corre con ``make carga`` (``python -m carga.ejecutar``), que ademas registra el CPU
y la memoria de los contenedores y genera las graficas. Para usar la interfaz web de Locust::

    locust -f carga/locustfile.py --host http://127.0.0.1:8000

Si la variable ``CARGA_PETICIONES_CSV`` tiene una ruta, cada peticion se escribe ahi (momento,
endpoint, milisegundos, si fallo, usuarios activos y filas): de ese archivo salen los percentiles
exactos por escalon.
"""

import csv
import os
from pathlib import Path
import random
import sys
import time

from locust import FastHttpUser, LoadTestShape, between, events, task

# `locust -f carga/locustfile.py` solo agrega carga/ a la ruta de importacion: se agrega la raiz
RAIZ = str(Path(__file__).resolve().parents[1])
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from carga.peticiones import peticion
from tc5061_proyecto_equipo18 import config

LLEGADAS_POR_SEG = 100


LOTE = " [lote]"
FILAS_POR_ENDPOINT = {
    "/predicciones/p2": config.CARGA_FILAS["p2"],
    f"/predicciones/p1{LOTE}": config.CARGA_FILAS["p1"],
    f"/predicciones/p3{LOTE}": config.CARGA_FILAS["p3"],
    "/salud": 0,
}


class ClienteApi(FastHttpUser):
    """Base: pide predicciones y revisa que llegue una por fila."""

    abstract = True

    def on_start(self):
        self.azar = random.Random()  # nosec B311  # datos de prueba, no criptografia

    def predecir(self, pregunta: str, filas: int, sufijo: str = ""):
        cuerpo = peticion(pregunta, self.azar, filas)
        with self.client.post(
            f"/predicciones/{pregunta}",
            json=cuerpo,
            name=f"/predicciones/{pregunta}{sufijo}",
            catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
                return
            predicciones = (respuesta.json() or {}).get("predicciones", [])
            if len(predicciones) != filas:
                respuesta.failure(f"{len(predicciones)} predicciones para {filas} filas")


class Despachador(ClienteApi):
    """Tablero de rebalanceo: riesgo de P2 en tiempo real."""

    wait_time = between(*config.CARGA_ESPERA_SEG)

    @task(config.CARGA_PESOS["p2"])
    def p2(self):
        self.predecir("p2", config.CARGA_FILAS["p2"])

    @task(config.CARGA_PESOS["salud"])
    def salud(self):
        self.client.get("/salud", name="/salud")


class ProcesoPorLotes(ClienteApi):
    """Proceso programado: lotes de P1 (dia siguiente) y P3 (grupos). Siempre es uno solo."""

    fixed_count = 1
    wait_time = between(*config.CARGA_ESPERA_LOTES_SEG)

    @task(config.CARGA_PESOS_LOTES["p1"])
    def p1(self):
        self.predecir("p1", config.CARGA_FILAS["p1"], LOTE)

    @task(config.CARGA_PESOS_LOTES["p3"])
    def p3(self):
        self.predecir("p3", config.CARGA_FILAS["p3"], LOTE)


class Escalones(LoadTestShape):
    """Sube los usuarios por escalones (``config.CARGA_ESCALONES``) y termina al acabar."""

    escalones = config.CARGA_ESCALONES

    def tick(self):
        transcurrido = self.get_run_time()
        fin = 0
        for usuarios, segundos in self.escalones:
            fin += segundos
            if transcurrido < fin:
                # Llegan hasta 100 usuarios por segundo (mas rapido Locust lo desaconseja): el
                # escalon se completa en ~2 s, dentro del calentamiento que no se resume
                return usuarios, min(usuarios, LLEGADAS_POR_SEG)
        return None


# =============================================================================
# Registro de cada peticion (para percentiles exactos por escalon)
# =============================================================================
COLUMNAS_PETICIONES = ["tiempo", "endpoint", "ms", "fallo", "usuarios", "filas"]
_registro: dict = {}


@events.test_start.add_listener
def abrir_registro(environment, **_):
    ruta = os.getenv("CARGA_PETICIONES_CSV")
    if not ruta:
        return
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    archivo = open(ruta, "w", newline="", encoding="utf-8")  # noqa: SIM115 - se cierra al final
    escritor = csv.writer(archivo)
    escritor.writerow(COLUMNAS_PETICIONES)
    _registro.update(archivo=archivo, escritor=escritor, entorno=environment)


@events.request.add_listener
def registrar_peticion(name, response_time, exception, **_):
    escritor = _registro.get("escritor")
    if escritor is None:
        return
    runner = _registro["entorno"].runner
    usuarios = runner.user_count if runner is not None else 0
    fallo = exception is not None
    escritor.writerow(
        [
            f"{time.time():.3f}",
            name,
            f"{response_time:.2f}",
            int(fallo),
            usuarios,
            0 if fallo else FILAS_POR_ENDPOINT.get(name, 0),  # predicciones entregadas
        ]
    )


@events.test_stop.add_listener
def cerrar_registro(**_):
    archivo = _registro.pop("archivo", None)
    _registro.clear()
    if archivo is not None:
        archivo.close()
