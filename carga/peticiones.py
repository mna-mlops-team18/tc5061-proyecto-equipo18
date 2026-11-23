"""Cuerpos de peticion para la prueba de carga, a partir de los ejemplos de ``api/ejemplos/``.

Cada fila es un ejemplo con la estacion, la hora (P1) o el bloque (P2) y el dia cambiados al
azar, siempre dentro de lo que acepta la API (``api/esquemas.py``), para que la carga no sea la
misma fila repetida. Esta aparte del locustfile para poder probarlo sin importar Locust.
"""

import copy
import json
import random

from tc5061_proyecto_equipo18 import config

EJEMPLOS_DIR = config.PROJ_ROOT / "api" / "ejemplos"
PREGUNTAS = ("p1", "p2", "p3")
ESTACION_MAX = 500  # MiBici tiene ~470 estaciones activas; el id no entra al modelo


def cargar_ejemplos() -> dict[str, list[dict]]:
    """Filas de ejemplo de cada pregunta (las mismas de la documentacion de la API)."""
    return {
        pregunta: json.loads(
            (EJEMPLOS_DIR / f"peticion_{pregunta}.json").read_text(encoding="utf-8")
        )["instancias"]
        for pregunta in PREGUNTAS
    }


EJEMPLOS = cargar_ejemplos()


def fila(pregunta: str, azar: random.Random) -> dict:
    """Una fila valida de ``pregunta``: un ejemplo con estacion, hora o bloque y dia al azar."""
    nueva = copy.deepcopy(azar.choice(EJEMPLOS[pregunta]))
    nueva["estacion_id"] = azar.randint(1, ESTACION_MAX)
    if pregunta in ("p1", "p2"):
        dia = azar.randint(0, 6)
        nueva["dia_semana"] = dia
        nueva["es_fin_semana"] = int(dia >= 5)  # la API exige que coincidan
    if pregunta == "p1":
        nueva["hora_dia"] = azar.randint(0, 23)
    elif pregunta == "p2":
        nueva["bloque_dia"] = 3 * azar.randint(0, 7)
    return nueva


def peticion(pregunta: str, azar: random.Random, instancias: int = 1) -> dict:
    """Cuerpo de ``POST /predicciones/<pregunta>`` con ``instancias`` filas."""
    return {"instancias": [fila(pregunta, azar) for _ in range(instancias)]}
