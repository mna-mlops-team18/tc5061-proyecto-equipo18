"""Modelos de prueba en ``models/servicio/`` para construir y probar la imagen de la API sin datos.

El Containerfile copia ``models/servicio/p1``, ``p2`` y ``p3``, que salen de
``make export-model`` (Model Registry con los champions reales) y no estan en Git. En el CI no
hay datos de MiBici ni registry, asi que este script guarda en su lugar modelos pequenos
entrenados con datos sinteticos, con las mismas columnas, tipos, firma, ejemplo de entrada y
``registro.json`` que los exportados. Son los mismos que usa ``tests/test_api.py``.

Sirven para comprobar que la imagen se construye, arranca, carga los tres modelos y responde;
NO para medir desempeno del modelo ni para la prueba de carga oficial (esa va con los modelos
reales).

Uso, desde la raiz del repo y con el ambiente del proyecto::

    python scripts/modelos_de_prueba.py            # escribe models/servicio/p1, p2 y p3
    python scripts/modelos_de_prueba.py --destino /tmp/servicio

No sobrescribe una carpeta que ya tenga un modelo (podria ser un champion exportado), salvo con
``--reemplazar``.
"""

import argparse
from pathlib import Path
import shutil
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))  # importar tests/ y el paquete al correr como script

from tests.test_api import PREGUNTAS, guardar_modelo, tabla

from tc5061_proyecto_equipo18 import config


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", type=Path, default=config.MODELO_SERVICIO_DIR)
    parser.add_argument("--reemplazar", action="store_true", help="Borra modelos existentes.")
    opciones = parser.parse_args(argumentos)

    for pregunta in PREGUNTAS:
        carpeta = opciones.destino / pregunta
        if (carpeta / "MLmodel").exists():
            if not opciones.reemplazar:
                print(f"{carpeta} ya tiene un modelo; usa --reemplazar para sobrescribirlo")
                return 1
            shutil.rmtree(carpeta)
        X, y = tabla(pregunta)
        guardar_modelo(carpeta, pregunta, X, y)
        print(f"{pregunta}: modelo de prueba en {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
