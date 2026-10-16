"""Experimento base: entrena la linea base y el candidato de P1, P2 y P3 y los registra en MLflow.

Despues de la ingesta y de features:

    python -m tc5061_proyecto_equipo18.modeling.train --todos
    python -m tc5061_proyecto_equipo18.modeling.train --pregunta P1 --modelo baseline

Este archivo solo orquesta. Cada pregunta vive en su propio modulo (``p1_regresion.py``,
``p2_clasificacion.py`` y ``p3_clustering.py``) y cumple el contrato descrito en ``comun.py``:
expone ``entrenar(roles) -> list[str]``. Asi cada integrante trabaja su pregunta en un PR
aparte, y una pregunta cuyo modulo aun no existe se omite con un aviso en lugar de fallar.
"""

import importlib
from types import ModuleType
from typing import Annotated

from loguru import logger
import typer

from tc5061_proyecto_equipo18.modeling.comun import PREGUNTAS, ROLES, configurar_mlflow

app = typer.Typer(add_completion=False)

MODULOS = {
    "P1": "tc5061_proyecto_equipo18.modeling.p1_regresion",
    "P2": "tc5061_proyecto_equipo18.modeling.p2_clasificacion",
    "P3": "tc5061_proyecto_equipo18.modeling.p3_clustering",
}


def cargar_modulo(pregunta: str) -> ModuleType | None:
    """Importa el modulo de una pregunta; devuelve None si todavia no existe en el repo."""
    nombre = MODULOS[pregunta]
    try:
        return importlib.import_module(nombre)
    except ModuleNotFoundError as error:
        if error.name != nombre:  # falta otra dependencia: es un error real, no un pendiente
            raise
        return None


def ejecutar(preguntas: list[str], roles: tuple[str, ...]) -> dict[str, list[str]]:
    """Corre ``entrenar(roles)`` de cada pregunta y devuelve los run_id por pregunta."""
    experimento_id = configurar_mlflow()
    logger.info(f"Experimento MLflow {experimento_id}; preguntas {preguntas}, modelos {roles}")
    runs: dict[str, list[str]] = {}
    for pregunta in preguntas:
        modulo = cargar_modulo(pregunta)
        if modulo is None:
            logger.warning(f"{pregunta}: {MODULOS[pregunta]} aun no existe; se omite")
            continue
        logger.info(f"{pregunta}: entrenando {', '.join(roles)}")
        runs[pregunta] = modulo.entrenar(roles)
        logger.success(f"{pregunta}: {len(runs[pregunta])} run(s) registrados")
    return runs


@app.command()
def main(
    pregunta: Annotated[
        list[str] | None,
        typer.Option("--pregunta", "-p", help="P1, P2 o P3; se puede repetir."),
    ] = None,
    modelo: Annotated[
        list[str] | None,
        typer.Option("--modelo", "-m", help="baseline o candidato; por defecto ambos."),
    ] = None,
    todos: bool = typer.Option(False, "--todos", help="Las tres preguntas con ambos modelos."),
):
    """Entrena y registra en MLflow los modelos elegidos del experimento base."""
    pregunta, modelo = pregunta or [], modelo or []
    preguntas = list(PREGUNTAS) if todos else sorted({p.strip().upper() for p in pregunta})
    if not preguntas:
        raise typer.BadParameter("Indica --pregunta P1/P2/P3 o --todos")
    desconocidas = set(preguntas) - set(PREGUNTAS)
    if desconocidas:
        raise typer.BadParameter(f"Preguntas desconocidas {sorted(desconocidas)}; usa {PREGUNTAS}")
    if set(modelo) - set(ROLES):
        raise typer.BadParameter(
            f"Modelos desconocidos {sorted(set(modelo) - set(ROLES))}; usa {ROLES}"
        )
    roles = ROLES if todos or not modelo else tuple(r for r in ROLES if r in modelo)

    runs = ejecutar(preguntas, roles)
    if not runs:
        logger.error("No se entreno ningun modelo: faltan los modulos de las preguntas elegidas")
        raise typer.Exit(code=1)
    total = sum(len(ids) for ids in runs.values())
    logger.success(
        f"Experimento listo: {total} run(s). Revisa los resultados con `make mlflow-ui`"
    )


if __name__ == "__main__":
    app()
