"""Entrena UN modelo de una pregunta y lo registra en MLflow (punto de entrada ``train``).

Se lanza desde el MLflow Project, desde la raiz del repositorio::

    mlflow run . -e train -P pregunta=p1 -P modelo=rezago_semanal --env-manager local
    mlflow run . -e train -P pregunta=p1 -P modelo=hgb_poisson -P learning_rate=0.05 --env-manager local

``pregunta`` es p1, p2 o p3 y ``modelo`` es un nombre del catalogo de config.py
(``P1_MODELOS``, ``P2_MODELOS`` y ``P3_MODELOS``). Cualquier hiperparametro del modelo se
cambia con ``-P nombre=valor``: MLflow lo pasa a este script como ``--nombre valor`` y los que
no se pasan toman su valor de config.py.

Este archivo solo orquesta: cada pregunta vive en su propio modulo y cumple el contrato
descrito en ``comun.py`` (``cargar_datos()`` y ``correr(modelo, hiperparametros, ...)``).
"""

from loguru import logger
import pandas as pd
import typer

from tc5061_proyecto_equipo18.modeling.comun import (
    configurar_mlflow,
    leer_opciones_extra,
    modulo_de,
    normalizar_pregunta,
    resolver_hiperparametros,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def entrenar(
    pregunta: str,
    modelo: str,
    cambios: dict | None = None,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
) -> str:
    """Resuelve los hiperparametros (config.py + ``cambios``) y corre el modelo; da el run_id."""
    pregunta = normalizar_pregunta(pregunta)
    hiperparametros = resolver_hiperparametros(pregunta, modelo, cambios)
    return modulo_de(pregunta).correr(
        modelo, hiperparametros, datos=datos, meta_features=meta_features
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    pregunta: str = typer.Option(..., "--pregunta", "-p", help="p1, p2 o p3."),
    modelo: str = typer.Option(..., "--modelo", "-m", help="Nombre del catalogo de config.py."),
):
    """Entrena un modelo; los hiperparametros extra van como --nombre valor."""
    cambios = leer_opciones_extra(ctx.args)
    # Se valida antes de tocar MLflow: un nombre mal escrito falla de inmediato
    resolver_hiperparametros(normalizar_pregunta(pregunta), modelo, cambios)
    configurar_mlflow()
    run_id = entrenar(pregunta, modelo, cambios)
    logger.success(f"Run {run_id} registrado ({pregunta.lower()}, {modelo})")


if __name__ == "__main__":
    app()
