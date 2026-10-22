"""Busqueda de hiperparametros organizada como nested runs (punto de entrada ``search``).

Una corrida padre por busqueda y una corrida hija por combinacion de la rejilla::

    mlflow run . -e search -P pregunta=p1 --env-manager local

El modelo y la rejilla de cada pregunta viven en config.py (``P1_BUSQUEDA``, ``P2_BUSQUEDA`` y
``P3_BUSQUEDA``). Cada hija es un run completo (metricas, figuras, modelo con firma y
conjuntos de datos) con ``etapa = busqueda``; la padre resume la busqueda:

- parametros ``rejilla``, ``n_combinaciones`` y ``metrica_seleccion``;
- la mejor combinacion como parametros ``mejor_<hiperparametro>``, la metrica
  ``mejor_<metrica>`` y las etiquetas ``mejor_run_id`` y ``mejor_run_name``;
- ``tablas/resultados_busqueda.csv`` (una fila por hija) y ``figuras/busqueda.png``.

La mejor combinacion se elige con la metrica de ``config.METRICAS_SELECCION``, igual que al
comparar contra el baseline.
"""

from itertools import product

from loguru import logger
import mlflow
import pandas as pd
import typer

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import cargar_metadatos_features
from tc5061_proyecto_equipo18.modeling.comun import (
    configurar_mlflow,
    especificacion,
    iniciar_run,
    modulo_de,
    normalizar_pregunta,
    registrar_figura,
    registrar_tabla,
    resolver_hiperparametros,
    resumen_json,
)
from tc5061_proyecto_equipo18.plots import grafica_busqueda

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def combinaciones(rejilla: dict[str, list]) -> list[dict]:
    """Todas las combinaciones de la rejilla, en orden: ``{"a": [1, 2]}`` -> ``[{"a": 1}, ...]``."""
    claves = list(rejilla)
    return [dict(zip(claves, valores)) for valores in product(*(rejilla[c] for c in claves))]


def nombre_hijo(pregunta: str, modelo: str, combinacion: dict) -> str:
    """``p1-busqueda-hgb_poisson-learning_rate0.05-max_leaf_nodes31``."""
    partes = [
        f"{clave}{valor:g}" if isinstance(valor, float) else f"{clave}{valor}"
        for clave, valor in combinacion.items()
    ]
    return "-".join([f"{pregunta.lower()}-busqueda-{modelo}", *partes])


def validar_busqueda(pregunta: str) -> tuple[str, dict]:
    """Modelo y rejilla de una pregunta; falla si la rejilla usa hiperparametros que no existen."""
    busqueda = config.BUSQUEDAS[normalizar_pregunta(pregunta)]
    modelo, rejilla = busqueda["modelo"], busqueda["rejilla"]
    por_defecto = especificacion(pregunta, modelo)["hiperparametros"]
    desconocidos = sorted(set(rejilla) - set(por_defecto))
    if desconocidos:
        raise ValueError(
            f"La rejilla de {pregunta} usa hiperparametros desconocidos: {desconocidos}"
        )
    return modelo, rejilla


def ejecutar_busqueda(
    pregunta: str,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
    rejilla: dict[str, list] | None = None,
) -> str:
    """Corre la busqueda de una pregunta y devuelve el run_id de la corrida padre."""
    pregunta = normalizar_pregunta(pregunta)
    modelo, rejilla_config = validar_busqueda(pregunta)
    rejilla = rejilla or rejilla_config
    modulo = modulo_de(pregunta)
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    datos = datos if datos is not None else modulo.cargar_datos()  # se lee una sola vez
    metrica, sentido = config.METRICAS_SELECCION[pregunta]
    combos = combinaciones(rejilla)
    descripcion = (
        f"Busqueda de {modelo} ({len(combos)} combinaciones, una corrida hija por cada una). "
        f"Reproducir: mlflow run . -e search -P pregunta={pregunta.lower()} --env-manager local"
    )

    with iniciar_run(
        pregunta,
        modelo,
        etapa="busqueda",
        meta_features=meta,
        nombre=f"{pregunta.lower()}-busqueda-{modelo}",
        parametros={
            "rejilla": resumen_json(rejilla),
            "n_combinaciones": len(combos),
            "metrica_seleccion": metrica,
        },
        descripcion=descripcion,
    ) as padre:
        filas = []
        for numero, combinacion in enumerate(combos, start=1):
            logger.info(f"{pregunta} busqueda {numero}/{len(combos)}: {combinacion}")
            hiperparametros = resolver_hiperparametros(pregunta, modelo, combinacion)
            run_id = modulo.correr(
                modelo,
                hiperparametros,
                datos=datos,
                meta_features=meta,
                etapa="busqueda",
                anidado=True,
                nombre=nombre_hijo(pregunta, modelo, combinacion),
            )
            hijo = mlflow.get_run(run_id)
            filas.append(
                {
                    "run_id": run_id,
                    "run_name": hijo.info.run_name,
                    **combinacion,
                    **dict(sorted(hijo.data.metrics.items())),
                }
            )

        resultados = pd.DataFrame(filas)
        orden = resultados.sort_values(metrica, ascending=sentido == "min")
        mejor = orden.iloc[0]
        mlflow.log_params({f"mejor_{clave}": mejor[clave] for clave in rejilla})
        mlflow.log_metric(f"mejor_{metrica}", float(mejor[metrica]))
        mlflow.set_tags({"mejor_run_id": mejor["run_id"], "mejor_run_name": mejor["run_name"]})
        registrar_tabla(orden, "resultados_busqueda")
        registrar_figura(
            grafica_busqueda(
                resultados, list(rejilla), metrica, sentido, f"Busqueda de {modelo} ({pregunta})"
            ),
            "busqueda",
        )
        logger.success(f"{pregunta}: mejor combinacion {mejor['run_name']} ({metrica})")
    return padre.info.run_id


@app.command()
def main(pregunta: str = typer.Option(..., "--pregunta", "-p", help="p1, p2 o p3.")):
    """Corre la busqueda de hiperparametros de una pregunta con nested runs."""
    validar_busqueda(normalizar_pregunta(pregunta))
    configurar_mlflow()
    run_id = ejecutar_busqueda(pregunta)
    logger.success(f"Busqueda registrada: corrida padre {run_id}")


if __name__ == "__main__":
    app()
