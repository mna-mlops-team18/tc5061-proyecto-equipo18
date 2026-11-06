"""Prueba de humo del Model Registry: carga, firma y prediccion de cada modelo registrado.

Para cada pregunta (P1, P2, P3) y cada alias pedido (por defecto champion y challenger):

1. carga ``models:/<modelo registrado>@<alias>``;
2. imprime la firma de entrada (``model.metadata.get_input_schema()``), que copian los
   esquemas Pydantic de la API;
3. predice unas filas de la particion de prueba;
4. con ``--requirements``, exporta las dependencias con
   ``mlflow.pyfunc.get_model_dependencies`` para fijarlas en ``requirements-api.txt``.

Uso, con el servidor de MLflow encendido (README > Servidor de tracking)::

    python scripts/check_registry.py
    python scripts/check_registry.py --aliases champion --filas 2
    python scripts/check_registry.py --requirements requirements-api.txt

Termina con codigo 1 si algun modelo no carga o no predice.
"""

from pathlib import Path
import sys

import mlflow
import pandas as pd
import typer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from tc5061_proyecto_equipo18 import config  # noqa: E402
from tc5061_proyecto_equipo18.modeling.comun import cargar_dataset  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# Parquet de cada pregunta y columna con la que se separa por tiempo (None: columna "periodo")
DATASETS = {
    "P1": (config.DATASET_P1, "hora"),
    "P2": (config.DATASET_P2, "bloque_inicio"),
    "P3": (config.DATASET_P3, None),
}


def filas_de_prueba(pregunta: str, columnas: list[str], n: int) -> pd.DataFrame:
    """Primeras ``n`` filas de la particion de prueba con las columnas de la firma, en su orden."""
    ruta, columna_tiempo = DATASETS[pregunta]
    prueba = cargar_dataset(ruta, columna_tiempo)["prueba"]
    faltan = [c for c in columnas if c not in prueba.columns]
    if faltan:
        raise RuntimeError(f"El dataset de {pregunta} no tiene las columnas de la firma: {faltan}")
    return prueba[columnas].head(n)


def revisar(pregunta: str, alias: str, filas: int) -> str:
    """Carga, muestra la firma y predice; devuelve el URI para exportar sus dependencias."""
    nombre = config.MODELOS_REGISTRADOS[pregunta]
    uri = f"models:/{nombre}@{alias}"
    typer.echo(f"\n=== {pregunta} {uri}")

    modelo = mlflow.pyfunc.load_model(uri)
    version = mlflow.MlflowClient().get_model_version_by_alias(nombre, alias).version
    typer.echo(f"alias {alias} -> version {version}")

    esquema = modelo.metadata.get_input_schema()
    typer.echo("Firma de entrada:")
    for columna in esquema.inputs:
        typer.echo(f"  {columna.name}: {columna.type}")
    typer.echo(f"Firma de salida: {modelo.metadata.get_output_schema()}")

    X = filas_de_prueba(pregunta, [c.name for c in esquema.inputs], filas)
    typer.echo(f"Entrada ({len(X)} filas de prueba):\n{X.to_string()}")
    predicho = modelo.predict(X)
    typer.echo(f"Prediccion: {list(predicho)}")
    if len(predicho) != len(X):
        raise RuntimeError(f"Se esperaban {len(X)} predicciones y llegaron {len(predicho)}")
    return uri


def exportar_dependencias(uris: list[str], destino: Path | None) -> None:
    """Imprime las dependencias de cada modelo y, si se pide, las junta en un solo archivo."""
    fijadas: dict[str, str] = {}
    conflictos: list[str] = []
    for uri in uris:
        ruta = Path(mlflow.pyfunc.get_model_dependencies(uri))
        lineas = [
            linea.strip()
            for linea in ruta.read_text().splitlines()
            if linea.strip() and not linea.startswith("#")
        ]
        typer.echo(f"\n=== Dependencias de {uri} ({ruta})")
        typer.echo("\n".join(f"  {linea}" for linea in lineas))
        for linea in lineas:
            paquete = linea.split("==")[0].lower()
            if fijadas.get(paquete, linea) != linea:
                conflictos.append(f"{fijadas[paquete]} contra {linea}")
            fijadas[paquete] = linea
    if conflictos:
        typer.secho(f"\nOJO: versiones distintas entre modelos: {conflictos}", fg="yellow")
    if destino is not None:
        destino.write_text("\n".join(sorted(fijadas.values(), key=str.lower)) + "\n")
        typer.echo(f"\nDependencias fijadas escritas en {destino}")


@app.command()
def main(
    aliases: list[str] = typer.Option(
        [config.ALIAS_PRODUCCION, config.ALIAS_RETADOR], "--aliases", "-a", help="Aliases a probar."
    ),
    filas: int = typer.Option(2, "--filas", "-n", help="Filas de prueba por modelo."),
    requirements: Path | None = typer.Option(
        None, "--requirements", help="Escribe aqui las dependencias fijadas de los champion."
    ),
):
    """Carga y predice cada modelo registrado, con cada alias, y exporta sus dependencias."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    typer.echo(f"Tracking URI: {config.MLFLOW_TRACKING_URI}")

    resultados: dict[str, str] = {}
    champions: list[str] = []
    for pregunta in config.MODELOS_REGISTRADOS:
        for alias in aliases:
            clave = f"{pregunta}@{alias}"
            try:
                uri = revisar(pregunta, alias, filas)
                resultados[clave] = "OK"
                if alias == config.ALIAS_PRODUCCION:
                    champions.append(uri)
            except Exception as error:  # noqa: BLE001 - se reporta y se sigue con los demas
                resultados[clave] = f"FALLO: {type(error).__name__}: {error}"

    if champions:
        exportar_dependencias(champions, requirements)

    typer.echo("\n=== Resumen")
    for clave, estado in resultados.items():
        typer.secho(f"  {clave}: {estado}", fg="green" if estado == "OK" else "red")
    if any(estado != "OK" for estado in resultados.values()):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
