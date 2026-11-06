"""Exporta un modelo del Model Registry a la carpeta que la API empaqueta en su imagen.

Es el puente entre el Model Registry (entrega 2) y el servicio de prediccion (entrega 4)::

    servidor de MLflow                    carpeta local (Git la ignora)      imagen de la API
    models:/<nombre>@champion  --------> models/servicio/p1/  ---COPY---> /app/models/servicio/p1

Uso, desde la raiz del repositorio, con el ambiente activado y el servidor de MLflow levantado
(README > Experimentos en MLflow)::

    python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1
    python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1 --version 3

Que hace:

1. Busca el modelo registrado de la pregunta (``config.MODELOS_REGISTRADOS``) y convierte el
   alias (``champion`` por defecto) en un numero de version. Descarga esa version exacta: si
   alguien mueve el alias mientras tanto, no se mezclan dos versiones.
2. Copia sus archivos (``MLmodel``, el modelo en formato skops, la firma y el ejemplo de
   entrada) a ``models/servicio/<pregunta>/``, reemplazando una exportacion anterior.
3. Comprueba que la copia se carga con ``mlflow.pyfunc`` y predice su ejemplo de entrada.
4. Escribe ``registro.json`` (nombre, version, alias, run de origen, linaje y huella SHA-256 de
   cada archivo), que la API lee al arrancar y reporta en cada respuesta, y
   ``peticion_ejemplo.json``, una peticion para la API con las filas del ejemplo de entrada.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from loguru import logger
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import mlflow.pyfunc
import typer

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling.comun import (
    es_servidor,
    normalizar_pregunta,
    verificar_servidor,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

ARCHIVO_REGISTRO = "registro.json"
ARCHIVO_PETICION = "peticion_ejemplo.json"


def destino_de(pregunta: str) -> Path:
    """Carpeta de exportacion de una pregunta: ``models/servicio/p1`` para P1."""
    return config.MODELO_SERVICIO_DIR / normalizar_pregunta(pregunta).lower()


def resolver_version(
    nombre: str, alias: str = config.ALIAS_PRODUCCION, version: str | None = None
) -> ModelVersion:
    """Version del registry que se exporta: la indicada o, si no, la que apunta el alias."""
    cliente = mlflow.MlflowClient()
    try:
        if version is not None:
            return cliente.get_model_version(nombre, str(version))
        return cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException as error:
        buscado = f"models:/{nombre}/{version}" if version else f"models:/{nombre}@{alias}"
        raise RuntimeError(
            f"No existe {buscado} en el Model Registry. Revisa en la interfaz de MLflow "
            "(seccion Models) el nombre y la version, o registra el modelo con: "
            "mlflow run . -e register -P pregunta=p1 --env-manager local"
        ) from error


def _carpeta_del_modelo(ruta: Path) -> Path:
    """Carpeta que contiene ``MLmodel`` (segun la version de MLflow, puede ser una subcarpeta)."""
    if (ruta / "MLmodel").is_file():
        return ruta
    encontradas = sorted(archivo.parent for archivo in ruta.rglob("MLmodel"))
    if len(encontradas) != 1:
        raise RuntimeError(f"La descarga no trae un unico modelo de MLflow: {encontradas}")
    return encontradas[0]


def _validar_destino(destino: Path) -> None:
    """Solo se reemplaza una carpeta vacia o una exportacion anterior (protege otras carpetas)."""
    if destino.is_file():
        raise ValueError(f"{destino} es un archivo, no una carpeta")
    if destino.exists() and any(destino.iterdir()) and not (destino / "MLmodel").is_file():
        raise ValueError(
            f"{destino} tiene archivos que no son una exportacion de un modelo; elige otra carpeta"
        )


def descargar(nombre: str, version: str, destino: Path) -> None:
    """Descarga ``models:/<nombre>/<version>`` y lo deja en ``destino`` (la reemplaza completa)."""
    _validar_destino(destino)
    with tempfile.TemporaryDirectory() as temporal:
        ruta = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{nombre}/{version}", dst_path=temporal
        )
        carpeta = _carpeta_del_modelo(Path(ruta))
        if destino.exists():
            shutil.rmtree(destino)  # sin restos de una version anterior
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(carpeta, destino)


def verificar_copia(destino: Path) -> mlflow.pyfunc.PyFuncModel:
    """Carga la copia como la cargara la API y predice su ejemplo de entrada."""
    modelo = mlflow.pyfunc.load_model(str(destino))
    if modelo.metadata.get_input_schema() is None:
        raise RuntimeError("El modelo no tiene firma de entrada; la API la necesita")
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise RuntimeError("El modelo no tiene ejemplo de entrada para verificar la copia")
    predicho = modelo.predict(ejemplo)
    logger.success(f"La copia en {destino} se carga y predice {len(predicho)} filas")
    return modelo


def huella_sha256(ruta: Path) -> str:
    """Huella SHA-256 de un archivo: permite comprobar que la imagen lleva la copia exacta."""
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


def huellas(destino: Path) -> dict[str, str]:
    """Huella de cada archivo del modelo exportado (sin los dos JSON que escribe este modulo)."""
    propios = {ARCHIVO_REGISTRO, ARCHIVO_PETICION}
    return {
        ruta.relative_to(destino).as_posix(): huella_sha256(ruta)
        for ruta in sorted(destino.rglob("*"))
        if ruta.is_file() and ruta.name not in propios
    }


def peticion_ejemplo(modelo: mlflow.pyfunc.PyFuncModel) -> dict:
    """Cuerpo de una peticion a la API con las filas reales del ejemplo de entrada del modelo."""
    filas = modelo.input_example.to_json(orient="records", double_precision=6)
    return {"instancias": json.loads(filas)}


def exportar(
    pregunta: str,
    alias: str = config.ALIAS_PRODUCCION,
    version: str | None = None,
    nombre: str | None = None,
    destino: Path | None = None,
) -> dict:
    """Exporta el modelo de una pregunta y devuelve lo escrito en ``registro.json``.

    Usa la URI de MLflow ya configurada (la linea de comandos la toma de ``config``).
    """
    pregunta = normalizar_pregunta(pregunta)
    nombre = nombre or config.MODELOS_REGISTRADOS[pregunta]
    destino = Path(destino) if destino is not None else destino_de(pregunta)

    elegida = resolver_version(nombre, alias, version)
    numero = str(elegida.version)
    logger.info(f"Exportando models:/{nombre}/{numero} (alias: {elegida.aliases or 'ninguno'})")
    descargar(nombre, numero, destino)
    modelo = verificar_copia(destino)

    registro = {
        "pregunta": pregunta,
        "nombre": nombre,
        "version": numero,
        "alias": alias if version is None else None,
        "aliases_al_exportar": sorted(elegida.aliases or []),
        "uri_registro": f"models:/{nombre}/{numero}",
        "run_id": elegida.run_id,
        "origen": elegida.source,
        "descripcion": elegida.description,
        "etiquetas": dict(sorted((elegida.tags or {}).items())),
        "exportado_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlflow": mlflow.__version__,
        "archivos": huellas(destino),
    }
    (destino / ARCHIVO_REGISTRO).write_text(
        json.dumps(registro, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (destino / ARCHIVO_PETICION).write_text(
        json.dumps(peticion_ejemplo(modelo), indent=2) + "\n", encoding="utf-8"
    )
    return registro


@app.command()
def main(
    pregunta: str = typer.Option("p1", "--pregunta", "-p", help="p1, p2 o p3."),
    alias: str = typer.Option(
        config.ALIAS_PRODUCCION, "--alias", help="Alias que se exporta (champion por defecto)."
    ),
    version: str | None = typer.Option(
        None, "--version", help="Numero de version exacto; si se da, se ignora el alias."
    ),
    nombre: str | None = typer.Option(
        None, "--nombre", help="Modelo registrado; por defecto el de config.MODELOS_REGISTRADOS."
    ),
):
    """Exporta el modelo de produccion de una pregunta para empaquetarlo en la imagen de la API."""
    tracking_uri = config.MLFLOW_TRACKING_URI
    if es_servidor(tracking_uri):
        verificar_servidor(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    registro = exportar(pregunta, alias, version, nombre)
    destino = destino_de(pregunta)
    logger.success(
        f"{registro['nombre']} version {registro['version']} exportado a {destino} "
        f"({len(registro['archivos'])} archivos y {ARCHIVO_REGISTRO})"
    )
    logger.info("Siguiente paso: podman build -t mibici-api:1.0 -f Containerfile .")


if __name__ == "__main__":
    app()
