"""Carga de los modelos de P1, P2 y P3 y prediccion, separadas de la API.

De donde sale el modelo de cada pregunta (variable de entorno ``MODELO_URI_P1``, ``MODELO_URI_P2``
o ``MODELO_URI_P3``):

- **Carpeta local** (por defecto ``models/servicio/p1``, ``p2`` o ``p3``; es la que copia el
  Containerfile): el modelo exportado del Model Registry con
  ``python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1``. Su ``registro.json``
  dice de que modelo registrado y de que version viene.
- **URI del Model Registry** (``models:/<nombre>@<alias>`` o ``models:/<nombre>/<version>``):
  se descarga al arrancar; requiere ``MLFLOW_TRACKING_URI``. Primero se resuelve el alias a un
  numero de version y se carga esa version exacta.

Al cargar se comprueba que la firma del modelo tenga exactamente las variables de la API y se
hace una autoprueba: el modelo debe predecir su propio ejemplo de entrada. Un modelo que no pasa
estas pruebas no se sirve (es mejor responder 503 que responder con un modelo equivocado).

MLflow se importa dentro de las funciones de carga: ``armar_entrada``, ``validar_salida`` y
``validar_clases`` se pueden probar sin MLflow instalado.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PREGUNTAS = ("p1", "p2", "p3")
RAIZ = Path(__file__).resolve().parents[1]  # raiz del repo; /app dentro de la imagen
MODELOS_DIR = RAIZ / "models" / "servicio"  # una carpeta por pregunta: p1/, p2/, p3/
CLASES_VALIDAS = {"p2": {0, 1, 2}}  # P3 no fija el numero de grupos: basta que sea un entero >= 0


def variable_de_entorno(pregunta: str) -> str:
    """Nombre de la variable que elige el modelo de una pregunta: ``MODELO_URI_P1`` para P1."""
    return f"MODELO_URI_{pregunta.upper()}"


ARCHIVO_REGISTRO = "registro.json"  # lo escribe modeling/exportar.py


class ErrorModelo(RuntimeError):
    """El modelo no se pudo cargar o no coincide con el contrato de la API."""


class ErrorPrediccion(RuntimeError):
    """El modelo devolvio algo que no es una prediccion valida."""


@dataclass(frozen=True)
class Identidad:
    """Nombre, version y linaje del modelo cargado (lo que la API reporta)."""

    nombre: str
    version: str
    alias: str | None = None
    run_id: str | None = None
    exportado_en: str | None = None
    etiquetas: dict[str, str] = field(default_factory=dict)


@dataclass
class ModeloServido:
    """Modelo listo para predecir: el modelo, su identidad y los tipos de su firma."""

    modelo: object  # mlflow.pyfunc.PyFuncModel; en las pruebas, cualquier objeto con predict
    identidad: Identidad
    tipos: dict[str, np.dtype]  # columna -> tipo de la firma, en el orden de la firma
    origen: str  # carpeta o URI de donde se cargo
    pregunta: str = "p1"  # decide como se revisa la salida: valores (P1) o clases (P2 y P3)

    def predecir(self, filas: list[dict]) -> np.ndarray:
        """Una prediccion por fila: valores finitos y no negativos (P1) o clases enteras."""
        entrada = armar_entrada(filas, self.tipos)
        predicho = self.modelo.predict(entrada)
        if self.pregunta == "p1":
            return validar_salida(predicho, len(filas))
        return validar_clases(predicho, len(filas), CLASES_VALIDAS.get(self.pregunta))

    def autoprueba(self) -> int:
        """Predice el ejemplo de entrada guardado con el modelo; devuelve cuantas filas."""
        ejemplo = getattr(self.modelo, "input_example", None)
        if ejemplo is None:
            return 0
        return len(self.predecir(pd.DataFrame(ejemplo).to_dict(orient="records")))


# =============================================================================
# Entrada y salida (sin MLflow)
# =============================================================================
def armar_entrada(filas: list[dict], tipos: dict[str, np.dtype]) -> pd.DataFrame:
    """DataFrame con las columnas, el orden y los tipos exactos de la firma del modelo.

    MLflow compara la entrada con la firma y no convierte por su cuenta un ``float64`` en
    ``float32`` ni un ``int64`` en ``int32``, porque podria perder informacion. Pydantic ya
    reviso los rangos, asi que aqui se convierte a los tipos del entrenamiento sin perder nada.
    """
    columnas = list(tipos)
    entrada = pd.DataFrame.from_records(filas, columns=columnas)
    faltan = [columna for columna in columnas if entrada[columna].isna().any()]
    if faltan:
        raise ValueError(f"Faltan valores de las variables {faltan}")
    return entrada.astype(tipos)


def validar_salida(predicho, n_filas: int) -> np.ndarray:
    """Comprueba la salida del modelo: una prediccion finita por fila; recorta negativos a 0.

    Las salidas de una estacion no pueden ser negativas. Un modelo con perdida Poisson nunca las
    da; uno con perdida cuadratica si podria, y la API las recorta a 0 en lugar de publicarlas.
    """
    valores = np.asarray(predicho, dtype=float).reshape(-1)
    if valores.shape[0] != n_filas:
        raise ErrorPrediccion(
            f"El modelo devolvio {valores.shape[0]} valores para {n_filas} filas"
        )
    if not np.isfinite(valores).all():
        raise ErrorPrediccion("El modelo devolvio valores no finitos (NaN o infinito)")
    return np.clip(valores, 0.0, None)


def validar_clases(predicho, n_filas: int, validas: set[int] | None = None) -> np.ndarray:
    """Comprueba la salida de un clasificador o agrupador: una clase entera por fila.

    Con ``validas`` (P2: 0, 1 o 2) cualquier otra clase es un error del modelo; sin el (P3: el
    numero de grupos lo define el entrenamiento) basta con que sea un entero no negativo.
    """
    valores = np.asarray(predicho).reshape(-1)
    if valores.shape[0] != n_filas:
        raise ErrorPrediccion(
            f"El modelo devolvio {valores.shape[0]} valores para {n_filas} filas"
        )
    if not np.isfinite(valores.astype(float)).all():
        raise ErrorPrediccion("El modelo devolvio valores no finitos (NaN o infinito)")
    clases = valores.astype(np.int64)
    if (clases != valores).any() or (clases < 0).any():
        raise ErrorPrediccion("El modelo devolvio clases que no son enteros no negativos")
    if validas is not None and not set(clases.tolist()) <= validas:
        raise ErrorPrediccion(
            f"El modelo devolvio clases fuera de {sorted(validas)}: {sorted(set(clases.tolist()))}"
        )
    return clases


# =============================================================================
# Carga (con MLflow)
# =============================================================================
def tipos_de_firma(modelo) -> dict[str, np.dtype]:
    """Columnas y tipos de la firma de entrada del modelo (``infer_signature`` al entrenar)."""
    esquema = modelo.metadata.get_input_schema()
    if esquema is None or not esquema.has_input_names():
        raise ErrorModelo("El modelo no tiene firma de entrada con nombres de columna")
    return {columna.name: np.dtype(columna.type.to_numpy()) for columna in esquema.inputs}


def verificar_columnas(tipos: dict[str, np.dtype], esperadas: list[str]) -> None:
    """La firma del modelo debe tener exactamente las variables que valida la API."""
    sobran = sorted(set(esperadas) - set(tipos))
    faltan = sorted(set(tipos) - set(esperadas))
    if sobran or faltan:
        raise ErrorModelo(
            "La firma del modelo no coincide con la API. "
            f"La API envia y el modelo no espera: {sobran}. "
            f"El modelo espera y la API no envia: {faltan}. "
            "Exporta el modelo de P1 o actualiza api/esquemas.py."
        )


def identidad_local(carpeta: Path, pregunta: str = "p1") -> Identidad:
    """Identidad del modelo exportado, leida de su ``registro.json``."""
    ruta = carpeta / ARCHIVO_REGISTRO
    if not ruta.is_file():
        raise ErrorModelo(
            f"Falta {ruta}: sin el no se sabe que version del Model Registry se sirve. "
            "Exporta el modelo con: python -m tc5061_proyecto_equipo18.modeling.exportar "
            f"--pregunta {pregunta}"
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return Identidad(
        nombre=datos["nombre"],
        version=str(datos["version"]),
        alias=datos.get("alias"),
        run_id=datos.get("run_id"),
        exportado_en=datos.get("exportado_en"),
        etiquetas={clave: str(valor) for clave, valor in datos.get("etiquetas", {}).items()},
    )


def identidad_registry(uri: str) -> Identidad:
    """Resuelve ``models:/<nombre>@<alias>`` o ``models:/<nombre>/<version>`` a una version."""
    import mlflow

    referencia = uri.removeprefix("models:/")
    cliente = mlflow.MlflowClient()
    if "@" in referencia:
        nombre, alias = referencia.split("@", 1)
        version = cliente.get_model_version_by_alias(nombre, alias)
    elif "/" in referencia:
        nombre, numero = referencia.split("/", 1)
        alias = None
        version = cliente.get_model_version(nombre, numero)
    else:
        raise ErrorModelo(f"URI no valida: {uri}. Usa models:/<nombre>@<alias> o /<version>")
    return Identidad(
        nombre=nombre,
        version=str(version.version),
        alias=alias,
        run_id=version.run_id,
        etiquetas={clave: str(valor) for clave, valor in (version.tags or {}).items()},
    )


def cargar_modelo(
    origen: str | os.PathLike | None = None,
    columnas: list[str] | None = None,
    pregunta: str = "p1",
) -> ModeloServido:
    """Carga el modelo de ``origen`` (o de ``MODELO_URI_<PREGUNTA>``, o de ``models/servicio``).

    Con ``columnas`` exige que la firma tenga exactamente esas variables. Siempre hace la
    autoprueba con el ejemplo de entrada.
    """
    import mlflow.pyfunc

    origen = str(
        origen or os.getenv(variable_de_entorno(pregunta)) or MODELOS_DIR / pregunta.lower()
    )
    if origen.startswith("models:/"):
        identidad = identidad_registry(origen)
        uri = f"models:/{identidad.nombre}/{identidad.version}"  # la version exacta resuelta
    else:
        carpeta = Path(origen)
        if not (carpeta / "MLmodel").is_file():
            raise ErrorModelo(
                f"No hay un modelo de MLflow en {carpeta}. Exportalo del Model Registry con: "
                f"python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta {pregunta}"
            )
        identidad = identidad_local(carpeta, pregunta)
        uri = str(carpeta)

    modelo = mlflow.pyfunc.load_model(uri)
    tipos = tipos_de_firma(modelo)
    if columnas is not None:
        verificar_columnas(tipos, columnas)
    servido = ModeloServido(
        modelo=modelo, identidad=identidad, tipos=tipos, origen=origen, pregunta=pregunta
    )
    servido.autoprueba()
    return servido
