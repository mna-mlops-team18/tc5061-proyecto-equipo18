"""Registro comun en MLflow para las tres preguntas (P1, P2 y P3).

Contrato de cada modulo de pregunta (``p1_regresion.py``, ``p2_clasificacion.py`` y
``p3_clustering.py``), que ``train.py`` y ``busqueda.py`` descubren e invocan::

    PREGUNTA = "P1"

    def cargar_datos() -> dict[str, pd.DataFrame]:
        '''El conjunto de la pregunta separado en entrenamiento, validacion y prueba.'''

    def correr(modelo, hiperparametros, datos=None, meta_features=None, etapa=None,
               anidado=False, nombre=None) -> str:
        '''Entrena (si aplica), evalua y registra UN run de MLflow; devuelve su run_id.'''
        with iniciar_run(PREGUNTA, modelo, etapa, hiperparametros, ...) as run:
            registrar_metricas({"mae": ...}, "validacion")
        return run.info.run_id

Los modelos de cada pregunta y sus hiperparametros por defecto viven en ``config.MODELOS``;
la metrica con que se eligen, en ``config.METRICAS_SELECCION``.

Cada run queda con:

- **Etiquetas** para filtrar en la interfaz: ``equipo``, ``pregunta``, ``etapa`` (baseline,
  candidato o busqueda), ``modelo``, ``familia_modelo``, ``data_version``, ``git_commit`` y
  ``git_sucio``.
- **Parametros:** semilla, parametros de features, ``data_version``, ``modelo`` e
  hiperparametros.
- **Artefactos** en subcarpetas: ``config/`` (corrida.json y los archivos de configuracion),
  ``datos/`` (manifiesto y metadatos de features), ``figuras/`` y, segun la pregunta,
  ``tablas/`` y ``mapas/``. El modelo se guarda con firma y ejemplo de entrada (en MLflow 3
  aparece en la seccion "Logged models" del run) y los conjuntos de datos con ``log_input``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess  # nosec B404: solo se usa en _git, con argumentos fijos y sin shell
import tempfile
import time
from types import ModuleType
import warnings

from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import skops.io

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import (
    PARTICIONES,
    asignar_particion,
    cargar_metadatos_features,
)
from tc5061_proyecto_equipo18.plots import grafica_importancias, guardar_figura

PREGUNTAS = ("P1", "P2", "P3")
ETAPAS = ("baseline", "candidato", "busqueda")
MODULOS = {
    "P1": "tc5061_proyecto_equipo18.modeling.p1_regresion",
    "P2": "tc5061_proyecto_equipo18.modeling.p2_clasificacion",
    "P3": "tc5061_proyecto_equipo18.modeling.p3_clustering",
}
COMANDO_SERVIDOR = (
    "mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000"
)


# =============================================================================
# Catalogo de modelos e hiperparametros (config.MODELOS)
# =============================================================================
def normalizar_pregunta(pregunta: str) -> str:
    """Acepta ``p1`` o ``P1`` (como en ``-P pregunta=p1``) y devuelve ``P1``."""
    normal = str(pregunta).strip().upper()
    if normal not in PREGUNTAS:
        raise ValueError(f"Pregunta desconocida: {pregunta!r}; opciones p1, p2 o p3")
    return normal


def modulo_de(pregunta: str) -> ModuleType:
    """Modulo que entrena y evalua una pregunta (cumple el contrato descrito arriba)."""
    return importlib.import_module(MODULOS[normalizar_pregunta(pregunta)])


def especificacion(pregunta: str, modelo: str) -> dict:
    """Entrada del catalogo ``config.MODELOS`` para un modelo: familia e hiperparametros."""
    catalogo = config.MODELOS[normalizar_pregunta(pregunta)]
    if modelo not in catalogo:
        raise ValueError(
            f"Modelo desconocido para {pregunta}: {modelo!r}. Opciones: {', '.join(catalogo)}"
        )
    return catalogo[modelo]


def es_baseline(pregunta: str, modelo: str) -> bool:
    return bool(especificacion(pregunta, modelo).get("baseline", False))


def baseline_de(pregunta: str) -> str:
    """Nombre del baseline de una pregunta (el modelo marcado ``baseline`` en el catalogo)."""
    pregunta = normalizar_pregunta(pregunta)
    return next(nombre for nombre in config.MODELOS[pregunta] if es_baseline(pregunta, nombre))


def convertir_valor(clave: str, valor, por_defecto):
    """Convierte el texto de ``-P clave=valor`` al tipo del valor por defecto de config.py.

    ``none`` o ``null`` dan ``None``. Un valor que no es texto se deja igual.
    """
    if not isinstance(valor, str):
        return valor
    texto = valor.strip()
    if texto.lower() in ("none", "null"):
        return None
    try:
        if isinstance(por_defecto, bool):  # antes que int: bool es un tipo de int
            if texto.lower() in ("true", "1", "si", "yes"):
                return True
            if texto.lower() in ("false", "0", "no"):
                return False
            raise ValueError(texto)
        if isinstance(por_defecto, int):
            numero = float(texto)
            if not numero.is_integer():
                raise ValueError(texto)
            return int(numero)
        if isinstance(por_defecto, float):
            return float(texto)
    except ValueError as error:
        tipo = type(por_defecto).__name__
        raise ValueError(f"{clave}={valor!r} no es un valor valido de tipo {tipo}") from error
    if por_defecto is None:
        for tipo in (int, float):
            try:
                return tipo(texto)
            except ValueError:
                pass
    return texto


def resolver_hiperparametros(pregunta: str, modelo: str, cambios: dict | None = None) -> dict:
    """Hiperparametros de la corrida: los de config.py con los ``cambios`` encima.

    Falla si un cambio no es hiperparametro del modelo (por ejemplo, un nombre mal escrito),
    para que una corrida nunca ignore en silencio lo que se le pidio.
    """
    por_defecto = especificacion(pregunta, modelo)["hiperparametros"]
    cambios = cambios or {}
    desconocidos = sorted(set(cambios) - set(por_defecto))
    if desconocidos:
        disponibles = ", ".join(por_defecto) or "ninguno (es una regla sin hiperparametros)"
        raise ValueError(
            f"{modelo} no tiene los hiperparametros {desconocidos}. Disponibles: {disponibles}"
        )
    convertidos = {
        clave: convertir_valor(clave, valor, por_defecto[clave])
        for clave, valor in cambios.items()
    }
    return {**por_defecto, **convertidos}


def cambios_respecto_a_config(pregunta: str, modelo: str, hiperparametros: dict | None) -> dict:
    """Hiperparametros cuyo valor difiere del de config.py (los que se pasaron con -P)."""
    por_defecto = especificacion(pregunta, modelo)["hiperparametros"]
    return {
        clave: valor
        for clave, valor in (hiperparametros or {}).items()
        if clave not in por_defecto or por_defecto[clave] != valor
    }


def leer_opciones_extra(argumentos: list[str]) -> dict[str, str]:
    """Convierte ``["--learning_rate", "0.05", "--max_leaf_nodes=31"]`` en un diccionario.

    Asi llegan los hiperparametros que no declara el MLproject: ``mlflow run`` pasa cada
    ``-P nombre=valor`` extra como ``--nombre valor``. Los guiones se cambian por guion bajo.
    """
    opciones: dict[str, str] = {}
    i = 0
    while i < len(argumentos):
        token = argumentos[i]
        if not token.startswith("--") or len(token) <= 2:
            raise ValueError(f"Argumento no reconocido: {token!r}. Usa --hiperparametro valor")
        clave = token[2:]
        if "=" in clave:
            clave, valor = clave.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(argumentos):
                raise ValueError(f"Falta el valor de --{clave}")
            valor = argumentos[i + 1]
            i += 2
        opciones[clave.replace("-", "_")] = valor
    return opciones


def _formato(valor) -> str:
    return f"{valor:g}" if isinstance(valor, float) else str(valor)


def comando_reproduccion(pregunta: str, modelo: str, hiperparametros: dict | None = None) -> str:
    """Comando de ``mlflow run`` que vuelve a entrenar exactamente este modelo."""
    cambios = cambios_respecto_a_config(pregunta, modelo, hiperparametros)
    extras = "".join(f" -P {clave}={_formato(valor)}" for clave, valor in cambios.items())
    return (
        f"mlflow run . -e train -P pregunta={pregunta.lower()} -P modelo={modelo}{extras} "
        "--env-manager local"
    )


# =============================================================================
# Configuracion de MLflow
# =============================================================================
def es_servidor(tracking_uri: str) -> bool:
    return str(tracking_uri).startswith(("http://", "https://"))


def verificar_servidor(tracking_uri: str, espera_seg: float = 5.0) -> None:
    """Falla con un mensaje claro si el servidor de tracking no responde."""
    try:
        respuesta = requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=espera_seg)
        respuesta.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"No hay un servidor de MLflow respondiendo en {tracking_uri}. Levantalo en otra "
            f"terminal, desde la raiz del repositorio, con:\n  {COMANDO_SERVIDOR}"
        ) from error


def configurar_mlflow(
    tracking_uri: str | None = None,
    experimento: str = config.MLFLOW_EXPERIMENTO,
    artefactos_dir: Path = config.MLFLOW_ARTEFACTOS_DIR,
) -> str:
    """Apunta MLflow al servidor del proyecto y activa el experimento del equipo.

    Con servidor (``http://...``), un experimento nuevo guarda sus artefactos donde decida el
    servidor (``mlartifacts/``). Sin servidor (solo en las pruebas) se usa ``mlruns/``. Si el
    proceso lo lanzo ``mlflow run``, comprueba que el run sea de este experimento. Devuelve el
    id del experimento.
    """
    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    if es_servidor(tracking_uri):
        verificar_servidor(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    existente = mlflow.get_experiment_by_name(experimento)
    if existente is None:
        ubicacion = None
        if not es_servidor(tracking_uri):
            artefactos_dir.mkdir(parents=True, exist_ok=True)
            ubicacion = artefactos_dir.resolve().as_uri()
        experimento_id = mlflow.create_experiment(
            experimento, artifact_location=ubicacion, tags={"equipo": config.EQUIPO}
        )
    else:
        experimento_id = existente.experiment_id
        if existente.artifact_location.startswith("mlflow-artifacts:") and not es_servidor(
            tracking_uri
        ):
            raise RuntimeError(
                f"El experimento {experimento} guarda sus artefactos en el servidor de MLflow, "
                f"pero MLFLOW_TRACKING_URI apunta a {tracking_uri}. Define "
                "MLFLOW_TRACKING_URI=http://127.0.0.1:5000 en la terminal y vuelve a correr."
            )
    _verificar_run_del_proyecto(experimento_id, experimento)
    mlflow.set_experiment(experiment_id=experimento_id)
    return experimento_id


def _verificar_run_del_proyecto(experimento_id: str, experimento: str) -> None:
    """``mlflow run`` crea el run antes de llamar al codigo: debe ser del experimento del equipo."""
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        return
    run = mlflow.MlflowClient().get_run(run_id)
    if run.info.experiment_id != experimento_id:
        raise RuntimeError(
            f"`mlflow run` creo este run en otro experimento (id {run.info.experiment_id}). "
            f"Define en la terminal MLFLOW_EXPERIMENT_NAME={experimento} (o agrega "
            f"--experiment-name {experimento}) y vuelve a correr el comando."
        )


def estado_git() -> dict[str, str]:
    """Commit actual y si hay cambios sin confirmar, para saber que codigo produjo cada run."""
    try:
        commit = _git("rev-parse", "HEAD")
        sucio = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "desconocido", "git_sucio": "desconocido"}
    return {"git_commit": commit, "git_sucio": str(sucio).lower()}


def _git(*argumentos: str) -> str:
    """Corre un comando de git de solo lectura en la raiz del repo y devuelve su salida.

    Seguridad (bandit B603/B607): se usa la ruta absoluta de git (no se busca en el PATH al
    ejecutar), sin shell, y los argumentos son fijos del codigo, nunca texto del usuario. Si git
    no esta instalado, ``estado_git`` lo trata como "desconocido".
    """
    ejecutable = shutil.which("git")
    if ejecutable is None:
        raise FileNotFoundError("git no esta instalado")
    resultado = subprocess.run(  # nosec B603: argumentos fijos, sin shell
        [ejecutable, *argumentos],
        cwd=config.PROJ_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return resultado.stdout.strip()


# =============================================================================
# Datos
# =============================================================================
def separar_particiones(df: pd.DataFrame, columna_tiempo: str | None) -> dict[str, pd.DataFrame]:
    """Separa un dataset en entrenamiento, validacion y prueba con el corte temporal de config.

    Con ``columna_tiempo=None`` se usa la columna ``periodo`` que ya trae el dataset (P3, cuyos
    perfiles se calculan por separado en cada periodo).
    """
    if columna_tiempo is None:
        etiquetas = df["periodo"]
    else:
        etiquetas = asignar_particion(df[columna_tiempo])
    return {
        particion: df[etiquetas == particion].reset_index(drop=True) for particion in PARTICIONES
    }


def cargar_dataset(ruta: Path, columna_tiempo: str | None) -> dict[str, pd.DataFrame]:
    """Lee el Parquet de una pregunta (lo escribe features.py) y lo separa por particion."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta: mlflow run . -e features --env-manager local"
        )
    return separar_particiones(pd.read_parquet(ruta), columna_tiempo)


def muestrear(df: pd.DataFrame, max_filas: int, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Muestra aleatoria reproducible de a lo mas ``max_filas`` filas (acota la memoria)."""
    if len(df) <= max_filas:
        return df
    return df.sample(n=max_filas, random_state=semilla).reset_index(drop=True)


# =============================================================================
# Runs
# =============================================================================
def nombre_run(pregunta: str, etapa: str, modelo: str, hiperparametros: dict | None = None) -> str:
    """Nombre legible del run: ``p1-candidato-hgb_poisson`` mas los hiperparametros que se
    cambiaron respecto a config.py, p. ej. ``p1-candidato-hgb_poisson-learning_rate0.05``."""
    cambios = cambios_respecto_a_config(pregunta, modelo, hiperparametros)
    sufijo = "".join(f"-{clave}{_formato(valor)}" for clave, valor in cambios.items())
    return f"{pregunta.lower()}-{etapa}-{modelo}{sufijo}"


@contextmanager
def iniciar_run(
    pregunta: str,
    modelo: str,
    etapa: str | None = None,
    hiperparametros: dict | None = None,
    meta_features: dict | None = None,
    nombre: str | None = None,
    anidado: bool = False,
    parametros: dict | None = None,
    descripcion: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Abre un run de MLflow con las etiquetas, parametros y artefactos comunes.

    Si el proceso lo lanzo ``mlflow run``, retoma el run que creo el proyecto. Con
    ``anidado=True`` crea un run hijo del run activo (busquedas). Registra la version de los
    datos (``data_version``), que se valida contra el manifiesto al leer los metadatos de
    features: si los datos o config.py cambiaron, falla antes de entrenar.
    """
    pregunta = normalizar_pregunta(pregunta)
    spec = especificacion(pregunta, modelo)
    etapa = etapa or ("baseline" if spec.get("baseline") else "candidato")
    if etapa not in ETAPAS:
        raise ValueError(f"Etapa desconocida: {etapa}; opciones {ETAPAS}")
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    tags = {
        "equipo": config.EQUIPO,
        "pregunta": pregunta,
        "etapa": etapa,
        "modelo": modelo,
        "familia_modelo": spec["familia"],
        "data_version": meta["data_version"],
        **estado_git(),
    }
    if descripcion is None and hiperparametros is not None:
        descripcion = "Reproducir: " + comando_reproduccion(pregunta, modelo, hiperparametros)
    with mlflow.start_run(
        run_name=nombre or nombre_run(pregunta, etapa, modelo, hiperparametros),
        tags=tags,
        nested=anidado,
        description=descripcion,
    ) as run:
        if anidado:
            heredar_origen(run.info.run_id)
        registrar_parametros(
            {
                "pregunta": pregunta.lower(),
                "semilla": config.SEMILLA,
                **{f"features.{clave}": valor for clave, valor in meta["parametros"].items()},
                **(hiperparametros or {}),
                **(parametros or {}),
                "data_version": meta["data_version"],
                "modelo": modelo,
            }
        )
        registrar_configuracion(pregunta, modelo, etapa, hiperparametros, meta, parametros)
        mlflow.log_dict(meta, "datos/features_meta.json")
        if config.MANIFIESTO_DATOS.exists():
            mlflow.log_artifact(str(config.MANIFIESTO_DATOS), "datos")
        yield run


# Etiquetas con que MLflow indica como se lanzo un run (``mlflow run`` las pone en el que crea)
ETIQUETAS_ORIGEN = (
    "mlflow.source.type",
    "mlflow.source.name",
    "mlflow.project.entryPoint",
    "mlflow.project.backend",
    "mlflow.project.env",
)


def heredar_origen(run_id: str) -> None:
    """Copia a una corrida hija las etiquetas de origen de su padre.

    ``mlflow run . -e search`` crea la corrida padre y MLflow la marca como ``PROJECT``; las hijas
    se abren dentro de ese mismo proceso y MLflow las marcaria como ``LOCAL``, aunque tambien las
    lanzo el MLflow Project.
    """
    cliente = mlflow.MlflowClient()
    padre = cliente.get_run(run_id).data.tags.get("mlflow.parentRunId")
    if not padre:
        return
    etiquetas = cliente.get_run(padre).data.tags
    for clave in ETIQUETAS_ORIGEN:
        if clave in etiquetas:
            cliente.set_tag(run_id, clave, etiquetas[clave])


def lanzado_por_proyecto(run_id: str) -> bool:
    """True si el run lo lanzo el MLflow Project (``mlflow run`` o su API)."""
    etiquetas = mlflow.MlflowClient().get_run(run_id).data.tags
    return etiquetas.get("mlflow.source.type") == "PROJECT"


def _run_activo() -> mlflow.ActiveRun:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("Llama a esta funcion dentro de `with iniciar_run(...)`")
    return run


def registrar_parametros(parametros: dict) -> None:
    """Registra parametros en el run activo, salvo los que ya tiene.

    ``mlflow run`` registra por su cuenta los parametros del comando (``pregunta``, ``modelo`` y
    cada ``-P`` extra) con el texto tal como se escribio, y MLflow no permite cambiar el valor de
    un parametro ya registrado; por eso aqui solo se agregan los que faltan.
    """
    run = _run_activo()
    existentes = mlflow.get_run(run.info.run_id).data.params
    nuevos = {clave: valor for clave, valor in parametros.items() if clave not in existentes}
    if nuevos:
        mlflow.log_params(nuevos)


def registrar_configuracion(
    pregunta: str,
    modelo: str,
    etapa: str,
    hiperparametros: dict | None,
    meta: dict,
    parametros: dict | None = None,
) -> None:
    """Guarda en ``config/`` lo necesario para repetir la corrida.

    ``config/corrida.json`` resume la corrida (modelo, hiperparametros, particion, version de
    datos, commit, comando de reproduccion y versiones de librerias) y junto a el se copian
    MLproject, python_env.yaml, requirements.txt y config.py tal como estaban al correr.
    """
    metrica, sentido = config.METRICAS_SELECCION[pregunta]
    corrida = {
        "pregunta": pregunta,
        "modelo": modelo,
        "familia_modelo": especificacion(pregunta, modelo)["familia"],
        "etapa": etapa,
        "hiperparametros": hiperparametros or {},
        "parametros": parametros or {},
        "semilla": config.SEMILLA,
        "particion": {
            "entrenamiento": f"{config.MES_INICIO} a {config.FIN_ENTRENAMIENTO}",
            "validacion": f"despues de {config.FIN_ENTRENAMIENTO} a {config.FIN_VALIDACION}",
            "prueba": f"despues de {config.FIN_VALIDACION} a {config.MES_FIN}",
        },
        "metrica_seleccion": {"nombre": metrica, "sentido": sentido},
        "data_version": meta["data_version"],
        "features": meta["parametros"],
        "git": estado_git(),
        "experimento": config.MLFLOW_EXPERIMENTO,
        "lanzado_por_mlflow_project": lanzado_por_proyecto(_run_activo().info.run_id),
        "versiones": _versiones(),
    }
    if hiperparametros is not None:
        corrida["comando_reproduccion"] = comando_reproduccion(pregunta, modelo, hiperparametros)
    mlflow.log_dict(corrida, "config/corrida.json")
    for archivo in config.ARCHIVOS_CONFIGURACION:
        ruta = config.PROJ_ROOT / archivo
        if ruta.exists():
            mlflow.log_artifact(str(ruta), "config")


def _versiones() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versiones = {"python": platform.python_version()}
    for paquete in ("mlflow", "scikit-learn", "pandas", "numpy", "skops"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = "no instalado"
    return versiones


def registrar_metricas(metricas: dict[str, float], particion: str) -> None:
    """Registra metricas con el prefijo de la particion: ``{"mae": 1.2}`` -> ``validacion_mae``."""
    if particion not in PARTICIONES:
        raise ValueError(f"Particion desconocida: {particion}; opciones {PARTICIONES}")
    mlflow.log_metrics(
        {f"{particion}_{nombre}": float(valor) for nombre, valor in metricas.items()}
    )


@contextmanager
def cronometro() -> Iterator[dict]:
    """Mide el tiempo de un bloque: ``with cronometro() as t: ...`` -> ``t["segundos"]``."""
    medida = {"segundos": 0.0}
    inicio = time.perf_counter()
    try:
        yield medida
    finally:
        medida["segundos"] = time.perf_counter() - inicio


@contextmanager
def vigilar_convergencia() -> Iterator[list]:
    """Junta los avisos que salgan al entrenar, para registrar en MLflow si hubo uno de no
    convergencia (scikit-learn solo los muestra en la terminal).

    ``with vigilar_convergencia() as avisos: estimador.fit(...)``; al salir del bloque los avisos
    se vuelven a mostrar igual que antes.
    """
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        yield avisos
    for aviso in avisos:
        warnings.warn_explicit(aviso.message, aviso.category, aviso.filename, aviso.lineno)


def registrar_convergencia(estimador, avisos: list) -> None:
    """Registra cuantas iteraciones uso el ajuste y si scikit-learn aviso que no convergio.

    En un ``Pipeline`` se lee el ultimo paso (el modelo). ``entrenamiento_iteraciones`` es el
    maximo de ``n_iter_`` (la logistica da uno por clase) y ``entrenamiento_sin_convergencia``
    vale 1 si hubo un ``ConvergenceWarning`` (llego a ``max_iter`` sin converger) y 0 si no.
    """
    final = estimador[-1] if isinstance(estimador, Pipeline) else estimador
    sin_convergencia = any(issubclass(aviso.category, ConvergenceWarning) for aviso in avisos)
    metricas = {"sin_convergencia": float(sin_convergencia)}
    if getattr(final, "n_iter_", None) is not None:
        metricas["iteraciones"] = float(np.max(final.n_iter_))
    registrar_metricas(metricas, "entrenamiento")


def registrar_costos(
    entrenamiento_seg: float | None = None,
    inferencia_seg: float | None = None,
    filas_inferencia: int | None = None,
) -> None:
    """Costo de entrenar y de predecir (criterio de seleccion junto con el desempeno).

    ``costo_inferencia_ms_por_1000_filas`` se mide al predecir validacion; depende de la
    computadora, asi que sirve para comparar runs de la misma maquina.
    """
    costos = {}
    if entrenamiento_seg is not None:
        costos["costo_entrenamiento_seg"] = entrenamiento_seg
    if inferencia_seg is not None and filas_inferencia:
        costos["costo_inferencia_ms_por_1000_filas"] = 1e6 * inferencia_seg / filas_inferencia
    if costos:
        mlflow.log_metrics(costos)


def registrar_figura(figura: Figure, nombre: str, carpeta: str = "figuras") -> None:
    """Sube la figura al run activo como ``figuras/<nombre>.png`` (sin copia en el repo)."""
    _run_activo()
    with tempfile.TemporaryDirectory() as temporal:
        ruta = guardar_figura(figura, Path(temporal) / f"{nombre}.png")
        mlflow.log_artifact(str(ruta), carpeta)


def registrar_tabla(tabla: pd.DataFrame, nombre: str, carpeta: str = "tablas") -> None:
    """Sube una tabla al run activo como ``tablas/<nombre>.csv``."""
    _run_activo()
    mlflow.log_text(tabla.to_csv(index=False), f"{carpeta}/{nombre}.csv")


def registrar_conjuntos(
    datos: dict[str, pd.DataFrame],
    pregunta: str,
    ruta: Path,
    columnas: list[str],
    objetivo: str | None = None,
    data_version: str | None = None,
) -> None:
    """Registra con ``mlflow.log_input`` los conjuntos de entrenamiento, validacion y prueba.

    MLflow guarda el esquema, el numero de filas y una huella (digest) de cada conjunto, no los
    datos. El nombre lleva la ``data_version``, que identifica los datos exactos. Solo hay
    agregados por estacion: ningun dato personal.
    """
    _run_activo()
    fuente = ruta.relative_to(config.PROJ_ROOT).as_posix()
    seleccion = [*columnas, objetivo] if objetivo else list(columnas)
    for particion in PARTICIONES:
        if datos[particion].empty:
            continue
        tabla = datos[particion][seleccion]
        with warnings.catch_warnings():
            # MLflow avisa que una ruta local puede leerse como dos tipos de fuente equivalentes
            # y que las columnas enteras no admiten faltantes (las variables no tienen faltantes)
            warnings.filterwarnings("ignore", message=".*can be interpreted in multiple ways.*")
            warnings.filterwarnings(
                "ignore", message=".*Inferred schema contains integer column.*"
            )
            conjunto = mlflow.data.from_pandas(
                tabla,
                source=fuente,
                name=f"{pregunta.lower()}-{particion}-{data_version or 'sin_version'}",
                targets=objetivo,
                digest=huella_conjunto(tabla, data_version),
            )
            mlflow.log_input(conjunto, context=particion)


def huella_conjunto(tabla: pd.DataFrame, data_version: str | None = None) -> str:
    """Huella corta (8 caracteres) de un conjunto: primeras 10 mil filas, filas y columnas.

    Es la misma idea que el ``digest`` de MLflow, pero se calcula aqui porque MLflow 3.17
    recorre todas las celdas de la tabla para hacerlo, lo que con los millones de filas de P1
    tarda y ocupa memoria. Incluye la ``data_version``, que es la huella fuerte de los datos.
    """
    huella = hashlib.md5(usedforsecurity=False)
    huella.update(pd.util.hash_pandas_object(tabla.head(10_000), index=False).to_numpy().tobytes())
    huella.update(np.int64(len(tabla)).tobytes())
    huella.update("|".join(map(str, tabla.columns)).encode("utf-8"))
    huella.update(str(data_version).encode("utf-8"))
    return huella.hexdigest()[:8]


def registrar_importancias(modelo, X: pd.DataFrame, y, scoring: str, titulo: str) -> None:
    """Importancia por permutacion en datos que el modelo no vio (caida de la metrica)."""
    importancia = permutation_importance(
        modelo, X, y, scoring=scoring, n_repeats=3, random_state=config.SEMILLA
    )
    importancias = pd.Series(importancia.importances_mean, index=X.columns)
    registrar_figura(grafica_importancias(importancias, titulo), "importancias")


def registrar_modelo(modelo, ejemplo: pd.DataFrame) -> str:
    """Registra un estimador de scikit-learn en el run activo y devuelve su URI.

    MLflow lo guarda en formato skops (no ejecuta codigo arbitrario al cargarlo) con la firma
    de entrada y salida inferida de ``ejemplo`` y un ejemplo de entrada de 5 filas. Tambien
    registra el tamanio del modelo serializado (``costo_tamano_modelo_mb``).
    """
    _run_activo()
    with warnings.catch_warnings():
        # Aviso de MLflow sobre columnas enteras con faltantes: las variables no tienen faltantes
        warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
        firma = infer_signature(ejemplo, modelo.predict(ejemplo))
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=config.SKOPS_TIPOS_CONFIABLES,
            signature=firma,
            input_example=ejemplo.head(5),
        )
    mlflow.log_metric("costo_tamano_modelo_mb", len(skops.io.dumps(modelo)) / 1e6)
    return info.model_uri


def resumen_json(datos: dict) -> str:
    """Texto JSON compacto y estable (para parametros como la rejilla de una busqueda)."""
    return json.dumps(datos, sort_keys=True, default=str)
