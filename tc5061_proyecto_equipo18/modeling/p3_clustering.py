"""P3 - Clustering: que grupos de estaciones comparten patrones de uso y donde se ubican.

Modelos del catalogo ``config.P3_MODELOS``, todos evaluados con las mismas metricas y en el
mismo espacio que en la entrega 1:

- ``cuartiles_volumen`` (baseline): grupos por cuartiles del volumen diario, sin aprendizaje.
  Es la segmentacion que haria cualquiera sin modelo ("estaciones muy, medio y poco usadas");
  los demas deben aportar algo que el volumen solo no explica.
- ``kmeans_pca``: K-means con ``n_clusters`` grupos (la busqueda prueba k de 2 a 8).

Todos agrupan en el mismo espacio: ``StandardScaler -> PCA`` (``config.P3_VARIANZA_PCA`` de la
varianza) ajustado solo con entrenamiento. Asi las metricas de todos son comparables.

Anti-fuga: el escalador, la PCA, los cortes de cuartiles y los grupos se ajustan SOLO con los
perfiles de entrenamiento. Validacion y prueba solo se asignan al grupo (o corte) mas cercano y
sirven para medir la calidad y la estabilidad de los grupos en periodos posteriores. La
ubicacion (latitud, longitud) no entra al clustering: los grupos se forman por uso y despues se
ubican en el mapa, que es lo que pregunta P3.

Las metricas se llaman ``<periodo>_<metrica>`` (por ejemplo ``validacion_silhouette``), igual
que en P1 y P2.

Uso::

    mlflow run . -e train -P pregunta=p3 -P modelo=kmeans_pca --env-manager local
    mlflow run . -e search -P pregunta=p3 --env-manager local
"""

from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import PARTICIONES, cargar_metadatos_features
from tc5061_proyecto_equipo18.modeling.comun import (
    cargar_dataset,
    cronometro,
    es_baseline,
    iniciar_run,
    registrar_conjuntos,
    registrar_costos,
    registrar_figura,
    registrar_metricas,
    registrar_modelo,
    registrar_tabla,
    resolver_hiperparametros,
)
from tc5061_proyecto_equipo18.plots import (
    grafica_mapa_grupos,
    grafica_perfiles_grupo,
    mapa_html_grupos,
)

PREGUNTA = "P3"
VARIABLES_RESUMEN = ["log_viajes_dia", "ratio_fin_semana", "balance_neto"]
COLUMNAS_MAPA = ["estacion_id", "codigo", "nombre", "municipio", "latitud", "longitud"]
TITULOS = {"cuartiles_volumen": "Cuartiles de volumen", "kmeans_pca": "K-means"}

# Asigna un grupo a cada estacion de un periodo: (perfiles, perfiles transformados) -> grupos
Asignador = Callable[[pd.DataFrame, np.ndarray], np.ndarray]


# =============================================================================
# Variables y ajustes (solo con entrenamiento)
# =============================================================================
def columnas_variables(df: pd.DataFrame) -> list[str]:
    """Variables de clustering: perfiles horarios (``sal_*`` / ``lle_*``) y el resumen por estacion.

    La ubicacion (latitud/longitud), la capacidad y los identificadores NO entran: los grupos se
    forman por uso y luego se mapean.
    """
    perfil = [c for c in df.columns if c.startswith(("sal_", "lle_"))]
    return perfil + VARIABLES_RESUMEN


def ajustar_preprocesamiento(entrenamiento: pd.DataFrame, columnas: list[str]) -> Pipeline:
    """Escalador + PCA (``config.P3_VARIANZA_PCA`` de la varianza) ajustados solo con entrenamiento."""
    prep = Pipeline(
        [
            ("escala", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=config.P3_VARIANZA_PCA,
                    svd_solver="full",
                    random_state=config.SEMILLA,
                ),
            ),
        ]
    )
    return prep.fit(entrenamiento[columnas])


def cortes_cuartiles(entrenamiento: pd.DataFrame, grupos: int) -> np.ndarray:
    """Cortes de volumen diario (escala log) calculados solo con entrenamiento."""
    cuantiles = np.linspace(0, 1, grupos + 1)[1:-1]
    return entrenamiento["log_viajes_dia"].quantile(cuantiles).to_numpy()


def asignar_por_cortes(df: pd.DataFrame, cortes: np.ndarray) -> np.ndarray:
    """Grupo de volumen (0 = menos usadas) con los cortes de entrenamiento."""
    return np.digitize(df["log_viajes_dia"].to_numpy(), cortes)


def ajustar_agrupador(modelo: str, hiperparametros: dict, z_entrenamiento: np.ndarray):
    """Ajusta el modelo de grupos en el espacio de entrenamiento y devuelve un estimador con
    ``predict`` (el que asigna estaciones de cualquier periodo)."""
    if modelo == "kmeans_pca":
        return KMeans(**hiperparametros, random_state=config.SEMILLA).fit(z_entrenamiento)
    raise ValueError(f"P3 no sabe construir el modelo {modelo!r}")


# =============================================================================
# Evaluacion
# =============================================================================
def metricas_grupos(z: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Calidad interna de los grupos en un periodo (requiere entre 2 y n-1 grupos).

    - ``silhouette`` (-1 a 1, mayor es mejor): cohesion y separacion.
    - ``davies_bouldin`` (menor es mejor): parecido promedio entre cada grupo y el mas cercano.
    - ``calinski_harabasz`` (mayor es mejor): dispersion entre grupos frente a dentro de ellos.
    """
    n_grupos = len(np.unique(etiquetas))
    if not 2 <= n_grupos < len(etiquetas):
        return {}
    return {
        "silhouette": float(silhouette_score(z, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(z, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(z, etiquetas)),
    }


def evaluar(
    prep: Pipeline,
    columnas: list[str],
    perfiles: dict[str, pd.DataFrame],
    asignar: Asignador,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, dict[str, np.ndarray]]:
    """Asigna grupos en cada periodo y calcula metricas de calidad y de estabilidad.

    - Calidad (``metricas_grupos``) en cada periodo, en el espacio del preprocesamiento de
      entrenamiento, para que todos los modelos se comparen igual.
    - ``grupos``: numero de grupos distintos asignados en el periodo.
    - ``estabilidad_ari`` (validacion y prueba): Adjusted Rand Index entre el grupo de
      entrenamiento y el asignado en ese periodo, para las estaciones presentes en ambos
      (1 = cada estacion conserva su grupo; 0 = como al azar). Un grupo que cambia cada
      trimestre no sirve para planear.

    Devuelve las metricas por periodo, una tabla con el grupo de cada estacion y periodo, y las
    etiquetas por periodo.
    """
    z = {p: prep.transform(df[columnas]) for p, df in perfiles.items()}
    etiquetas = {p: np.asarray(asignar(df, z[p])) for p, df in perfiles.items()}
    entrenamiento = perfiles["entrenamiento"]
    referencia = pd.Series(etiquetas["entrenamiento"], index=entrenamiento["estacion_id"])

    metricas: dict[str, dict[str, float]] = {}
    for periodo, df in perfiles.items():
        metricas[periodo] = {
            "estaciones": float(len(df)),
            "grupos": float(len(np.unique(etiquetas[periodo]))),
            **metricas_grupos(z[periodo], etiquetas[periodo]),
        }
        if periodo != "entrenamiento":
            actual = pd.Series(etiquetas[periodo], index=df["estacion_id"])
            comunes = referencia.index.intersection(actual.index)
            if len(comunes) > 1:
                metricas[periodo]["estabilidad_ari"] = float(
                    adjusted_rand_score(referencia.loc[comunes], actual.loc[comunes])
                )

    asignaciones = pd.concat(
        [
            df[COLUMNAS_MAPA].assign(periodo=periodo, grupo=etiquetas[periodo])
            for periodo, df in perfiles.items()
        ],
        ignore_index=True,
    )
    return metricas, asignaciones, etiquetas


def resumen_grupos(entrenamiento: pd.DataFrame, grupos: np.ndarray) -> pd.DataFrame:
    """Tamanio y uso promedio de cada grupo en entrenamiento (para interpretarlos)."""
    return (
        entrenamiento[VARIABLES_RESUMEN]
        .assign(viajes_dia=np.expm1(entrenamiento["log_viajes_dia"]), grupo=grupos)
        .groupby("grupo")
        .agg(
            estaciones=("viajes_dia", "size"),
            viajes_dia_medio=("viajes_dia", "mean"),
            ratio_fin_semana=("ratio_fin_semana", "mean"),
            balance_neto=("balance_neto", "mean"),
        )
        .round(3)
        .reset_index()
    )


# =============================================================================
# Runs
# =============================================================================
def registrar_evaluacion(
    perfiles: dict[str, pd.DataFrame],
    metricas: dict[str, dict[str, float]],
    asignaciones: pd.DataFrame,
    etiquetas: dict[str, np.ndarray],
    titulo: str,
) -> None:
    """Metricas por periodo y artefactos del run activo: mapa, perfiles, resumen y asignaciones."""
    for periodo, valores in metricas.items():
        registrar_metricas(valores, periodo)

    entrenamiento = perfiles["entrenamiento"].assign(grupo=etiquetas["entrenamiento"])
    registrar_figura(grafica_mapa_grupos(entrenamiento, f"{titulo}: ubicacion"), "mapa_grupos")
    registrar_figura(
        grafica_perfiles_grupo(entrenamiento, f"{titulo}: perfil horario"), "perfiles_grupo"
    )
    mlflow.log_text(mapa_html_grupos(entrenamiento, titulo), "mapas/mapa_grupos.html")
    registrar_tabla(
        resumen_grupos(perfiles["entrenamiento"], etiquetas["entrenamiento"]), "resumen_grupos"
    )
    registrar_tabla(asignaciones, "asignaciones")


def preparar(datos: dict[str, pd.DataFrame]) -> tuple[list[str], Pipeline, dict]:
    """Variables, preprocesamiento de entrenamiento y parametros comunes a todos los runs."""
    vacios = [p for p in PARTICIONES if datos[p].empty]
    if vacios:
        raise RuntimeError(f"No hay perfiles de P3 para: {vacios}. Revisa features.")
    columnas = columnas_variables(datos["entrenamiento"])
    prep = ajustar_preprocesamiento(datos["entrenamiento"], columnas)
    parametros = {
        "varianza_pca": config.P3_VARIANZA_PCA,
        "n_componentes_pca": int(prep.named_steps["pca"].n_components_),
        "n_variables": len(columnas),
    }
    return columnas, prep, parametros


def cargar_datos() -> dict[str, pd.DataFrame]:
    """Perfiles de P3 (los escribe features.py) separados por la columna ``periodo``."""
    return cargar_dataset(config.DATASET_P3, columna_tiempo=None)


def correr(
    modelo: str,
    hiperparametros: dict | None = None,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
    etapa: str | None = None,
    anidado: bool = False,
    nombre: str | None = None,
) -> str:
    """Agrupa, evalua y registra un run de P3; devuelve su run_id.

    ``hiperparametros`` son los ya resueltos (config.py con los cambios de ``-P``); si no se
    pasan se usan los de config.py. El baseline no registra modelo (es una regla de cortes);
    los demas registran ``Pipeline(prep, agrupador)``, que recibe el perfil de una estacion y
    devuelve su grupo.
    """
    if hiperparametros is None:
        hiperparametros = resolver_hiperparametros(PREGUNTA, modelo)
    if datos is None:
        datos = cargar_datos()
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    columnas, prep, parametros = preparar(datos)
    titulo = TITULOS.get(modelo, modelo)
    baseline = es_baseline(PREGUNTA, modelo)
    if baseline:
        cortes = cortes_cuartiles(datos["entrenamiento"], hiperparametros["grupos"])
        parametros["cortes_log_viajes_dia"] = str([round(float(c), 4) for c in cortes])

    with iniciar_run(
        PREGUNTA, modelo, etapa, hiperparametros, meta, nombre, anidado, parametros
    ) as run:
        registrar_conjuntos(
            datos, PREGUNTA, config.DATASET_P3, columnas, None, meta["data_version"]
        )
        if baseline:
            with cronometro() as inferencia:
                metricas, asignaciones, etiquetas = evaluar(
                    prep, columnas, datos, lambda df, z: asignar_por_cortes(df, cortes)
                )
            registrar_costos(0.0, inferencia["segundos"], len(asignaciones))
            registrar_evaluacion(datos, metricas, asignaciones, etiquetas, titulo)
            return run.info.run_id

        z_entrenamiento = prep.transform(datos["entrenamiento"][columnas])
        with cronometro() as ajuste:
            agrupador = ajustar_agrupador(modelo, hiperparametros, z_entrenamiento)
        with cronometro() as inferencia:
            metricas, asignaciones, etiquetas = evaluar(
                prep, columnas, datos, lambda df, z: agrupador.predict(z)
            )
        registrar_costos(ajuste["segundos"], inferencia["segundos"], len(asignaciones))
        if hasattr(agrupador, "inertia_"):
            metricas["entrenamiento"]["inercia"] = float(agrupador.inertia_)
        if hasattr(agrupador, "bic"):
            metricas["entrenamiento"]["bic"] = float(agrupador.bic(z_entrenamiento))
        registrar_evaluacion(datos, metricas, asignaciones, etiquetas, titulo)
        modelo_final = Pipeline([("prep", prep), ("agrupador", agrupador)])
        registrar_modelo(modelo_final, datos["entrenamiento"][columnas].head(100))
    return run.info.run_id
