"""P3 - Clustering: que grupos de estaciones comparten patrones de uso y donde se ubican.

Dos runs en MLflow, evaluados con las mismas metricas y periodos:

- ``baseline`` (``cuartiles_volumen``): grupos por cuartiles del volumen diario, sin
  aprendizaje. Es la segmentacion que haria cualquiera sin modelo ("estaciones muy, medio y
  poco usadas"); el candidato debe aportar algo que el volumen solo no explica.
- ``candidato`` (``kmeans_pca``): StandardScaler -> PCA -> K-means, con ``k`` elegido entre
  ``config.P3_K_CANDIDATOS`` por silhouette en entrenamiento.

Anti-fuga: el escalador, la PCA, los cortes de cuartiles y los centroides se ajustan SOLO con
los perfiles de entrenamiento. Validacion y prueba solo se asignan al grupo (o corte) mas
cercano y sirven para medir la calidad y la estabilidad de los grupos en periodos posteriores.
La ubicacion (latitud, longitud) no entra al clustering: los grupos se forman por uso y despues
se ubican en el mapa, que es lo que pregunta P3.

Las metricas se llaman ``<periodo>_<metrica>`` (por ejemplo ``validacion_silhouette``), igual
que en P1 y P2.

Uso::

    python -m tc5061_proyecto_equipo18.modeling.train --pregunta P3
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
from tc5061_proyecto_equipo18.features import PARTICIONES
from tc5061_proyecto_equipo18.modeling.comun import (
    ROLES,
    cargar_dataset,
    iniciar_run,
    registrar_figura,
    registrar_metricas,
    registrar_modelo,
)
from tc5061_proyecto_equipo18.plots import (
    grafica_mapa_grupos,
    grafica_metrica_por_k,
    grafica_perfiles_grupo,
    mapa_html_grupos,
)

PREGUNTA = "P3"
VARIABLES_RESUMEN = ["log_viajes_dia", "ratio_fin_semana", "balance_neto"]
COLUMNAS_MAPA = ["estacion_id", "codigo", "nombre", "municipio", "latitud", "longitud"]

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


def cortes_cuartiles(
    entrenamiento: pd.DataFrame, grupos: int = config.P3_GRUPOS_BASELINE
) -> np.ndarray:
    """Cortes de volumen diario (escala log) calculados solo con entrenamiento."""
    cuantiles = np.linspace(0, 1, grupos + 1)[1:-1]
    return entrenamiento["log_viajes_dia"].quantile(cuantiles).to_numpy()


def asignar_por_cortes(df: pd.DataFrame, cortes: np.ndarray) -> np.ndarray:
    """Grupo de volumen (0 = menos usadas) con los cortes de entrenamiento."""
    return np.digitize(df["log_viajes_dia"].to_numpy(), cortes)


def elegir_k(z_entrenamiento: np.ndarray) -> tuple[int, KMeans, dict[int, float]]:
    """Ajusta K-means para cada k de ``config.P3_K_CANDIDATOS`` y elige el de mayor silhouette.

    Solo usa entrenamiento. Devuelve el k elegido, su modelo y la silhouette de cada k.
    """
    modelos, siluetas = {}, {}
    for k in config.P3_K_CANDIDATOS:
        if k >= len(z_entrenamiento):
            continue
        modelo = KMeans(n_clusters=k, random_state=config.SEMILLA, **config.P3_HIPERPARAMETROS)
        modelo.fit(z_entrenamiento)
        modelos[k] = modelo
        siluetas[k] = float(silhouette_score(z_entrenamiento, modelo.labels_))
    k_mejor = max(siluetas, key=siluetas.get)
    return k_mejor, modelos[k_mejor], siluetas


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
      entrenamiento, para que baseline y candidato se comparen igual.
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
    mlflow.log_text(
        resumen_grupos(perfiles["entrenamiento"], etiquetas["entrenamiento"]).to_csv(index=False),
        "tablas/resumen_grupos.csv",
    )
    mlflow.log_text(asignaciones.to_csv(index=False), "tablas/asignaciones.csv")


def preparar(datos: dict[str, pd.DataFrame]) -> tuple[list[str], Pipeline, dict]:
    """Variables, preprocesamiento de entrenamiento y parametros comunes a los dos runs."""
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


def correr_baseline(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run de la linea base. No registra modelo: es una regla con cortes de entrenamiento."""
    columnas, prep, parametros = preparar(datos)
    cortes = cortes_cuartiles(datos["entrenamiento"])
    parametros |= {
        "grupos": config.P3_GRUPOS_BASELINE,
        "cortes_log_viajes_dia": str([round(float(c), 4) for c in cortes]),
    }
    with iniciar_run(PREGUNTA, "baseline", "cuartiles_volumen", parametros, meta_features) as run:
        metricas, asignaciones, etiquetas = evaluar(
            prep, columnas, datos, lambda df, z: asignar_por_cortes(df, cortes)
        )
        registrar_evaluacion(datos, metricas, asignaciones, etiquetas, "Cuartiles de volumen")
    return run.info.run_id


def correr_candidato(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run del candidato: elige k, evalua, registra artefactos y el modelo (prep + K-means)."""
    columnas, prep, parametros = preparar(datos)
    k, kmeans, siluetas = elegir_k(prep.transform(datos["entrenamiento"][columnas]))
    parametros |= {
        "k": k,
        "k_candidatos": str(list(config.P3_K_CANDIDATOS)),
        **{f"kmeans_{clave}": valor for clave, valor in config.P3_HIPERPARAMETROS.items()},
    }
    with iniciar_run(PREGUNTA, "candidato", "kmeans_pca", parametros, meta_features) as run:
        metricas, asignaciones, etiquetas = evaluar(
            prep, columnas, datos, lambda df, z: kmeans.predict(z)
        )
        metricas["entrenamiento"] |= {f"silhouette_k{n}": s for n, s in siluetas.items()}
        metricas["entrenamiento"]["inercia"] = float(kmeans.inertia_)
        registrar_evaluacion(datos, metricas, asignaciones, etiquetas, f"K-means (k = {k})")
        registrar_figura(
            grafica_metrica_por_k(siluetas, k, "Silhouette", "Silhouette en entrenamiento por k"),
            "silhouette_por_k",
        )
        modelo = Pipeline([("prep", prep), ("kmeans", kmeans)])
        registrar_modelo(modelo, datos["entrenamiento"][columnas].head(100))
    return run.info.run_id


def entrenar(
    roles: tuple[str, ...] = ROLES,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
) -> list[str]:
    """Punto de entrada que invoca train.py: corre los roles pedidos y devuelve sus run_id.

    ``datos`` y ``meta_features`` solo se pasan en las pruebas; por defecto se leen del
    Parquet de P3 (separado por la columna ``periodo``) y de los metadatos de features (que
    validan la version de los datos).
    """
    if datos is None:
        datos = cargar_dataset(config.DATASET_P3, columna_tiempo=None)
    corredores = {"baseline": correr_baseline, "candidato": correr_candidato}
    return [corredores[rol](datos, meta_features) for rol in roles]
