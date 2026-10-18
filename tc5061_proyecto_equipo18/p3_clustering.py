"""P3 - Experimento base de clustering de estaciones, registrado en MLflow.

Ubicacion en el repo: tc5061_proyecto_equipo18/modeling/p3_clustering.py

Uso (desde la raiz del repo, con el ambiente activado y despues de correr ``dataset ingest``
y ``features``)::

    python -m tc5061_proyecto_equipo18.modeling.p3_clustering

Registra dos runs en el experimento ``config.MLFLOW_EXPERIMENTO``:

1. ``P3-baseline-cuartiles-volumen``: grupos por cuartiles de volumen diario (sin aprendizaje).
2. ``P3-kmeans-pca-k<k>``: StandardScaler -> PCA -> K-means, con k elegido entre
   ``config.P3_K_CANDIDATOS`` por silhouette en entrenamiento.

Anti-fuga: el escalador, la PCA, los cortes de cuartiles y los centroides se ajustan SOLO con
los perfiles de entrenamiento. Validacion y prueba solo se asignan al grupo mas cercano y se
usan para medir la calidad y la estabilidad de los grupos en periodos posteriores.

Las metricas se llaman ``<periodo>_<metrica>`` (por ejemplo ``validacion_silhouette``), igual
que en P2.
"""

import subprocess

import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda archivos
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import typer  # noqa: E402

from tc5061_proyecto_equipo18 import config  # noqa: E402
from tc5061_proyecto_equipo18.features import cargar_metadatos_features  # noqa: E402

app = typer.Typer(add_completion=False)

PREGUNTA = "P3"
PERIODOS = ("entrenamiento", "validacion", "prueba")
VARIABLES_RESUMEN = ["log_viajes_dia", "ratio_fin_semana", "balance_neto"]
PERFILES_GRAFICA = (
    ("sal_lab", "Salidas, dias laborales"),
    ("lle_lab", "Llegadas, dias laborales"),
    ("sal_fds", "Salidas, fines de semana y festivos"),
    ("lle_fds", "Llegadas, fines de semana y festivos"),
)


# =============================================================================
# Datos
# =============================================================================
def cargar_datos() -> pd.DataFrame:
    """Dataset de P3 (una fila por estacion y periodo) que escribe features.py."""
    return pd.read_parquet(config.DATASET_P3)


def separar_periodos(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Perfiles de P3 por periodo (entrenamiento, validacion, prueba)."""
    perfiles = {p: datos[datos["periodo"] == p].reset_index(drop=True) for p in PERIODOS}
    vacios = [p for p, df in perfiles.items() if df.empty]
    if vacios:
        raise RuntimeError(f"No hay perfiles de P3 para: {vacios}. Revisa features.")
    return perfiles


def columnas_variables(df: pd.DataFrame) -> list[str]:
    """Variables de clustering: perfiles horarios (sal_*/lle_*) y el resumen por estacion.

    La ubicacion (latitud/longitud) NO entra: los grupos se forman por uso y luego se mapean.
    """
    perfil = [c for c in df.columns if c.startswith(("sal_", "lle_"))]
    return perfil + VARIABLES_RESUMEN


def ajustar_preprocesamiento(entrenamiento: pd.DataFrame, columnas: list[str]) -> Pipeline:
    """Escalador + PCA ajustados solo con entrenamiento."""
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


def cortes_cuartiles(entrenamiento: pd.DataFrame, grupos: int = config.P3_GRUPOS_BASELINE):
    """Cortes de volumen diario (escala log) calculados solo con entrenamiento."""
    cuantiles = np.linspace(0, 1, grupos + 1)[1:-1]
    return entrenamiento["log_viajes_dia"].quantile(cuantiles).to_numpy()


def asignar_por_cortes(df: pd.DataFrame, cortes: np.ndarray) -> np.ndarray:
    """Grupo de volumen (0 = menos usadas) con los cortes de entrenamiento."""
    return np.digitize(df["log_viajes_dia"].to_numpy(), cortes)


# =============================================================================
# Evaluacion
# =============================================================================
def evaluar(prep, columnas, perfiles, asignar):
    """Asigna grupos en cada periodo y calcula metricas de calidad y de estabilidad.

    - ``<periodo>_silhouette`` y ``<periodo>_davies_bouldin``: calidad de los grupos.
    - ``<periodo>_estabilidad_ari`` (validacion y prueba): Adjusted Rand Index entre el grupo
      de entrenamiento y el grupo asignado en ese periodo, para las estaciones presentes en
      ambos (1 = cada estacion conserva su grupo).
    """
    Z = {p: prep.transform(df[columnas]) for p, df in perfiles.items()}
    etiquetas = {p: np.asarray(asignar(df, Z[p])) for p, df in perfiles.items()}
    ent = perfiles["entrenamiento"]
    referencia = pd.Series(etiquetas["entrenamiento"], index=ent["estacion_id"])

    metricas: dict[str, float] = {}
    for periodo, df in perfiles.items():
        metricas[f"{periodo}_estaciones"] = float(len(df))
        if len(np.unique(etiquetas[periodo])) > 1:
            metricas[f"{periodo}_silhouette"] = float(
                silhouette_score(Z[periodo], etiquetas[periodo])
            )
            metricas[f"{periodo}_davies_bouldin"] = float(
                davies_bouldin_score(Z[periodo], etiquetas[periodo])
            )
        if periodo != "entrenamiento":
            actual = pd.Series(etiquetas[periodo], index=df["estacion_id"])
            comunes = referencia.index.intersection(actual.index)
            if len(comunes) > 1:
                metricas[f"{periodo}_estabilidad_ari"] = float(
                    adjusted_rand_score(referencia.loc[comunes], actual.loc[comunes])
                )

    asignaciones = pd.concat(
        [
            df[["estacion_id", "nombre", "municipio", "latitud", "longitud", "periodo"]].assign(
                grupo=etiquetas[periodo]
            )
            for periodo, df in perfiles.items()
        ],
        ignore_index=True,
    )
    return metricas, asignaciones, etiquetas


# =============================================================================
# Artefactos (figuras en "figuras/", tablas en "tablas/" dentro del run)
# =============================================================================
def guardar_artefactos(nombre: str, ent: pd.DataFrame, grupo: np.ndarray, asignaciones):
    """Devuelve (figuras, tablas): mapa de grupos, perfiles horarios, resumen y asignaciones."""
    carpeta = config.FIGURAS_EXPERIMENTO_DIR
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_mapa = carpeta / f"p3_{nombre}_mapa_grupos.png"
    fig, ax = plt.subplots(figsize=(7, 7))
    puntos = ax.scatter(ent["longitud"], ent["latitud"], c=grupo, cmap="tab10", s=22)
    ax.set(title=f"Estaciones por grupo - {nombre}", xlabel="Longitud", ylabel="Latitud")
    ax.legend(*puntos.legend_elements(), title="Grupo")
    fig.savefig(ruta_mapa, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ruta_perfiles = carpeta / f"p3_{nombre}_perfiles_horarios.png"
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for eje, (prefijo, titulo) in zip(ejes.ravel(), PERFILES_GRAFICA):
        columnas = [f"{prefijo}_h{hora:02d}" for hora in range(24)]
        for g, fila in ent[columnas].groupby(grupo).mean().iterrows():
            eje.plot(range(24), fila.to_numpy(), label=f"Grupo {g}")
        eje.set_title(titulo)
        eje.set_xlabel("Hora del dia")
        eje.set_ylabel("Fraccion del total")
    ejes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(ruta_perfiles, dpi=150)
    plt.close(fig)

    ruta_resumen = carpeta / f"p3_{nombre}_resumen_grupos.csv"
    resumen = (
        ent[VARIABLES_RESUMEN]
        .assign(viajes_dia=np.expm1(ent["log_viajes_dia"]), grupo=grupo)
        .groupby("grupo")
        .agg(
            estaciones=("grupo", "size"),
            viajes_dia_medio=("viajes_dia", "mean"),
            ratio_fin_semana=("ratio_fin_semana", "mean"),
            balance_neto=("balance_neto", "mean"),
        )
        .round(3)
    )
    resumen.to_csv(ruta_resumen)

    ruta_asignaciones = carpeta / f"p3_{nombre}_asignaciones.csv"
    asignaciones.to_csv(ruta_asignaciones, index=False)
    return [ruta_mapa, ruta_perfiles], [ruta_resumen, ruta_asignaciones]


def grafica_barrido_k(barrido: dict, k_elegido: int):
    ruta = config.FIGURAS_EXPERIMENTO_DIR / "p3_barrido_k.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(barrido)
    ax.plot(ks, [barrido[k][0] for k in ks], marker="o")
    ax.axvline(k_elegido, linestyle="--", color="gray")
    ax.set(title="Silhouette en entrenamiento por k", xlabel="k", ylabel="Silhouette")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


# =============================================================================
# MLflow
# =============================================================================
def commit_git() -> tuple[str, bool]:
    """Commit con el que se ejecuta el run y si hay cambios sin commitear."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJ_ROOT, text=True
        ).strip()
        sucio = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=config.PROJ_ROOT, text=True
            ).strip()
        )
        return commit, sucio
    except (OSError, subprocess.CalledProcessError):
        return "desconocido", True


def registrar_run(nombre_run, tipo, parametros, metricas, figuras, tablas, modelo=None) -> str:
    """Crea un run con tags, parametros, metricas y artefactos; devuelve su run_id."""
    commit, sucio = commit_git()
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tags(
            {
                "equipo": config.EQUIPO,
                "pregunta": PREGUNTA,
                "tipo": tipo,
                "git_commit": commit,
                "git_cambios_sin_commit": str(sucio),
            }
        )
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        for figura in figuras:
            mlflow.log_artifact(str(figura), "figuras")
        for tabla in tablas:
            mlflow.log_artifact(str(tabla), "tablas")
        if modelo is not None:
            ruta_modelo = config.ruta_modelo(nombre_run)
            ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, ruta_modelo)
            mlflow.log_artifact(str(ruta_modelo), "modelo")
    logger.info(f"Run {nombre_run}: { {k: round(v, 3) for k, v in metricas.items()} }")
    return run.info.run_id


def entrenar(datos: pd.DataFrame | None = None, meta_features: dict | None = None) -> list[str]:
    """Baseline y candidato de P3. Devuelve los run_id, en ese orden.

    ``datos`` y ``meta_features`` se pueden pasar directamente (pruebas); si no, se leen de
    ``features.py`` (``cargar_metadatos_features`` valida que las features correspondan a la
    version vigente de los datos y de config.py).
    """
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    perfiles = separar_periodos(datos if datos is not None else cargar_datos())
    ent = perfiles["entrenamiento"]
    columnas = columnas_variables(ent)
    prep = ajustar_preprocesamiento(ent, columnas)
    Z_ent = prep.transform(ent[columnas])

    mlflow.set_experiment(config.MLFLOW_EXPERIMENTO)
    comunes = {
        "data_version": meta["data_version"],
        "mes_inicio": config.MES_INICIO,
        "mes_fin": config.MES_FIN,
        "fin_entrenamiento": config.FIN_ENTRENAMIENTO,
        "fin_validacion": config.FIN_VALIDACION,
        "p3_min_viajes": config.P3_MIN_VIAJES,
        "semilla": config.SEMILLA,
        "varianza_pca": config.P3_VARIANZA_PCA,
        "n_componentes_pca": prep.named_steps["pca"].n_components_,
        "n_variables": len(columnas),
    }

    # ---- Baseline: cuartiles de volumen diario (cortes de entrenamiento) ----
    cortes = cortes_cuartiles(ent)
    nombre = "baseline-cuartiles-volumen"
    metricas, asignaciones, etiquetas = evaluar(
        prep, columnas, perfiles, lambda df, Z: asignar_por_cortes(df, cortes)
    )
    figuras, tablas = guardar_artefactos(nombre, ent, etiquetas["entrenamiento"], asignaciones)
    id_baseline = registrar_run(
        f"P3-{nombre}",
        "baseline",
        {**comunes, "modelo": "cuartiles_volumen", "k": config.P3_GRUPOS_BASELINE},
        metricas,
        figuras,
        tablas,
    )

    # ---- Candidato: K-means sobre variables escaladas y PCA; k por silhouette (entren.) ----
    barrido = {}
    for k in config.P3_K_CANDIDATOS:
        km = KMeans(n_clusters=k, random_state=config.SEMILLA, **config.P3_HIPERPARAMETROS)
        km.fit(Z_ent)
        barrido[k] = (float(silhouette_score(Z_ent, km.labels_)), km)
    k_mejor = max(barrido, key=lambda k: barrido[k][0])
    km = barrido[k_mejor][1]

    nombre = f"kmeans-pca-k{k_mejor}"
    metricas, asignaciones, etiquetas = evaluar(
        prep, columnas, perfiles, lambda df, Z: km.predict(Z)
    )
    metricas.update({f"entrenamiento_silhouette_k{k}": s for k, (s, _) in barrido.items()})
    metricas["entrenamiento_inercia"] = float(km.inertia_)
    figuras, tablas = guardar_artefactos(nombre, ent, etiquetas["entrenamiento"], asignaciones)
    figuras.append(grafica_barrido_k(barrido, k_mejor))
    id_candidato = registrar_run(
        f"P3-{nombre}",
        "candidato",
        {
            **comunes,
            "modelo": "kmeans_pca",
            "k": k_mejor,
            "k_candidatos": str(list(config.P3_K_CANDIDATOS)),
            **{f"kmeans_{clave}": valor for clave, valor in config.P3_HIPERPARAMETROS.items()},
        },
        metricas,
        figuras,
        tablas,
        modelo=Pipeline([("prep", prep), ("kmeans", km)]),
    )
    return [id_baseline, id_candidato]


@app.command()
def main():
    """Ejecuta el baseline y el candidato de P3 y los registra en MLflow."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    entrenar()
    logger.success("P3: runs registrados. Revisa la interfaz con: mlflow ui")


if __name__ == "__main__":
    app()
