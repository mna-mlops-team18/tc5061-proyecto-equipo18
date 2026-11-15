"""A/B testing de P1: ``@champion`` (A) contra ``@challenger`` (B) sobre el mismo conjunto.

Punto de entrada ``abtest`` del MLflow Project, desde la raiz del repositorio::

    mlflow run . -e abtest --env-manager local                 # o: make abtest
    mlflow run . -e abtest -P promover=si --env-manager local  # o: make abtest PROMOVER=si

Que hace (todo lo que fija ``config.py``, seccion Monitoreo):

1. Carga las versiones ``@champion`` y ``@challenger`` de P1 del Model Registry y comprueba que
   el periodo de evaluacion (``config.AB_EVALUACION``, sep 2026) es posterior a los datos con que
   se entreno cada una: ninguna de las dos lo vio.
2. Predice con las dos sobre las MISMAS filas de P1 de ese periodo y calcula MAE (la metrica de
   seleccion de P1), RMSE, WAPE y devianza de Poisson, en total y separando estaciones conocidas
   (con filas en el entrenamiento del champion) y nuevas.
3. Bootstrap pareado por dia: remuestrea los dias del periodo con reemplazo
   (``config.AB_REMUESTREOS`` veces) y da el intervalo de confianza de MAE(A) - MAE(B). Se
   remuestrean dias y no filas porque las horas de un mismo dia no son independientes.
4. Aplica la regla de decision, fijada antes de ver los resultados: B gana solo si el intervalo
   queda completo por encima de 0 y, en las estaciones conocidas, su MAE no es mas de
   ``config.AB_TOLERANCIA_CONOCIDAS`` peor que el de A.
5. Mide el MAE del champion mes a mes en ``config.MONITOREO_MENSUAL`` (abr-sep 2026): si crece
   con el tiempo, el modelo se degrada y es evidencia para el Continuous Training.
6. Escribe ``reports/monitoreo/ab_test.md`` (tablas para el reporte), ``ab_test.json`` y dos
   figuras, y deja todo en un run de MLflow (etiqueta ``etapa = ab_test``).

Con ``--promover si`` mueve los aliases (B pasa a ``@champion`` y A a ``@challenger``) solo si
la regla dice que B gana. Sin la opcion, el registro no cambia.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import typer

from monitoreo import conjuntos
from tc5061_proyecto_equipo18 import config, plots
from tc5061_proyecto_equipo18.dataset import Manifiesto
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1
from tc5061_proyecto_equipo18.modeling import registro
from tc5061_proyecto_equipo18.modeling.comun import configurar_mlflow, estado_git
from tc5061_proyecto_equipo18.modeling.p1_regresion import metricas_regresion

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

PREGUNTA = "P1"
ETAPA = "ab_test"
COLUMNA_TIEMPO = "hora"
COLUMNA_ESTACION = "estacion_id"
GRUPOS = ("todas", "conocidas", "nuevas")
METRICAS_TABLA = ("mae", "rmse", "wape", "devianza_poisson")


@dataclass
class Variante:
    """Una de las dos versiones que se comparan: alias, version del registro y modelo cargado."""

    nombre: str  # "A" o "B"
    alias: str
    version: ModelVersion
    modelo: object  # modelo pyfunc de MLflow
    fin_entrenamiento: str  # ultima fecha (inclusive) de sus datos de entrenamiento

    def describir(self) -> str:
        return f"{self.nombre} = v{self.version.version} (@{self.alias})"


# =============================================================================
# Versiones del registro
# =============================================================================
def fin_entrenamiento_de(version: ModelVersion) -> str:
    """Ultima fecha con que se entreno la version.

    El reentrenado (``modeling/reentrenar.py``) la deja en la etiqueta ``reentrenamiento_fin``.
    Las versiones que registra el ``register`` salen de candidatos o busquedas, que entrenan con
    la particion de entrenamiento: hasta ``config.FIN_ENTRENAMIENTO``.
    """
    return version.tags.get("reentrenamiento_fin") or config.FIN_ENTRENAMIENTO


def cargar_variante(
    nombre: str, alias: str, modelo_registrado: str = config.MODELOS_REGISTRADOS[PREGUNTA]
) -> Variante:
    """Version del alias y su modelo pyfunc; falla con el comando que crea el alias si no existe."""
    cliente = mlflow.MlflowClient()
    try:
        version = cliente.get_model_version_by_alias(modelo_registrado, alias)
    except MlflowException as error:
        comando = (
            "mlflow run . -e retrain --env-manager local"
            if alias == config.ALIAS_RETADOR
            else "mlflow run . -e register -P pregunta=p1 --env-manager local"
        )
        raise RuntimeError(
            f"{modelo_registrado} no tiene @{alias}. Crealo con: {comando}"
        ) from error
    modelo = mlflow.pyfunc.load_model(f"models:/{modelo_registrado}@{alias}")
    return Variante(nombre, alias, version, modelo, fin_entrenamiento_de(version))


def verificar_sin_fuga(
    variantes: list[Variante], evaluacion: tuple[str, str] = config.AB_EVALUACION
) -> None:
    """Ninguna version puede haber entrenado con datos del periodo de evaluacion."""
    inicio = pd.Timestamp(evaluacion[0])
    for variante in variantes:
        fin = pd.Timestamp(variante.fin_entrenamiento) + pd.Timedelta(hours=23)
        if fin >= inicio:
            raise ValueError(
                f"{variante.describir()} entreno con datos hasta {variante.fin_entrenamiento}, "
                f"dentro o despues de AB_EVALUACION ({evaluacion[0]}): la comparacion tendria "
                "fuga a su favor."
            )
    if variantes[0].version.version == variantes[1].version.version:
        raise ValueError("@champion y @challenger son la misma version: no hay nada que comparar")


# =============================================================================
# Datos y predicciones
# =============================================================================
def cargar_p1(ruta: Path = config.DATASET_P1) -> pd.DataFrame:
    """Solo las columnas que hacen falta del Parquet de P1 (lo escribe features.py)."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corre primero: make data && make features")
    return pd.read_parquet(
        ruta, columns=[COLUMNA_ESTACION, COLUMNA_TIEMPO, *COLUMNAS_P1, OBJETIVO_P1]
    )


def filas_de(p1: pd.DataFrame, periodo: tuple[str, str]) -> pd.DataFrame:
    """Filas de P1 dentro del periodo (ambos dias inclusivos); falla si no hay ninguna."""
    filas = p1[conjuntos.en_periodo(p1[COLUMNA_TIEMPO], periodo)].reset_index(drop=True)
    if filas.empty:
        raise ValueError(f"P1 no tiene filas entre {periodo[0]} y {periodo[1]}")
    return filas


def predecir(variante: Variante, datos: pd.DataFrame) -> np.ndarray:
    """Predicciones del modelo registrado, sin recortar (igual que las devuelve la API)."""
    return np.asarray(variante.modelo.predict(datos[COLUMNAS_P1]), dtype=float).ravel()


# =============================================================================
# Metricas, bootstrap y decision
# =============================================================================
def tabla_metricas(
    datos: pd.DataFrame, predicciones: dict[str, np.ndarray], conocidas: set[int]
) -> pd.DataFrame:
    """Una fila por grupo de estaciones y version, con filas, estaciones y metricas."""
    es_conocida = datos[COLUMNA_ESTACION].isin(conocidas).to_numpy()
    mascaras = {
        "todas": np.ones(len(datos), bool),
        "conocidas": es_conocida,
        "nuevas": ~es_conocida,
    }
    filas = []
    for grupo, mascara in mascaras.items():
        if not mascara.any():
            continue
        real = datos.loc[mascara, OBJETIVO_P1].to_numpy()
        for nombre, predicho in predicciones.items():
            metricas = metricas_regresion(real, predicho[mascara])
            filas.append(
                {
                    "grupo": grupo,
                    "version": nombre,
                    "filas": int(mascara.sum()),
                    "estaciones": int(datos.loc[mascara, COLUMNA_ESTACION].nunique()),
                    **{clave: float(metricas[clave]) for clave in METRICAS_TABLA},
                }
            )
    return pd.DataFrame(filas)


def errores_por_dia(
    datos: pd.DataFrame, prediccion_a: np.ndarray, prediccion_b: np.ndarray
) -> pd.DataFrame:
    """Por dia: suma de errores absolutos de A y de B y numero de filas (unidad del bootstrap)."""
    real = datos[OBJETIVO_P1].to_numpy(dtype=float)
    return (
        pd.DataFrame(
            {
                "dia": datos[COLUMNA_TIEMPO].dt.floor("D").to_numpy(),
                "error_a": np.abs(real - prediccion_a),
                "error_b": np.abs(real - prediccion_b),
                "filas": 1,
            }
        )
        .groupby("dia", as_index=False)
        .sum()
    )


def bootstrap_pareado(
    por_dia: pd.DataFrame,
    remuestreos: int = config.AB_REMUESTREOS,
    confianza: float = config.AB_CONFIANZA,
    semilla: int = config.SEMILLA,
) -> dict[str, float]:
    """Intervalo de confianza (percentiles) de MAE(A) - MAE(B) remuestreando dias completos.

    Es pareado: cada remuestreo toma los mismos dias para A y para B, asi que la variacion entre
    dias (un dia lluvioso es dificil para los dos) se cancela en la diferencia. El MAE de cada
    remuestreo es la suma de errores entre la suma de filas, igual que el MAE de todo el periodo.
    """
    if len(por_dia) < 2:
        raise ValueError("El bootstrap por dia necesita al menos 2 dias de evaluacion")
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, len(por_dia), size=(remuestreos, len(por_dia)))
    error_a = por_dia["error_a"].to_numpy()[indices].sum(axis=1)
    error_b = por_dia["error_b"].to_numpy()[indices].sum(axis=1)
    filas = por_dia["filas"].to_numpy()[indices].sum(axis=1)
    diferencias = (error_a - error_b) / filas
    alfa = 1 - confianza
    total = por_dia[["error_a", "error_b", "filas"]].sum()
    return {
        "diferencia_mae": float((total["error_a"] - total["error_b"]) / total["filas"]),
        "ic_inferior": float(np.quantile(diferencias, alfa / 2)),
        "ic_superior": float(np.quantile(diferencias, 1 - alfa / 2)),
        "fraccion_b_mejor": float(np.mean(diferencias > 0)),
        "dias": len(por_dia),
        "remuestreos": remuestreos,
        "confianza": confianza,
    }


def decidir(
    bootstrap: dict[str, float],
    metricas: pd.DataFrame,
    tolerancia: float = config.AB_TOLERANCIA_CONOCIDAS,
) -> dict:
    """Regla de decision de ``config.py``: B gana solo si cumple las dos condiciones."""
    b_mejor = bootstrap["ic_inferior"] > 0
    conocidas = metricas[metricas["grupo"] == "conocidas"].set_index("version")["mae"]
    if conocidas.empty:  # sin estaciones conocidas no hay regresion que medir
        cambio_conocidas, sin_regresion = 0.0, True
    else:
        cambio_conocidas = float(conocidas["B"] / conocidas["A"] - 1)
        sin_regresion = cambio_conocidas <= tolerancia
    promover = bool(b_mejor and sin_regresion)
    if promover:
        motivo = "B gana: el intervalo de MAE(A) - MAE(B) queda arriba de 0 y no empeora en las conocidas."
    elif not b_mejor:
        motivo = (
            "A se queda: el intervalo de MAE(A) - MAE(B) incluye el 0 o queda debajo; la "
            "mejora de B no es distinguible del azar entre dias."
        )
    else:
        motivo = (
            f"A se queda: B empeora {cambio_conocidas:.1%} en las estaciones conocidas, mas que "
            f"la tolerancia de {tolerancia:.0%}."
        )
    return {
        "b_mejor_en_mae": bool(b_mejor),
        "cambio_mae_conocidas": cambio_conocidas,
        "sin_regresion_en_conocidas": bool(sin_regresion),
        "promover_b": promover,
        "motivo": motivo,
    }


def mae_mensual(
    p1: pd.DataFrame, variante: Variante, periodo: tuple[str, str] = config.MONITOREO_MENSUAL
) -> pd.DataFrame:
    """MAE de una version mes a mes; marca si el mes es de validacion o del periodo de prueba."""
    datos = filas_de(p1, periodo)
    error = np.abs(datos[OBJETIVO_P1].to_numpy(dtype=float) - predecir(variante, datos))
    tabla = (
        pd.DataFrame({"mes": datos[COLUMNA_TIEMPO].dt.strftime("%Y-%m"), "error": error})
        .groupby("mes")["error"]
        .agg(mae="mean", filas="size")
        .reset_index()
    )
    fin_validacion = pd.Timestamp(config.FIN_VALIDACION).strftime("%Y-%m")
    tabla["particion"] = np.where(tabla["mes"] <= fin_validacion, "validacion", "prueba")
    return tabla


# =============================================================================
# Figuras y resumen
# =============================================================================
def grafica_mae_por_grupo(metricas: pd.DataFrame) -> Figure:
    """MAE de A y B en todas las estaciones, las conocidas y las nuevas."""
    grupos = [g for g in GRUPOS if g in set(metricas["grupo"])]
    tabla = metricas.pivot(index="grupo", columns="version", values="mae").loc[grupos]
    x = np.arange(len(grupos))
    figura, eje = plt.subplots(figsize=(6.5, 3.6))
    for desplazamiento, version, color in (
        (-0.2, "A", plots.SERIES[0]),
        (0.2, "B", plots.SERIES[1]),
    ):
        barras = eje.bar(x + desplazamiento, tabla[version], width=0.4, color=color, label=version)
        eje.bar_label(barras, fmt="%.3f", fontsize=8, color=plots.TINTA_SECUNDARIA, padding=2)
    eje.set_xticks(x, grupos)
    eje.set_ylabel("MAE (salidas por estacion-hora)")
    eje.grid(axis="x", visible=False)
    eje.legend(["A (@champion)", "B (@challenger)"])
    eje.set_title("MAE en el periodo de evaluacion del A/B")
    figura.tight_layout()
    return figura


def grafica_mae_mensual(mensual: pd.DataFrame) -> Figure:
    """MAE mensual del champion; en gris claro los meses de validacion (con que se eligio)."""
    figura, eje = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(mensual))
    validacion = mensual["particion"].eq("validacion").to_numpy()
    if validacion.any():
        eje.axvspan(-0.5, validacion.sum() - 0.5, color=plots.GRIS_CONTEXTO, alpha=0.5, lw=0)
        eje.text(
            -0.4,
            mensual["mae"].max(),
            "validacion",
            color=plots.TINTA_SECUNDARIA,
            fontsize=8,
            va="top",
        )
    eje.plot(x, mensual["mae"], marker="o", color=plots.SERIES[0])
    eje.set_xticks(x, mensual["mes"])
    eje.set_ylabel("MAE")
    eje.set_title("MAE mensual del champion")
    figura.tight_layout()
    return figura


def _tabla_md(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def escribir_resumen(resultado: dict, ruta: Path = config.RESUMEN_AB_MD) -> Path:
    """Markdown con la configuracion, la tabla comparativa, el bootstrap y la decision."""
    a, b = resultado["versiones"]["A"], resultado["versiones"]["B"]
    boot, decision = resultado["bootstrap"], resultado["decision"]
    metricas = pd.DataFrame(resultado["metricas"])
    filas_metricas = []
    for grupo in GRUPOS:
        parte = metricas[metricas["grupo"] == grupo].set_index("version")
        if parte.empty:
            continue
        for metrica in METRICAS_TABLA:
            va, vb = parte.loc["A", metrica], parte.loc["B", metrica]
            filas_metricas.append(
                [
                    grupo,
                    metrica,
                    f"{va:.4f}",
                    f"{vb:.4f}",
                    f"{vb - va:+.4f}",
                    f"{vb / va - 1:+.1%}",
                ]
            )
    filas_tamano = [
        [g, f"{int(p['filas'].iloc[0]):,}", str(int(p["estaciones"].iloc[0]))]
        for g, p in metricas.groupby("grupo", sort=False)
    ]
    mensual = resultado["mae_mensual_champion"]
    texto = f"""# A/B testing de P1: champion contra challenger

Generado con `mlflow run . -e abtest` (`modeling/ab_test.py`). No editar a mano.

- **Modelo registrado:** `{resultado["modelo_registrado"]}`
- **A (`@{a["alias"]}`):** version {a["version"]}, run `{a["run_id"]}`, entrenada con datos hasta {a["fin_entrenamiento"]}
- **B (`@{b["alias"]}`):** version {b["version"]}, run `{b["run_id"]}`, entrenada con datos hasta {b["fin_entrenamiento"]}{f" (reentrenamiento de la version {b['version_origen']})" if b.get("version_origen") else ""}
- **Conjunto de evaluacion (el mismo para las dos):** P1 de {resultado["evaluacion"][0]} a {resultado["evaluacion"][1]}, sin muestreo
- **Version de datos (`data_version`):** `{resultado["data_version"]}`
- **Estaciones conocidas:** las que tienen filas en el entrenamiento del champion ({resultado["referencia"][0]} a {resultado["referencia"][1]}); las demas son nuevas.

## Conjunto de evaluacion

{_tabla_md(["Grupo", "Filas", "Estaciones"], filas_tamano)}

## Metricas

Cambio = B - A; un cambio negativo en estas metricas favorece a B.

{_tabla_md(["Grupo", "Metrica", "A", "B", "Cambio", "Cambio relativo"], filas_metricas)}

## Bootstrap pareado por dia

{boot["remuestreos"]:,} remuestreos de los {boot["dias"]} dias del periodo (semilla {config.SEMILLA}).

| Diferencia MAE(A) - MAE(B) | IC {boot["confianza"]:.0%} | Remuestreos en que B es mejor |
|---:|---|---:|
| {boot["diferencia_mae"]:+.4f} | [{boot["ic_inferior"]:+.4f}, {boot["ic_superior"]:+.4f}] | {boot["fraccion_b_mejor"]:.1%} |

## Decision

Regla fijada en `config.py` antes de ver los resultados: B gana solo si (1) el IC de MAE(A) -
MAE(B) queda completo por encima de 0 y (2) en las estaciones conocidas el MAE de B no es mas de
{config.AB_TOLERANCIA_CONOCIDAS:.0%} peor que el de A.

| Condicion | Cumple |
|---|---|
| (1) B mejor en MAE (IC arriba de 0) | {"si" if decision["b_mejor_en_mae"] else "no"} |
| (2) Sin regresion en conocidas (cambio {decision["cambio_mae_conocidas"]:+.1%}) | {"si" if decision["sin_regresion_en_conocidas"] else "no"} |

**{decision["motivo"]}** Aliases: {resultado["aliases"]}.

## MAE mensual del champion

Abr-jun es la validacion con que se eligio el champion; jul-sep, meses que no vio.

{_tabla_md(["Mes", "Particion", "Filas", "MAE"], [[m["mes"], m["particion"], f"{m['filas']:,}", f"{m['mae']:.4f}"] for m in mensual])}
"""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


# =============================================================================
# Aliases y registro en MLflow
# =============================================================================
def promover_b(a: Variante, b: Variante) -> None:
    """B pasa a ``@champion`` y A queda como ``@challenger`` (no se borra ninguna version)."""
    nombre = a.version.name
    registro.asignar_alias(nombre, config.ALIAS_PRODUCCION, b.version)
    registro.asignar_alias(nombre, config.ALIAS_RETADOR, a.version)
    registro.verificar_carga(nombre, config.ALIAS_PRODUCCION)


def registrar_en_mlflow(resultado: dict, archivos: list[Path]) -> str:
    """Run con las metricas del A/B, la decision y los archivos del resumen."""
    a, b = resultado["versiones"]["A"], resultado["versiones"]["B"]
    metricas = pd.DataFrame(resultado["metricas"])
    etiquetas = {
        "equipo": config.EQUIPO,
        "pregunta": PREGUNTA,
        "etapa": ETAPA,
        "data_version": resultado["data_version"],
        "version_a": a["version"],
        "version_b": b["version"],
        "decision": "promover_b" if resultado["decision"]["promover_b"] else "mantener_a",
        **estado_git(),
    }
    with mlflow.start_run(
        run_name=f"p1-abtest-v{a['version']}-vs-v{b['version']}", tags=etiquetas
    ) as run:
        mlflow.log_params(
            {
                "evaluacion_inicio": resultado["evaluacion"][0],
                "evaluacion_fin": resultado["evaluacion"][1],
                "metrica": config.AB_METRICA,
                "remuestreos": config.AB_REMUESTREOS,
                "confianza": config.AB_CONFIANZA,
                "tolerancia_conocidas": config.AB_TOLERANCIA_CONOCIDAS,
                "semilla": config.SEMILLA,
            }
        )
        valores = {
            f"{fila.grupo}_{fila.version.lower()}_{metrica}": getattr(fila, metrica)
            for fila in metricas.itertuples()
            for metrica in METRICAS_TABLA
        }
        boot = resultado["bootstrap"]
        valores |= {
            "diferencia_mae": boot["diferencia_mae"],
            "ic_inferior": boot["ic_inferior"],
            "ic_superior": boot["ic_superior"],
            "promover_b": float(resultado["decision"]["promover_b"]),
        }
        valores |= {
            f"champion_mae_{m['mes'].replace('-', '_')}": m["mae"]
            for m in resultado["mae_mensual_champion"]
        }
        mlflow.log_metrics(valores)
        for ruta in archivos:
            mlflow.log_artifact(str(ruta), "ab_test")
        return run.info.run_id


def _datos_version(variante: Variante) -> dict:
    return {
        "alias": variante.alias,
        "version": variante.version.version,
        "run_id": variante.version.run_id,
        "fin_entrenamiento": variante.fin_entrenamiento,
        "version_origen": variante.version.tags.get("version_origen"),
        "data_version": variante.version.tags.get("data_version"),
    }


# =============================================================================
# Flujo completo
# =============================================================================
def ejecutar(
    p1: pd.DataFrame | None = None,
    promover: bool = False,
    carpeta: Path = config.MONITOREO_DIR,
    figuras: Path = config.MONITOREO_FIGURAS_DIR,
    data_version: str | None = None,
) -> dict:
    """Corre el A/B, escribe sus salidas, lo registra en MLflow y (opcional) mueve los aliases."""
    a = cargar_variante("A", config.ALIAS_PRODUCCION)
    b = cargar_variante("B", config.ALIAS_RETADOR)
    verificar_sin_fuga([a, b])
    logger.info(f"{a.describir()} contra {b.describir()} en {config.AB_EVALUACION}")

    p1 = p1 if p1 is not None else cargar_p1()
    if data_version is None:
        data_version = Manifiesto.cargar().version
    for variante in (a, b):
        if variante.version.tags.get("data_version", data_version) != data_version:
            logger.warning(
                f"{variante.describir()} se entreno con data_version "
                f"{variante.version.tags['data_version']} y los datos actuales son {data_version}"
            )
    conocidas = conjuntos.estaciones_conocidas(p1)
    datos = filas_de(p1, config.AB_EVALUACION)
    predicciones = {"A": predecir(a, datos), "B": predecir(b, datos)}
    metricas = tabla_metricas(datos, predicciones, conocidas)
    boot = bootstrap_pareado(errores_por_dia(datos, predicciones["A"], predicciones["B"]))
    decision = decidir(boot, metricas)
    mensual = mae_mensual(p1, a)

    aliases = "sin cambios"
    if promover and decision["promover_b"]:
        promover_b(a, b)
        aliases = f"v{b.version.version} pasa a @champion y v{a.version.version} a @challenger"
    elif promover:
        aliases = "sin cambios (se pidio promover, pero la regla no lo indica)"
    elif decision["promover_b"]:
        aliases = "sin cambios; para promover a B: make abtest PROMOVER=si"

    resultado = {
        "modelo_registrado": a.version.name,
        "data_version": data_version,
        "evaluacion": list(config.AB_EVALUACION),
        "referencia": list(config.MONITOREO_REFERENCIA),
        "versiones": {"A": _datos_version(a), "B": _datos_version(b)},
        "metricas": metricas.to_dict(orient="records"),
        "bootstrap": boot,
        "decision": decision,
        "aliases": aliases,
        "mae_mensual_champion": mensual.to_dict(orient="records"),
    }
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_md = escribir_resumen(resultado, carpeta / config.RESUMEN_AB_MD.name)
    ruta_json = carpeta / config.RESULTADOS_AB_JSON.name
    ruta_json.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    ruta_grupos = plots.guardar_figura(
        grafica_mae_por_grupo(metricas), figuras / "ab_mae_por_grupo.png"
    )
    ruta_mensual = plots.guardar_figura(
        grafica_mae_mensual(mensual), figuras / "ab_mae_mensual.png"
    )
    resultado["run_id"] = registrar_en_mlflow(
        resultado, [ruta_md, ruta_json, ruta_grupos, ruta_mensual]
    )
    mae_total = metricas[metricas["grupo"] == "todas"].set_index("version")["mae"]
    logger.success(
        f"MAE A {mae_total['A']:.4f}, B {mae_total['B']:.4f}; "
        f"IC de A-B [{boot['ic_inferior']:+.4f}, {boot['ic_superior']:+.4f}]. {decision['motivo']} "
        f"Aliases: {aliases}. Resumen en {ruta_md}"
    )
    return resultado


@app.command()
def main(
    promover: str = typer.Option(
        "no", help="si: mueve los aliases si la regla de decision indica promover a B"
    ),
):
    """A/B testing de P1: @champion contra @challenger sobre config.AB_EVALUACION."""
    if promover not in ("si", "no"):
        raise typer.BadParameter("promover debe ser si o no")
    configurar_mlflow()
    ejecutar(promover=promover == "si")


if __name__ == "__main__":
    app()
