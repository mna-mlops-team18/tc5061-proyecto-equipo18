"""P2 - Clasificacion: una estacion se quedara sin bicicletas o sin anclajes en el siguiente bloque.

Clases (``config.P2_CLASES``): 0 sin riesgo, 1 riesgo de vaciado, 2 riesgo de llenado. Dos runs
en MLflow, evaluados con las mismas metricas y particiones:

- ``baseline`` (``frecuencia_mismo_bloque``): la probabilidad de cada clase es la fraccion de
  los ultimos 7 dias en que ese mismo bloque estuvo en esa clase; predice la mas probable.
  Es la regla que el equipo de rebalanceo aplicaria con su experiencia ("esta estacion
  suele vaciarse a esta hora").
- ``candidato`` (``hgb_balanceado``): ``HistGradientBoostingClassifier`` con pesos de clase
  balanceados (las clases de riesgo son minoritarias) sobre ``features.COLUMNAS_P2``.

Ambos se evaluan en validacion (para comparar y elegir) y en prueba (estimacion final).

Uso::

    python -m tc5061_proyecto_equipo18.modeling.train --pregunta P2
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P2, OBJETIVO_P2
from tc5061_proyecto_equipo18.modeling.comun import (
    ROLES,
    cargar_dataset,
    iniciar_run,
    muestrear,
    registrar_figura,
    registrar_metricas,
    registrar_modelo,
)
from tc5061_proyecto_equipo18.plots import (
    grafica_curvas_pr,
    grafica_importancias,
    grafica_matriz_confusion,
    grafica_riesgo_por_bloque,
)

PREGUNTA = "P2"
PARTICIONES_EVALUACION = ("validacion", "prueba")
CLASES = list(config.P2_CLASES)
CLASES_ARRAY = np.array(CLASES)  # para pasar de indice de columna a clase
CLASES_RIESGO = {1: "vaciado", 2: "llenado"}


def metricas_clasificacion(real: np.ndarray, probabilidades: np.ndarray) -> dict[str, float]:
    """Metricas de P2 para comparar baseline y candidato.

    - ``f1_macro``: promedia por igual las tres clases; la exactitud seria enganosa porque
      casi todos los bloques son "sin riesgo".
    - ``balanced_accuracy``: recall promedio por clase.
    - ``recall_<riesgo>``: fraccion de bloques en riesgo que se detectan (un riesgo no
      detectado deja usuarios sin bicicleta o sin anclaje).
    - ``precision_<riesgo>``: fraccion de alertas correctas (una alerta falsa es un viaje de
      rebalanceo inutil).
    - ``pr_auc_<riesgo>``: precision promedio sobre todos los umbrales; no depende de la
      regla de decision y se compara contra la prevalencia de la clase.
    """
    real = np.asarray(real)
    predicho = CLASES_ARRAY[np.argmax(probabilidades, axis=1)]
    metricas = {
        "f1_macro": f1_score(real, predicho, labels=CLASES, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(real, predicho),
    }
    recall = recall_score(real, predicho, labels=CLASES, average=None, zero_division=0)
    precision = precision_score(real, predicho, labels=CLASES, average=None, zero_division=0)
    for clase, nombre in CLASES_RIESGO.items():
        positivos = real == clase
        metricas[f"recall_{nombre}"] = recall[clase]
        metricas[f"precision_{nombre}"] = precision[clase]
        metricas[f"pr_auc_{nombre}"] = (
            average_precision_score(positivos, probabilidades[:, clase])
            if positivos.any()
            else np.nan
        )
    return metricas


def probabilidades_baseline(datos: pd.DataFrame) -> np.ndarray:
    """Probabilidad de cada clase segun la frecuencia del mismo bloque en los ultimos 7 dias.

    Usa ``frac_vaciado_mismo_bloque_7d`` y ``frac_llenado_mismo_bloque_7d``, calculadas solo
    con dias anteriores, asi que el baseline no usa informacion del bloque que predice.
    """
    vaciado = datos["frac_vaciado_mismo_bloque_7d"].to_numpy(dtype=float)
    llenado = datos["frac_llenado_mismo_bloque_7d"].to_numpy(dtype=float)
    sin_riesgo = np.clip(1 - vaciado - llenado, 0, 1)
    return np.column_stack([sin_riesgo, vaciado, llenado])


def probabilidades_modelo(modelo: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """``predict_proba`` con una columna por clase de ``CLASES``, aunque falte alguna en el ajuste."""
    probabilidades = np.zeros((len(X), len(CLASES)))
    probabilidades[:, modelo.classes_] = modelo.predict_proba(X)
    return probabilidades


def entrenar_candidato(entrenamiento: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Ajusta el gradient boosting con pesos balanceados sobre todo el entrenamiento."""
    modelo = HistGradientBoostingClassifier(
        **config.P2_HIPERPARAMETROS, random_state=config.SEMILLA
    )
    return modelo.fit(entrenamiento[COLUMNAS_P2], entrenamiento[OBJETIVO_P2])


def registrar_evaluacion(datos: dict[str, pd.DataFrame], probabilidades: dict[str, np.ndarray]):
    """Registra las metricas por particion y las figuras de diagnostico de validacion."""
    for particion in PARTICIONES_EVALUACION:
        metricas = metricas_clasificacion(datos[particion][OBJETIVO_P2], probabilidades[particion])
        registrar_metricas(metricas, particion)

    real = datos["validacion"][OBJETIVO_P2].to_numpy()
    predicho = CLASES_ARRAY[np.argmax(probabilidades["validacion"], axis=1)]
    registrar_figura(
        grafica_matriz_confusion(real, predicho, config.P2_CLASES, "Matriz de confusion"),
        "matriz_confusion",
    )
    registrar_figura(
        grafica_curvas_pr(
            real, probabilidades["validacion"], config.P2_CLASES, "Curvas precision-recall"
        ),
        "curvas_pr",
    )
    validacion = datos["validacion"][["bloque_dia", OBJETIVO_P2]].assign(predicho=predicho)
    registrar_figura(
        grafica_riesgo_por_bloque(
            validacion, OBJETIVO_P2, "predicho", config.P2_CLASES, "Riesgo por bloque del dia"
        ),
        "riesgo_por_bloque",
    )


def correr_baseline(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run de la linea base. No registra modelo: es una regla sin parametros que ajustar."""
    with iniciar_run(PREGUNTA, "baseline", "frecuencia_mismo_bloque", None, meta_features) as run:
        probabilidades = {p: probabilidades_baseline(datos[p]) for p in PARTICIONES_EVALUACION}
        registrar_evaluacion(datos, probabilidades)
    return run.info.run_id


def correr_candidato(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run del candidato: entrena, evalua, explica (importancias) y registra el modelo."""
    parametros = {**config.P2_HIPERPARAMETROS, "n_variables": len(COLUMNAS_P2)}
    with iniciar_run(PREGUNTA, "candidato", "hgb_balanceado", parametros, meta_features) as run:
        modelo = entrenar_candidato(datos["entrenamiento"])
        probabilidades = {
            p: probabilidades_modelo(modelo, datos[p][COLUMNAS_P2]) for p in PARTICIONES_EVALUACION
        }
        registrar_evaluacion(datos, probabilidades)

        # Importancia por permutacion en validacion: datos que el modelo no vio al entrenar
        muestra = muestrear(datos["validacion"], config.P2_FILAS_IMPORTANCIA)
        importancia = permutation_importance(
            modelo,
            muestra[COLUMNAS_P2],
            muestra[OBJETIVO_P2],
            scoring="f1_macro",
            n_repeats=3,
            random_state=config.SEMILLA,
        )
        importancias = pd.Series(importancia.importances_mean, index=COLUMNAS_P2)
        registrar_figura(
            grafica_importancias(importancias, "Importancia de variables (caida del F1 macro)"),
            "importancias",
        )
        registrar_modelo(modelo, datos["entrenamiento"][COLUMNAS_P2].head(100))
    return run.info.run_id


def entrenar(
    roles: tuple[str, ...] = ROLES,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
) -> list[str]:
    """Punto de entrada que invoca train.py: corre los roles pedidos y devuelve sus run_id.

    ``datos`` y ``meta_features`` solo se pasan en las pruebas; por defecto se leen del
    Parquet de P2 y de los metadatos de features (que validan la version de los datos).
    """
    if datos is None:
        datos = cargar_dataset(config.DATASET_P2, columna_tiempo="bloque_inicio")
    corredores = {"baseline": correr_baseline, "candidato": correr_candidato}
    return [corredores[rol](datos, meta_features) for rol in roles]
