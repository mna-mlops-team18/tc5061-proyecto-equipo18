"""P2 - Clasificacion: una estacion se quedara sin bicicletas o sin anclajes en el siguiente bloque.

Clases (``config.P2_CLASES``): 0 sin riesgo, 1 riesgo de vaciado, 2 riesgo de llenado. Modelos
del catalogo ``config.P2_MODELOS``, todos evaluados con las mismas metricas y particiones que
en la entrega 1:

- ``frecuencia_mismo_bloque`` (baseline): la probabilidad de cada clase es la fraccion de los
  ultimos 7 dias en que ese mismo bloque estuvo en esa clase; predice la mas probable. Es la
  regla que el equipo de rebalanceo aplicaria con su experiencia ("esta estacion suele
  vaciarse a esta hora").
- ``hgb_balanceado``: ``HistGradientBoostingClassifier`` con pesos de clase balanceados (el
  candidato de la entrega 1).
- ``hgb_ponderado``: el mismo algoritmo con un peso ``peso_riesgo`` para las dos clases de
  riesgo. En la entrega 1 los pesos balanceados detectaban casi todos los riesgos pero con
  muchas alertas falsas; un peso intermedio cambia ese equilibrio.
- ``logistica``: regresion logistica multinomial con variables escaladas y el mismo
  ``peso_riesgo``: referencia lineal, barata e interpretable.

Uso::

    mlflow run . -e train -P pregunta=p2 -P modelo=hgb_ponderado -P peso_riesgo=10 --env-manager local
    mlflow run . -e search -P pregunta=p2 --env-manager local
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P2, OBJETIVO_P2, cargar_metadatos_features
from tc5061_proyecto_equipo18.modeling.comun import (
    cargar_dataset,
    cronometro,
    es_baseline,
    iniciar_run,
    muestrear,
    registrar_conjuntos,
    registrar_costos,
    registrar_figura,
    registrar_importancias,
    registrar_metricas,
    registrar_modelo,
    resolver_hiperparametros,
)
from tc5061_proyecto_equipo18.plots import (
    grafica_curvas_pr,
    grafica_matriz_confusion,
    grafica_riesgo_por_bloque,
)

PREGUNTA = "P2"
PARTICIONES_EVALUACION = ("validacion", "prueba")
CLASES = list(config.P2_CLASES)
CLASES_ARRAY = np.array(CLASES)  # para pasar de indice de columna a clase
CLASES_RIESGO = {1: "vaciado", 2: "llenado"}


def metricas_clasificacion(real: np.ndarray, probabilidades: np.ndarray) -> dict[str, float]:
    """Metricas de P2 para comparar todos los modelos.

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


def probabilidades_modelo(modelo, X: pd.DataFrame) -> np.ndarray:
    """``predict_proba`` con una columna por clase de ``CLASES``, aunque falte alguna en el ajuste."""
    probabilidades = np.zeros((len(X), len(CLASES)))
    probabilidades[:, modelo.classes_] = modelo.predict_proba(X)
    return probabilidades


def pesos_clase(peso_riesgo: float) -> dict[int, float]:
    """Peso de cada clase: 1 para "sin riesgo" y ``peso_riesgo`` para vaciado y llenado."""
    return {0: 1.0, 1: float(peso_riesgo), 2: float(peso_riesgo)}


def construir_estimador(modelo: str, hiperparametros: dict):
    """Estimador de scikit-learn sin entrenar para un modelo del catalogo de P2."""
    hiperparametros = dict(hiperparametros)
    if modelo == "hgb_balanceado":
        return HistGradientBoostingClassifier(**hiperparametros, random_state=config.SEMILLA)
    if modelo == "hgb_ponderado":
        pesos = pesos_clase(hiperparametros.pop("peso_riesgo"))
        return HistGradientBoostingClassifier(
            **hiperparametros, class_weight=pesos, random_state=config.SEMILLA
        )
    if modelo == "logistica":
        pesos = pesos_clase(hiperparametros.pop("peso_riesgo"))
        return Pipeline(
            [
                ("escala", StandardScaler()),
                (
                    "logistica",
                    LogisticRegression(
                        **hiperparametros, class_weight=pesos, random_state=config.SEMILLA
                    ),
                ),
            ]
        )
    raise ValueError(f"P2 no sabe construir el modelo {modelo!r}")


def cargar_datos() -> dict[str, pd.DataFrame]:
    """Parquet de P2 (lo escribe features.py) separado en las tres particiones."""
    return cargar_dataset(config.DATASET_P2, columna_tiempo="bloque_inicio")


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


def correr(
    modelo: str,
    hiperparametros: dict | None = None,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
    etapa: str | None = None,
    anidado: bool = False,
    nombre: str | None = None,
) -> str:
    """Entrena (salvo el baseline), evalua y registra un run de P2; devuelve su run_id.

    ``hiperparametros`` son los ya resueltos (config.py con los cambios de ``-P``); si no se
    pasan se usan los de config.py. ``datos`` y ``meta_features`` se pasan en las busquedas
    (para leer el Parquet una sola vez) y en las pruebas.
    """
    if hiperparametros is None:
        hiperparametros = resolver_hiperparametros(PREGUNTA, modelo)
    if datos is None:
        datos = cargar_datos()
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    parametros = {"n_variables": len(COLUMNAS_P2)}

    with iniciar_run(
        PREGUNTA, modelo, etapa, hiperparametros, meta, nombre, anidado, parametros
    ) as run:
        registrar_conjuntos(
            datos, PREGUNTA, config.DATASET_P2, COLUMNAS_P2, OBJETIVO_P2, meta["data_version"]
        )
        if es_baseline(PREGUNTA, modelo):
            with cronometro() as inferencia:
                probabilidades = {
                    p: probabilidades_baseline(datos[p]) for p in PARTICIONES_EVALUACION
                }
            registrar_costos(0.0, inferencia["segundos"], len(datos["validacion"]))
            registrar_evaluacion(datos, probabilidades)
            return run.info.run_id

        estimador = construir_estimador(modelo, hiperparametros)
        entrenamiento = datos["entrenamiento"]
        with cronometro() as ajuste:
            estimador.fit(entrenamiento[COLUMNAS_P2], entrenamiento[OBJETIVO_P2])
        with cronometro() as inferencia:
            probabilidades = {
                "validacion": probabilidades_modelo(estimador, datos["validacion"][COLUMNAS_P2])
            }
        probabilidades["prueba"] = probabilidades_modelo(estimador, datos["prueba"][COLUMNAS_P2])
        registrar_costos(ajuste["segundos"], inferencia["segundos"], len(datos["validacion"]))
        if hasattr(estimador, "n_iter_"):  # iteraciones reales con parada temprana
            registrar_metricas({"iteraciones": np.max(estimador.n_iter_)}, "entrenamiento")
        registrar_evaluacion(datos, probabilidades)

        muestra_validacion = muestrear(datos["validacion"], config.P2_FILAS_IMPORTANCIA)
        registrar_importancias(
            estimador,
            muestra_validacion[COLUMNAS_P2],
            muestra_validacion[OBJETIVO_P2],
            scoring="f1_macro",
            titulo="Importancia de variables (caida del F1 macro)",
        )
        registrar_modelo(estimador, entrenamiento[COLUMNAS_P2].head(100))
    return run.info.run_id
