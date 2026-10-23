"""P1 - Regresion: cuantos viajes saldran de cada estacion en cada hora del dia siguiente.

Modelos del catalogo ``config.P1_MODELOS``, todos evaluados con las mismas metricas y
particiones que en la entrega 1:

- ``rezago_semanal`` (baseline): las salidas de la misma hora de hace una semana. Es el
  pronostico ingenuo estacional; cualquier modelo debe superarlo para justificarse.
- ``hgb_poisson``: ``HistGradientBoostingRegressor`` con perdida Poisson (las salidas son
  conteos no negativos) sobre las variables de ``features.COLUMNAS_P1``.
- ``hgb_cuadratico``: el mismo algoritmo con perdida cuadratica, para medir cuanto aporta la
  perdida Poisson.
- ``glm_poisson``: modelo lineal generalizado Poisson; hora, dia y mes van en one-hot, los
  rezagos y medias de viajes pasan por ``log1p`` y todo lo numerico se escala. Es barato de
  entrenar y facil de interpretar.

Ademas de las metricas, cada candidato registra cuantas iteraciones uso su ajuste, si
scikit-learn aviso que no convergio y la fraccion de predicciones negativas (posible con la
perdida cuadratica; las salidas reales nunca son negativas).

Uso::

    mlflow run . -e train -P pregunta=p1 -P modelo=hgb_poisson --env-manager local
    mlflow run . -e search -P pregunta=p1 --env-manager local
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1, cargar_metadatos_features
from tc5061_proyecto_equipo18.modeling.comun import (
    cargar_dataset,
    cronometro,
    es_baseline,
    iniciar_run,
    muestrear,
    registrar_conjuntos,
    registrar_convergencia,
    registrar_costos,
    registrar_figura,
    registrar_importancias,
    registrar_metricas,
    registrar_modelo,
    resolver_hiperparametros,
    vigilar_convergencia,
)
from tc5061_proyecto_equipo18.plots import (
    grafica_error_por_hora,
    grafica_real_vs_predicho,
    grafica_residuales,
)

PREGUNTA = "P1"
PARTICIONES_EVALUACION = ("validacion", "prueba")
COLUMNA_BASELINE = "salidas_lag_168h"
# La devianza Poisson no esta definida para predicciones iguales a 0 (el baseline puede darlas)
PREDICCION_MINIMA = 1e-6
# Variables de calendario que el modelo lineal recibe en one-hot (su efecto no es lineal)
CATEGORICAS_GLM = ["hora_dia", "dia_semana", "mes"]
# Rezagos y medias de viajes: el GLM Poisson tiene enlace log, asi que entran como log1p (un
# efecto multiplicativo por cada 1 % mas de viajes) en lugar de crudos, que en estaciones con
# mucha demanda dispararian la prediccion con exp(b * viajes)
CONTEOS_GLM = [columna for columna in COLUMNAS_P1 if columna.startswith(("salidas_", "llegadas_"))]


def metricas_regresion(real: np.ndarray, predicho: np.ndarray) -> dict[str, float]:
    """Metricas de P1 para comparar todos los modelos.

    - ``mae``: error medio en viajes por estacion-hora; es la que entiende el operador.
    - ``rmse``: castiga mas los errores grandes (horas pico mal estimadas).
    - ``wape``: error absoluto total entre viajes totales; comparable entre periodos con
      distinto volumen y estable aunque muchas horas tengan 0 salidas.
    - ``devianza_poisson``: ajuste propio de conteos; es la perdida que optimiza ``hgb_poisson``.
    - ``r2``: fraccion de la variacion explicada.
    - ``fraccion_negativas``: predicciones menores que 0 (solo posibles con perdida cuadratica);
      se reportan sin recortar, igual que las devuelve el modelo registrado.
    """
    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)
    total = real.sum()
    return {
        "mae": mean_absolute_error(real, predicho),
        "rmse": root_mean_squared_error(real, predicho),
        "wape": np.abs(real - predicho).sum() / total if total > 0 else np.nan,
        "devianza_poisson": mean_poisson_deviance(
            real, np.clip(predicho, PREDICCION_MINIMA, None)
        ),
        "r2": r2_score(real, predicho),
        "fraccion_negativas": float(np.mean(predicho < 0)),
    }


def predecir_baseline(datos: pd.DataFrame) -> np.ndarray:
    """Pronostico ingenuo estacional: las salidas de la misma hora, siete dias antes.

    Usa ``salidas_lag_168h``, que ya respeta el horizonte de 24 h (se conoce al cierre del dia
    anterior), asi que el baseline tampoco usa informacion posterior a la prediccion.
    """
    return datos[COLUMNA_BASELINE].to_numpy(dtype=float)


def construir_estimador(modelo: str, hiperparametros: dict):
    """Estimador de scikit-learn sin entrenar para un modelo del catalogo de P1."""
    if modelo in ("hgb_poisson", "hgb_cuadratico"):
        return HistGradientBoostingRegressor(**hiperparametros, random_state=config.SEMILLA)
    if modelo == "glm_poisson":
        otras = [c for c in COLUMNAS_P1 if c not in CATEGORICAS_GLM and c not in CONTEOS_GLM]
        conteos = Pipeline(
            [
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("escala", StandardScaler()),
            ]
        )
        preparacion = ColumnTransformer(
            [
                ("calendario", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS_GLM),
                ("conteos", conteos, CONTEOS_GLM),
                ("numericas", StandardScaler(), otras),
            ]
        )
        return Pipeline(
            [
                ("prep", preparacion),
                ("glm", PoissonRegressor(**hiperparametros, solver="newton-cholesky")),
            ]
        )
    raise ValueError(f"P1 no sabe construir el modelo {modelo!r}")


def cargar_datos() -> dict[str, pd.DataFrame]:
    """Parquet de P1 (lo escribe features.py) separado en las tres particiones."""
    return cargar_dataset(config.DATASET_P1, columna_tiempo="hora")


def registrar_evaluacion(datos: dict[str, pd.DataFrame], predicciones: dict[str, np.ndarray]):
    """Registra las metricas por particion y las figuras de diagnostico de validacion."""
    for particion in PARTICIONES_EVALUACION:
        metricas = metricas_regresion(datos[particion][OBJETIVO_P1], predicciones[particion])
        registrar_metricas(metricas, particion)

    validacion = datos["validacion"][["estacion_id", "hora", "hora_dia", OBJETIVO_P1]].assign(
        predicho=predicciones["validacion"]
    )
    registrar_figura(
        grafica_real_vs_predicho(validacion, "hora", OBJETIVO_P1, "predicho", "Real vs. predicho"),
        "real_vs_predicho",
    )
    registrar_figura(
        grafica_residuales(
            validacion[OBJETIVO_P1].to_numpy(dtype=float),
            validacion["predicho"].to_numpy(dtype=float),
            "Residuales en validacion",
        ),
        "residuales",
    )
    registrar_figura(
        grafica_error_por_hora(validacion, OBJETIVO_P1, "predicho", "Error por hora del dia"),
        "error_por_hora",
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
    """Entrena (salvo el baseline), evalua y registra un run de P1; devuelve su run_id.

    ``hiperparametros`` son los ya resueltos (config.py con los cambios de ``-P``); si no se
    pasan se usan los de config.py. ``datos`` y ``meta_features`` se pasan en las busquedas
    (para leer el Parquet una sola vez) y en las pruebas.
    """
    if hiperparametros is None:
        hiperparametros = resolver_hiperparametros(PREGUNTA, modelo)
    if datos is None:
        datos = cargar_datos()
    meta = meta_features if meta_features is not None else cargar_metadatos_features()
    baseline = es_baseline(PREGUNTA, modelo)
    parametros = {"n_variables": len(COLUMNAS_P1)}
    if baseline:
        parametros["columna_baseline"] = COLUMNA_BASELINE
    else:
        parametros["max_filas_entrenamiento"] = config.P1_MAX_FILAS_ENTRENAMIENTO

    with iniciar_run(
        PREGUNTA, modelo, etapa, hiperparametros, meta, nombre, anidado, parametros
    ) as run:
        registrar_conjuntos(
            datos, PREGUNTA, config.DATASET_P1, COLUMNAS_P1, OBJETIVO_P1, meta["data_version"]
        )
        if baseline:
            with cronometro() as inferencia:
                predicciones = {p: predecir_baseline(datos[p]) for p in PARTICIONES_EVALUACION}
            registrar_costos(0.0, inferencia["segundos"], len(datos["validacion"]))
            registrar_evaluacion(datos, predicciones)
            return run.info.run_id

        # La muestra (P1_MAX_FILAS_ENTRENAMIENTO filas, con la semilla del proyecto) acota la
        # memoria sin perder estaciones ni horas: con 18 meses hay mas filas de las necesarias.
        muestra = muestrear(datos["entrenamiento"], config.P1_MAX_FILAS_ENTRENAMIENTO)
        estimador = construir_estimador(modelo, hiperparametros)
        with cronometro() as entrenamiento, vigilar_convergencia() as avisos:
            estimador.fit(muestra[COLUMNAS_P1], muestra[OBJETIVO_P1])
        with cronometro() as inferencia:
            predicciones = {"validacion": estimador.predict(datos["validacion"][COLUMNAS_P1])}
        predicciones["prueba"] = estimador.predict(datos["prueba"][COLUMNAS_P1])
        registrar_costos(
            entrenamiento["segundos"], inferencia["segundos"], len(datos["validacion"])
        )
        registrar_convergencia(estimador, avisos)  # tambien el GLM, que va en un Pipeline
        registrar_evaluacion(datos, predicciones)

        muestra_validacion = muestrear(datos["validacion"], config.P1_FILAS_IMPORTANCIA)
        registrar_importancias(
            estimador,
            muestra_validacion[COLUMNAS_P1],
            muestra_validacion[OBJETIVO_P1],
            scoring="neg_mean_absolute_error",
            titulo="Importancia de variables (caida del MAE)",
        )
        registrar_modelo(estimador, datos["entrenamiento"][COLUMNAS_P1].head(100))
    return run.info.run_id
