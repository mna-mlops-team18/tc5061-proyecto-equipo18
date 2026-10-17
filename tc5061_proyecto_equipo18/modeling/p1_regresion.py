"""P1 - Regresion: cuantos viajes saldran de cada estacion en cada hora del dia siguiente.

Dos runs en MLflow, evaluados con las mismas metricas y particiones:

- ``baseline`` (``rezago_semanal``): predice las salidas de la misma hora de hace una semana.
  Es el pronostico ingenuo estacional; cualquier modelo debe superarlo para justificarse.
- ``candidato`` (``hgb_poisson``): ``HistGradientBoostingRegressor`` con perdida Poisson
  (las salidas son conteos no negativos) sobre las variables de ``features.COLUMNAS_P1``.

Ambos se evaluan en validacion (para comparar y elegir) y en prueba (estimacion final).

Uso::

    python -m tc5061_proyecto_equipo18.modeling.train --pregunta P1
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    r2_score,
    root_mean_squared_error,
)

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1
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
    grafica_error_por_hora,
    grafica_importancias,
    grafica_real_vs_predicho,
)

PREGUNTA = "P1"
PARTICIONES_EVALUACION = ("validacion", "prueba")
COLUMNA_BASELINE = "salidas_lag_168h"
# La devianza Poisson no esta definida para predicciones iguales a 0 (el baseline puede darlas)
PREDICCION_MINIMA = 1e-6


def metricas_regresion(real: np.ndarray, predicho: np.ndarray) -> dict[str, float]:
    """Metricas de P1 para comparar baseline y candidato.

    - ``mae``: error medio en viajes por estacion-hora; es la que entiende el operador.
    - ``rmse``: castiga mas los errores grandes (horas pico mal estimadas).
    - ``wape``: error absoluto total entre viajes totales; comparable entre periodos con
      distinto volumen y estable aunque muchas horas tengan 0 salidas.
    - ``devianza_poisson``: ajuste propio de conteos; es la perdida que optimiza el candidato.
    - ``r2``: fraccion de la variacion explicada.
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
    }


def predecir_baseline(datos: pd.DataFrame) -> np.ndarray:
    """Pronostico ingenuo estacional: las salidas de la misma hora, siete dias antes.

    Usa ``salidas_lag_168h``, que ya respeta el horizonte de 24 h (se conoce al cierre del dia
    anterior), asi que el baseline tampoco usa informacion posterior a la prediccion.
    """
    return datos[COLUMNA_BASELINE].to_numpy(dtype=float)


def entrenar_candidato(entrenamiento: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Ajusta el gradient boosting con perdida Poisson sobre una muestra del entrenamiento.

    La muestra (``P1_MAX_FILAS_ENTRENAMIENTO`` filas, con la semilla del proyecto) acota la
    memoria sin perder estaciones ni horas: con 18 meses hay mas filas de las necesarias.
    """
    muestra = muestrear(entrenamiento, config.P1_MAX_FILAS_ENTRENAMIENTO)
    modelo = HistGradientBoostingRegressor(
        **config.P1_HIPERPARAMETROS, random_state=config.SEMILLA
    )
    return modelo.fit(muestra[COLUMNAS_P1], muestra[OBJETIVO_P1])


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
        grafica_error_por_hora(validacion, OBJETIVO_P1, "predicho", "Error por hora del dia"),
        "error_por_hora",
    )


def correr_baseline(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run de la linea base. No registra modelo: es una regla sin parametros que ajustar."""
    with iniciar_run(
        PREGUNTA,
        "baseline",
        "rezago_semanal",
        {"columna_baseline": COLUMNA_BASELINE},
        meta_features,
    ) as run:
        predicciones = {p: predecir_baseline(datos[p]) for p in PARTICIONES_EVALUACION}
        registrar_evaluacion(datos, predicciones)
    return run.info.run_id


def correr_candidato(datos: dict[str, pd.DataFrame], meta_features: dict | None = None) -> str:
    """Run del candidato: entrena, evalua, explica (importancias) y registra el modelo."""
    parametros = {
        **config.P1_HIPERPARAMETROS,
        "max_filas_entrenamiento": config.P1_MAX_FILAS_ENTRENAMIENTO,
        "n_variables": len(COLUMNAS_P1),
    }
    with iniciar_run(PREGUNTA, "candidato", "hgb_poisson", parametros, meta_features) as run:
        modelo = entrenar_candidato(datos["entrenamiento"])
        predicciones = {p: modelo.predict(datos[p][COLUMNAS_P1]) for p in PARTICIONES_EVALUACION}
        registrar_evaluacion(datos, predicciones)

        # Importancia por permutacion en validacion: datos que el modelo no vio al entrenar
        muestra = muestrear(datos["validacion"], config.P1_FILAS_IMPORTANCIA)
        importancia = permutation_importance(
            modelo,
            muestra[COLUMNAS_P1],
            muestra[OBJETIVO_P1],
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=config.SEMILLA,
        )
        importancias = pd.Series(importancia.importances_mean, index=COLUMNAS_P1)
        registrar_figura(
            grafica_importancias(importancias, "Importancia de variables (caida del MAE)"),
            "importancias",
        )
        registrar_modelo(modelo, datos["entrenamiento"][COLUMNAS_P1].head(100))
    return run.info.run_id


def entrenar(
    roles: tuple[str, ...] = ROLES,
    datos: dict[str, pd.DataFrame] | None = None,
    meta_features: dict | None = None,
) -> list[str]:
    """Punto de entrada que invoca train.py: corre los roles pedidos y devuelve sus run_id.

    ``datos`` y ``meta_features`` solo se pasan en las pruebas; por defecto se leen del
    Parquet de P1 y de los metadatos de features (que validan la version de los datos).
    """
    if datos is None:
        datos = cargar_dataset(config.DATASET_P1, columna_tiempo="hora")
    corredores = {"baseline": correr_baseline, "candidato": correr_candidato}
    return [corredores[rol](datos, meta_features) for rol in roles]
