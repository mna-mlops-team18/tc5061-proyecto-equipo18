"""Pruebas del Continuous Training de P1: periodos sin fuga, configuracion del champion y registro
del reentrenado como nueva version con ``@challenger`` (sin tocar ``@champion``)."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from monitoreo import sintetico
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1
from tc5061_proyecto_equipo18.modeling import comun, reentrenar, registro

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
NOMBRE_P1 = config.MODELOS_REGISTRADOS["P1"]


@pytest.fixture(scope="module")
def p1():
    """P1 sintetico de oct 2024 a sep 2026 (un dia de cada 8, para que sea rapido)."""
    return sintetico.p1_sintetico(conocidas=12, nuevas=4, cada_dias=8)


def registrar_champion(modelo: str = "hgb_poisson", cambios: dict | None = None) -> str:
    """Run de P1 con un modelo cualquiera, registrado como version con ``@champion``."""
    hiperparametros = comun.resolver_hiperparametros("P1", modelo, cambios)
    with comun.iniciar_run(
        "P1", modelo, hiperparametros=hiperparametros, meta_features=META
    ) as run:
        if not comun.es_baseline("P1", modelo):
            X = pd.DataFrame(np.ones((10, len(COLUMNAS_P1))), columns=COLUMNAS_P1)
            comun.registrar_modelo(DummyRegressor().fit(X, np.arange(10.0)), X)
    nombre = registro.asegurar_modelo_registrado("P1")
    model_id = registro.model_id_de(mlflow.get_run(run.info.run_id))
    if model_id is not None:
        version = mlflow.register_model(f"models:/{model_id}", nombre)
        registro.asignar_alias(nombre, config.ALIAS_PRODUCCION, version)
    return run.info.run_id


# =============================================================================
# Periodos: el challenger no puede ver el mes del A/B
# =============================================================================
def test_periodos_de_config_son_validos():
    reentrenar.verificar_periodos()  # REENTRENAMIENTO_FIN antes de AB_EVALUACION


@pytest.mark.parametrize(
    ("fin", "mensaje"),
    [
        ("2026-09-01", "anterior a AB_EVALUACION"),  # entrenaria con el mes del A/B
        ("2026-09-30", "anterior a AB_EVALUACION"),
        (config.FIN_ENTRENAMIENTO, "no agrega datos"),  # igual que el champion
    ],
)
def test_rechaza_periodos_con_fuga_o_sin_datos_nuevos(fin, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        reentrenar.verificar_periodos(fin)


def test_datos_llegan_hasta_el_fin_y_no_tocan_la_evaluacion(p1):
    datos = reentrenar.datos_reentrenamiento(p1)
    tiempo = pd.to_datetime(datos["hora"])
    assert tiempo.max() <= reentrenar.fin_del_dia(config.REENTRENAMIENTO_FIN)
    assert tiempo.max() < pd.Timestamp(config.AB_EVALUACION[0])
    assert tiempo.min() == pd.to_datetime(p1["hora"]).min()  # desde el inicio de los datos
    # Agrega lo que el champion no vio: despues de FIN_ENTRENAMIENTO
    assert (tiempo > reentrenar.fin_del_dia(config.FIN_ENTRENAMIENTO)).any()


def test_falla_si_los_datos_no_llegan_al_fin(p1):
    viejos = p1[pd.to_datetime(p1["hora"]) < "2026-06-01"]
    with pytest.raises(ValueError, match="sin pasar de"):
        reentrenar.datos_reentrenamiento(viejos)


# =============================================================================
# Configuracion del champion
# =============================================================================
def test_toma_modelo_e_hiperparametros_del_run_del_champion(mlflow_temporal):
    run_id = registrar_champion("hgb_poisson", {"learning_rate": "0.05", "max_leaf_nodes": "31"})
    modelo, hiperparametros = reentrenar.configuracion_del_champion(mlflow.get_run(run_id))
    assert modelo == "hgb_poisson"
    assert hiperparametros["learning_rate"] == 0.05  # el del run, no el de config.py
    assert hiperparametros["max_leaf_nodes"] == 31 and isinstance(
        hiperparametros["max_leaf_nodes"], int
    )
    assert hiperparametros["loss"] == "poisson"


def test_falla_sin_champion(mlflow_temporal):
    with pytest.raises(RuntimeError, match="Registra primero el champion"):
        reentrenar.champion_actual()


def test_no_reentrena_un_baseline(mlflow_temporal):
    run_id = registrar_champion("rezago_semanal")
    with pytest.raises(RuntimeError, match="baseline"):
        reentrenar.configuracion_del_champion(mlflow.get_run(run_id))


# =============================================================================
# Flujo completo
# =============================================================================
def test_registra_nueva_version_como_challenger_sin_mover_el_champion(mlflow_temporal, p1):
    run_champion = registrar_champion(
        "hgb_poisson", {"max_iter": "30", "learning_rate": "0.2", "early_stopping": "False"}
    )
    cliente = mlflow.MlflowClient()
    champion = cliente.get_model_version_by_alias(NOMBRE_P1, config.ALIAS_PRODUCCION)

    nueva = reentrenar.reentrenar(p1=p1, meta_features=META)

    # Nueva version del MISMO modelo registrado, con @challenger; @champion no se mueve
    assert nueva.name == NOMBRE_P1 and int(nueva.version) == int(champion.version) + 1
    retador = cliente.get_model_version_by_alias(NOMBRE_P1, config.ALIAS_RETADOR)
    assert str(retador.version) == str(nueva.version)
    sigue = cliente.get_model_version_by_alias(NOMBRE_P1, config.ALIAS_PRODUCCION)
    assert str(sigue.version) == str(champion.version)

    # Linaje en la version
    assert nueva.tags["version_origen"] == str(champion.version)
    assert nueva.tags["run_origen"] == run_champion
    assert nueva.tags["etapa"] == "reentrenamiento"
    assert nueva.tags["reentrenamiento_fin"] == config.REENTRENAMIENTO_FIN
    assert nueva.tags["data_version"] == META["data_version"]
    assert "Continuous Training" in nueva.description

    # El run usa los hiperparametros del champion y registra su conjunto de datos
    run = mlflow.get_run(nueva.run_id)
    assert run.data.tags["etapa"] == "reentrenamiento"
    assert run.data.params["learning_rate"] == "0.2" and run.data.params["max_iter"] == "30"
    assert run.data.params["reentrenamiento_fin"] == config.REENTRENAMIENTO_FIN
    assert [entrada.tags[0].value for entrada in run.inputs.dataset_inputs] == ["reentrenamiento"]
    assert "costo_entrenamiento_seg" in run.data.metrics

    # Se carga por alias y predice conteos no negativos (perdida Poisson)
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_P1}@{config.ALIAS_RETADOR}")
    predicho = modelo.predict(p1[COLUMNAS_P1].head(50))
    assert len(predicho) == 50 and (np.asarray(predicho) >= 0).all()


def test_el_register_no_selecciona_runs_de_reentrenamiento():
    """El reentrenado ya vio la validacion: no compite en el register, solo en el A/B."""
    assert reentrenar.ETAPA in comun.ETAPAS
    assert reentrenar.ETAPA not in registro.ETAPAS_REGISTRABLES
