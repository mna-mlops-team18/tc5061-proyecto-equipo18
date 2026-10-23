"""Pruebas del Model Registry: seleccion contra el baseline, linaje, aliases y carga."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun, registro

META = {"data_version": "abc123def456", "parametros": {"mes_inicio": "2024-10"}}
NOMBRE_P1 = config.MODELOS_REGISTRADOS["P1"]


def run_p1(modelo: str, mae: float, cambios: dict | None = None, con_modelo: bool = True) -> str:
    """Run falso de P1 con su metrica de seleccion y, si aplica, un modelo con ejemplo."""
    hiperparametros = comun.resolver_hiperparametros("P1", modelo, cambios)
    with comun.iniciar_run(
        "P1", modelo, hiperparametros=hiperparametros, meta_features=META
    ) as run:
        comun.registrar_metricas({"mae": mae}, "validacion")
        if con_modelo:
            X = pd.DataFrame({"a": np.arange(10.0), "b": np.ones(10)})
            comun.registrar_modelo(DummyRegressor().fit(X, np.arange(10.0)), X)
    return run.info.run_id


@pytest.fixture
def runs_p1(mlflow_temporal):
    """Baseline (MAE 1.0) y tres candidatos: hgb_poisson 0.75, el mismo repetido 0.75 y GLM 0.82."""
    return {
        "baseline": run_p1("rezago_semanal", 1.0, con_modelo=False),
        "hgb": run_p1("hgb_poisson", 0.75),
        "hgb_repetido": run_p1("hgb_poisson", 0.75),
        "glm": run_p1("glm_poisson", 0.82),
        "cuadratico_lr": run_p1("hgb_cuadratico", 0.80, {"learning_rate": "0.05"}),
    }


def experimento():
    return mlflow.get_experiment_by_name("prueba").experiment_id


def test_mejora_relativa():
    assert registro.mejora_relativa(0.75, 1.0, "min") == pytest.approx(0.25)
    assert registro.mejora_relativa(0.6, 0.5, "max") == pytest.approx(0.2)
    assert registro.mejora_relativa(1.2, 1.0, "min") < 0


def test_seleccion_ordena_y_excluye_baseline(runs_p1):
    seleccion = registro.seleccionar("p1", experimento(), solo_oficiales=False)
    assert seleccion.baseline.info.run_id == runs_p1["baseline"]
    assert runs_p1["baseline"] not in set(seleccion.ranking["run_id"])
    assert list(seleccion.ranking["validacion_mae"]) == [0.75, 0.75, 0.80, 0.82]
    assert seleccion.ranking.loc[0, "mejora_vs_baseline"] == pytest.approx(0.25)


def test_solo_oficiales_excluye_runs_fuera_del_proyecto(runs_p1):
    with pytest.raises(RuntimeError, match="baseline"):
        registro.seleccionar("p1", experimento(), solo_oficiales=True)


def test_challenger_es_otra_configuracion(runs_p1):
    seleccion = registro.seleccionar("p1", experimento(), solo_oficiales=False)
    champion, retador = registro.elegir(seleccion)
    assert champion["modelo"] == "hgb_poisson"
    # El segundo hgb_poisson repite la configuracion; el challenger es el siguiente distinto
    assert retador["modelo"] == "hgb_cuadratico"
    _, sin_retador = registro.elegir(seleccion, challenger_run_id=registro.NINGUNO)
    assert sin_retador is None


def test_rechaza_champion_peor_que_el_baseline_o_sin_modelo(mlflow_temporal):
    run_p1("rezago_semanal", 1.0, con_modelo=False)
    peor = run_p1("glm_poisson", 1.3)
    sin_modelo = run_p1("hgb_poisson", 0.5, con_modelo=False)
    seleccion = registro.seleccionar("p1", experimento(), solo_oficiales=False)
    with pytest.raises(RuntimeError, match="no mejora al baseline"):
        registro.elegir(seleccion, run_id=peor)
    with pytest.raises(RuntimeError, match="no es registrable"):
        registro.elegir(seleccion, run_id=sin_modelo)


def test_registrar_asigna_aliases_linaje_y_se_carga(runs_p1):
    resumen = registro.registrar("p1", experimento_id=experimento(), solo_oficiales=False)
    cliente = mlflow.MlflowClient()

    modelo = cliente.get_registered_model(NOMBRE_P1)
    assert modelo.description == config.DESCRIPCIONES_REGISTRO["P1"]
    assert modelo.tags["pregunta"] == "P1"

    champion = cliente.get_model_version_by_alias(NOMBRE_P1, config.ALIAS_PRODUCCION)
    assert champion.run_id == resumen["champion_run_id"]
    assert champion.tags["data_version"] == "abc123def456"
    assert champion.tags["experimento"] == "prueba"
    assert champion.tags["baseline_run_id"] == runs_p1["baseline"]
    assert champion.tags["comando_reproduccion"].startswith("mlflow run . -e train")
    assert "git_commit" in champion.tags and "mejora 25.0%" in champion.description

    retador = cliente.get_model_version_by_alias(NOMBRE_P1, config.ALIAS_RETADOR)
    assert retador.run_id == runs_p1["cuadratico_lr"]
    assert registro.verificar_carga(NOMBRE_P1) == 5  # ejemplo de entrada de 5 filas


def test_registrar_dos_veces_no_duplica_versiones(runs_p1):
    registro.registrar("p1", experimento_id=experimento(), solo_oficiales=False)
    registro.registrar("p1", experimento_id=experimento(), solo_oficiales=False)
    versiones = mlflow.MlflowClient().search_model_versions(f"name = '{NOMBRE_P1}'")
    assert len(versiones) == 2  # champion y challenger


def test_promover_otro_run_mueve_el_alias(runs_p1):
    registro.registrar("p1", experimento_id=experimento(), solo_oficiales=False)
    registro.registrar(
        "p1",
        run_id=runs_p1["glm"],
        challenger_run_id=registro.NINGUNO,
        experimento_id=experimento(),
        solo_oficiales=False,
    )
    cliente = mlflow.MlflowClient()
    assert cliente.get_model_version_by_alias(NOMBRE_P1, "champion").run_id == runs_p1["glm"]
    with pytest.raises(mlflow.exceptions.MlflowException):
        cliente.get_model_version_by_alias(NOMBRE_P1, "challenger")
