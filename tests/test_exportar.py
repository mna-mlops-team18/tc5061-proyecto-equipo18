"""Pruebas de la exportacion del Model Registry a la carpeta que empaqueta la API."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun, exportar

NOMBRE_P1 = config.MODELOS_REGISTRADOS["P1"]


def registrar_version(valor: float, alias: str | None = None) -> str:
    """Registra en P1 un modelo que siempre predice ``valor``; devuelve el numero de version."""
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.ones(10)})
    with mlflow.start_run():
        uri = comun.registrar_modelo(
            DummyRegressor(constant=valor, strategy="constant").fit(X, np.zeros(10)), X
        )
    version = mlflow.register_model(uri, NOMBRE_P1).version
    if alias:
        mlflow.MlflowClient().set_registered_model_alias(NOMBRE_P1, alias, version)
    return str(version)


@pytest.fixture
def registry(mlflow_temporal):
    """Dos versiones de P1: la 1 (predice 1.0) es challenger y la 2 (predice 2.0) es champion."""
    return {
        "challenger": registrar_version(1.0, config.ALIAS_RETADOR),
        "champion": registrar_version(2.0, config.ALIAS_PRODUCCION),
    }


def test_exporta_el_champion_con_su_version_y_huellas(registry, tmp_path):
    destino = tmp_path / "servicio" / "p1"
    registro = exportar.exportar("p1", destino=destino)

    assert registro["nombre"] == NOMBRE_P1
    assert registro["version"] == registry["champion"]
    assert registro["alias"] == config.ALIAS_PRODUCCION
    assert config.ALIAS_PRODUCCION in registro["aliases_al_exportar"]
    assert (destino / "MLmodel").is_file()
    escrito = json.loads((destino / exportar.ARCHIVO_REGISTRO).read_text(encoding="utf-8"))
    assert escrito == registro
    for archivo, huella in registro["archivos"].items():
        assert exportar.huella_sha256(destino / archivo) == huella
    # La copia predice lo mismo que la version del registry (el champion predice 2.0)
    modelo = mlflow.pyfunc.load_model(str(destino))
    assert np.allclose(modelo.predict(modelo.input_example), 2.0)


def test_peticion_de_ejemplo_tiene_las_columnas_de_la_firma(registry, tmp_path):
    destino = tmp_path / "p1"
    exportar.exportar("p1", destino=destino)
    peticion = json.loads((destino / exportar.ARCHIVO_PETICION).read_text(encoding="utf-8"))
    assert len(peticion["instancias"]) == 5  # comun.registrar_modelo guarda 5 filas de ejemplo
    assert set(peticion["instancias"][0]) == {"a", "b"}


def test_version_explicita_ignora_el_alias(registry, tmp_path):
    registro = exportar.exportar("p1", version=registry["challenger"], destino=tmp_path / "p1")
    assert registro["version"] == registry["challenger"]
    assert registro["alias"] is None


def test_reexportar_reemplaza_la_carpeta_completa(registry, tmp_path):
    destino = tmp_path / "p1"
    exportar.exportar("p1", version=registry["challenger"], destino=destino)
    (destino / "resto_de_otra_version.txt").write_text("viejo", encoding="utf-8")
    registro = exportar.exportar("p1", destino=destino)
    assert registro["version"] == registry["champion"]
    assert not (destino / "resto_de_otra_version.txt").exists()


def test_no_borra_una_carpeta_que_no_es_una_exportacion(registry, tmp_path):
    ajena = tmp_path / "otra"
    ajena.mkdir()
    (ajena / "importante.txt").write_text("no borrar", encoding="utf-8")
    with pytest.raises(ValueError, match="no son una exportacion"):
        exportar.exportar("p1", destino=ajena)
    assert (ajena / "importante.txt").exists()


def test_alias_inexistente_da_un_mensaje_claro(registry, tmp_path):
    with pytest.raises(RuntimeError, match="No existe models:/"):
        exportar.exportar("p1", alias="no-existe", destino=tmp_path / "p1")


def test_destino_por_defecto_es_models_servicio():
    assert exportar.destino_de("P1") == config.MODELS_DIR / "servicio" / "p1"
