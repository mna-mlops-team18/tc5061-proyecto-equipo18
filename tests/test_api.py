"""Pruebas de la API REST (FastAPI) con un modelo de MLflow pequeno, guardado como lo exporta
``modeling/exportar.py``: endpoints, validacion de peticiones, carga del modelo y sus fallas."""

import copy
import json

from fastapi.testclient import TestClient
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from api import esquemas
from api.main import crear_app
from api.modelo import ErrorModelo, Identidad, ModeloServido, cargar_modelo
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun

EJEMPLO = esquemas.EJEMPLO_PETICION_P1
NOMBRE_P1 = config.MODELOS_REGISTRADOS["P1"]
CONTEOS = [c for c in esquemas.COLUMNAS_P1 if c.startswith(("salidas_", "llegadas_"))]
# Tipos con que features.py construye P1 (los que quedan en la firma del modelo)
TIPOS_ENTRENAMIENTO = {
    **{c: "int8" for c in ("hora_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes")},
    "capacidad": "int32",
    "latitud": "float64",
    "longitud": "float64",
    **{c: "float32" for c in CONTEOS},
}
REGISTRO = {
    "pregunta": "P1",
    "nombre": NOMBRE_P1,
    "version": "7",
    "alias": "champion",
    "run_id": "run-de-prueba",
    "exportado_en": "2026-11-05T12:00:00+00:00",
    "etiquetas": {"data_version": "abc123def456", "modelo": "hgb_poisson"},
}


def tabla_p1(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """Filas sinteticas de P1 con las columnas y tipos del entrenamiento, y su objetivo."""
    rng = np.random.default_rng(config.SEMILLA)
    dia = rng.integers(0, 7, n)
    X = pd.DataFrame(
        {
            "hora_dia": rng.integers(0, 24, n),
            "dia_semana": dia,
            "es_fin_semana": (dia >= 5).astype(int),
            "es_festivo": np.zeros(n, dtype=int),
            "mes": rng.integers(1, 13, n),
            "capacidad": rng.integers(10, 40, n),
            "latitud": rng.uniform(20.62, 20.72, n),
            "longitud": rng.uniform(-103.42, -103.30, n),
            **{c: rng.poisson(3, n) for c in CONTEOS},
        }
    )[esquemas.COLUMNAS_P1].astype(TIPOS_ENTRENAMIENTO)
    return X, rng.poisson(1 + X["salidas_lag_168h"].to_numpy())


def guardar_modelo(carpeta, X: pd.DataFrame, y, registro: dict | None = REGISTRO):
    """Guarda un HistGradientBoosting como MLflow (skops, firma y ejemplo) y su registro.json."""
    estimador = HistGradientBoostingRegressor(
        loss="poisson", max_iter=20, random_state=config.SEMILLA
    ).fit(X, y)
    mlflow.sklearn.save_model(
        estimador,
        str(carpeta),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
        skops_trusted_types=config.SKOPS_TIPOS_CONFIABLES,
        signature=infer_signature(X, estimador.predict(X)),
        input_example=X.head(5),
    )
    if registro is not None:
        (carpeta / "registro.json").write_text(json.dumps(registro), encoding="utf-8")
    return estimador


@pytest.fixture
def modelo_exportado(tmp_path):
    """Carpeta con un modelo de P1 como la deja ``exportar.py`` y el estimador original."""
    carpeta = tmp_path / "p1"
    X, y = tabla_p1()
    return carpeta, guardar_modelo(carpeta, X, y)


@pytest.fixture
def cliente(modelo_exportado):
    carpeta, _ = modelo_exportado
    app = crear_app(lambda: cargar_modelo(carpeta, columnas=esquemas.COLUMNAS_P1))
    with TestClient(app) as cliente:  # el with corre el arranque (carga del modelo)
        yield cliente


def filas_del_ejemplo() -> pd.DataFrame:
    filas = [{c: fila[c] for c in esquemas.COLUMNAS_P1} for fila in EJEMPLO["instancias"]]
    return pd.DataFrame(filas).astype(TIPOS_ENTRENAMIENTO)


# =============================================================================
# Endpoints
# =============================================================================
def test_salud_reporta_modelo_y_version(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo": NOMBRE_P1, "version": "7"}


def test_modelo_reporta_variables_y_linaje(cliente):
    cuerpo = cliente.get("/modelo").json()
    assert cuerpo["variables"] == esquemas.COLUMNAS_P1
    assert cuerpo["run_id"] == "run-de-prueba"
    assert cuerpo["etiquetas"]["data_version"] == "abc123def456"


def test_inicio_lleva_a_la_documentacion(cliente):
    respuesta = cliente.get("/", follow_redirects=False)
    assert respuesta.status_code in (302, 307)
    assert respuesta.headers["location"] == "/docs"
    assert "/predicciones/p1" in cliente.get("/openapi.json").json()["paths"]


def test_predice_lo_mismo_que_el_modelo_original(cliente, modelo_exportado):
    _, estimador = modelo_exportado
    respuesta = cliente.post("/predicciones/p1", json=EJEMPLO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == {
        "nombre": NOMBRE_P1,
        "version": "7",
        "alias": "champion",
        "run_id": "run-de-prueba",
    }
    assert cuerpo["n"] == 2
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == [101, 202]
    esperado = np.clip(estimador.predict(filas_del_ejemplo()), 0, None)
    obtenido = [p["salidas_esperadas"] for p in cuerpo["predicciones"]]
    assert obtenido == pytest.approx(esperado, abs=1e-3)
    assert all(valor >= 0 for valor in obtenido)


@pytest.mark.parametrize(
    ("cambios", "campo"),
    [
        ({"hora_dia": 24}, "hora_dia"),
        ({"es_festivo": "no"}, "es_festivo"),
        ({"latitud": 19.43}, "latitud"),
        ({"hora": 8}, "hora"),
    ],
)
def test_peticion_invalida_da_422_con_el_campo(cliente, cambios, campo):
    peticion = copy.deepcopy(EJEMPLO)
    peticion["instancias"][1].update(cambios)
    respuesta = cliente.post("/predicciones/p1", json=peticion)
    assert respuesta.status_code == 422
    ubicaciones = [detalle["loc"] for detalle in respuesta.json()["detail"]]
    assert ["body", "instancias", 1, campo] in ubicaciones


@pytest.mark.parametrize("cuerpo", [{}, {"instancias": []}, {"instancias": "fila"}])
def test_cuerpo_mal_formado_da_422(cliente, cuerpo):
    assert cliente.post("/predicciones/p1", json=cuerpo).status_code == 422


def test_texto_que_no_es_json_da_422(cliente):
    respuesta = cliente.post(
        "/predicciones/p1", content="no es json", headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422


# =============================================================================
# Respuestas del modelo que no son validas
# =============================================================================
class ModeloConstante:
    """Sustituto del modelo que siempre predice ``valor``."""

    def __init__(self, valor: float):
        self.valor = valor

    def predict(self, entrada: pd.DataFrame) -> np.ndarray:
        return np.full(len(entrada), self.valor)


def cliente_con(valor: float) -> TestClient:
    tipos = {c: np.dtype(t) for c, t in TIPOS_ENTRENAMIENTO.items()}
    servido = ModeloServido(ModeloConstante(valor), Identidad(NOMBRE_P1, "1"), tipos, "prueba")
    return TestClient(crear_app(lambda: servido))


def test_prediccion_negativa_se_recorta_a_cero():
    with cliente_con(-3.0) as cliente:
        cuerpo = cliente.post("/predicciones/p1", json=EJEMPLO).json()
    assert [p["salidas_esperadas"] for p in cuerpo["predicciones"]] == [0.0, 0.0]


def test_prediccion_no_finita_da_500():
    with cliente_con(float("nan")) as cliente:
        respuesta = cliente.post("/predicciones/p1", json=EJEMPLO)
    assert respuesta.status_code == 500
    assert "no finitos" in respuesta.json()["detail"]


# =============================================================================
# Carga del modelo
# =============================================================================
def test_lee_la_carpeta_de_modelo_uri(modelo_exportado, monkeypatch):
    carpeta, _ = modelo_exportado
    monkeypatch.setenv("MODELO_URI", str(carpeta))
    servido = cargar_modelo(columnas=esquemas.COLUMNAS_P1)
    assert (servido.identidad.nombre, servido.identidad.version) == (NOMBRE_P1, "7")
    # La firma de MLflow guarda los enteros de 8 bits como "integer" (int32); el resto igual
    firma = {c: "int32" if t == "int8" else t for c, t in TIPOS_ENTRENAMIENTO.items()}
    assert {c: str(t) for c, t in servido.tipos.items()} == firma


def test_carpeta_sin_modelo_no_arranca(tmp_path):
    app = crear_app(lambda: cargar_modelo(tmp_path / "vacia"))
    with pytest.raises(ErrorModelo, match="No hay un modelo"), TestClient(app):
        pass


def test_modelo_sin_registro_json_no_arranca(tmp_path):
    X, y = tabla_p1()
    guardar_modelo(tmp_path / "p1", X, y, registro=None)
    with pytest.raises(ErrorModelo, match="registro.json"):
        cargar_modelo(tmp_path / "p1")


def test_modelo_con_otras_variables_no_arranca(tmp_path):
    X, y = tabla_p1()
    guardar_modelo(tmp_path / "p1", X.drop(columns=["mes"]), y)
    with pytest.raises(ErrorModelo, match="mes"):
        cargar_modelo(tmp_path / "p1", columnas=esquemas.COLUMNAS_P1)


def test_carga_directa_del_model_registry_por_alias(mlflow_temporal):
    """Opcion B del README: MODELO_URI=models:/<nombre>@champion resuelve la version exacta."""
    X, y = tabla_p1()
    estimador = HistGradientBoostingRegressor(max_iter=10, random_state=config.SEMILLA).fit(X, y)
    with mlflow.start_run() as run:
        uri = comun.registrar_modelo(estimador, X)
    version = mlflow.register_model(uri, NOMBRE_P1).version
    mlflow.MlflowClient().set_registered_model_alias(NOMBRE_P1, "champion", version)

    servido = cargar_modelo(f"models:/{NOMBRE_P1}@champion", columnas=esquemas.COLUMNAS_P1)
    assert servido.identidad.version == str(version)
    assert servido.identidad.alias == "champion"
    assert servido.identidad.run_id == run.info.run_id
