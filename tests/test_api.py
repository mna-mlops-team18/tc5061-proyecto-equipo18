"""Pruebas de la API REST (FastAPI) con modelos de MLflow pequenos, guardados como los exporta
``modeling/exportar.py``: endpoints de P1, P2 y P3, validacion de peticiones, carga de los modelos
y sus fallas."""

import copy
import json

from fastapi.testclient import TestClient
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from api import esquemas
from api.main import crear_app
from api.modelo import ErrorModelo, Identidad, ModeloServido, cargar_modelo
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.modeling import comun

PREGUNTAS = ("p1", "p2", "p3")
EJEMPLOS = {
    "p1": esquemas.EJEMPLO_PETICION_P1,
    "p2": esquemas.EJEMPLO_PETICION_P2,
    "p3": esquemas.EJEMPLO_PETICION_P3,
}
NOMBRES = {p: config.MODELOS_REGISTRADOS[p.upper()] for p in PREGUNTAS}
NOMBRE_P1 = NOMBRES["p1"]
VERSION = "7"
ENTEROS = ("hora_dia", "bloque_dia", "dia_semana", "es_fin_semana", "es_festivo", "mes")
# Tipos de la firma de los champions registrados: los enteros y las banderas son ``integer`` (int32),
# capacidad y coordenadas son ``double`` y los conteos son ``float``.
TIPOS = {
    p: {
        c: np.dtype(
            "int32"
            if c in ENTEROS
            else "float32"
            if c.startswith(("sal", "lle", "neto", "frac")) and p != "p3"
            else "float64"
        )
        for c in esquemas.COLUMNAS[p]
    }
    for p in PREGUNTAS
}


def registro_de(pregunta: str) -> dict:
    """Lo que ``exportar.py`` escribe en ``registro.json`` (solo lo que lee la API)."""
    return {
        "pregunta": pregunta.upper(),
        "nombre": NOMBRES[pregunta],
        "version": VERSION,
        "alias": "champion",
        "run_id": f"run-de-prueba-{pregunta}",
        "exportado_en": "2026-11-05T12:00:00+00:00",
        "etiquetas": {"data_version": "abc123def456", "modelo": "modelo-de-prueba"},
    }


# =============================================================================
# Modelos de prueba: pequenos, con las columnas y los tipos de cada pregunta
# =============================================================================
def tabla(pregunta: str, n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """Filas sinteticas de la pregunta con las columnas y tipos de su firma, y su objetivo."""
    rng = np.random.default_rng(config.SEMILLA)
    if pregunta == "p3":
        columnas = {
            f"{perfil}_h{hora:02d}": columna
            for perfil in esquemas.PERFILES_P3
            for hora, columna in enumerate(rng.dirichlet(np.ones(24), n).T)
        }
        datos = pd.DataFrame(
            {
                **columnas,
                "log_viajes_dia": rng.uniform(2, 6, n),
                "ratio_fin_semana": rng.uniform(0.2, 3, n),
                "balance_neto": rng.uniform(-0.4, 0.5, n),
            }
        )
        return datos[esquemas.COLUMNAS_P3].astype(TIPOS["p3"]), np.zeros(n)

    dia = rng.integers(0, 7, n)
    base = {
        "dia_semana": dia,
        "es_fin_semana": (dia >= 5).astype(int),
        "es_festivo": np.zeros(n, dtype=int),
        "mes": rng.integers(1, 13, n),
        "capacidad": rng.integers(10, 40, n),
        "latitud": rng.uniform(20.62, 20.72, n),
        "longitud": rng.uniform(-103.42, -103.30, n),
    }
    if pregunta == "p1":
        datos = pd.DataFrame(
            {
                **base,
                "hora_dia": rng.integers(0, 24, n),
                **{
                    c: rng.poisson(3, n)
                    for c in esquemas.COLUMNAS_P1
                    if c.startswith(("sal", "lle"))
                },
            }
        )
        X = datos[esquemas.COLUMNAS_P1].astype(TIPOS["p1"])
        return X, rng.poisson(1 + X["salidas_lag_168h"].to_numpy())

    datos = pd.DataFrame(
        {
            **base,
            "bloque_dia": rng.integers(0, 8, n) * 3,
            **{c: rng.normal(0, 3, n) for c in esquemas.COLUMNAS_P2 if c.startswith("neto")},
            **{c: rng.uniform(0, 1, n) for c in esquemas.COLUMNAS_P2 if c.startswith("frac")},
            **{
                c: rng.poisson(5, n)
                for c in ("salidas_lag_1b", "llegadas_lag_1b", "salidas_previas_24h")
            },
        }
    )
    return datos[esquemas.COLUMNAS_P2].astype(TIPOS["p2"]), rng.integers(0, 3, n)


def estimador_de(pregunta: str):
    """Estimador pequeno del tipo de la pregunta: regresor, clasificador o agrupador."""
    semilla = config.SEMILLA
    if pregunta == "p1":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=20, random_state=semilla)
    if pregunta == "p2":
        return HistGradientBoostingClassifier(max_iter=20, random_state=semilla)
    return KMeans(n_clusters=3, n_init=1, random_state=semilla)


def guardar_modelo(carpeta, pregunta: str, X: pd.DataFrame, y, registro: dict | None = True):
    """Guarda el modelo como MLflow (skops, firma y ejemplo) y, si se pide, su registro.json."""
    estimador = estimador_de(pregunta).fit(X, y)
    mlflow.sklearn.save_model(
        estimador,
        str(carpeta),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
        skops_trusted_types=config.SKOPS_TIPOS_CONFIABLES,
        signature=infer_signature(X, estimador.predict(X)),
        input_example=X.head(5),
    )
    if registro:
        contenido = registro_de(pregunta) if registro is True else registro
        (carpeta / "registro.json").write_text(json.dumps(contenido), encoding="utf-8")
    return estimador


@pytest.fixture(scope="module")
def modelos_exportados(tmp_path_factory):
    """Carpeta por pregunta con un modelo como lo deja ``exportar.py`` y su estimador original."""
    raiz = tmp_path_factory.mktemp("servicio")
    exportados = {}
    for pregunta in PREGUNTAS:
        X, y = tabla(pregunta)
        exportados[pregunta] = (raiz / pregunta, guardar_modelo(raiz / pregunta, pregunta, X, y))
    return exportados


def cargador_de(carpetas: dict):
    return lambda pregunta: cargar_modelo(
        carpetas[pregunta], columnas=esquemas.COLUMNAS[pregunta], pregunta=pregunta
    )


@pytest.fixture
def cliente(modelos_exportados):
    carpetas = {p: carpeta for p, (carpeta, _) in modelos_exportados.items()}
    with TestClient(crear_app(cargador_de(carpetas))) as cliente:  # el with corre el arranque
        yield cliente


def filas_del_ejemplo(pregunta: str) -> pd.DataFrame:
    columnas = esquemas.COLUMNAS[pregunta]
    filas = [{c: fila[c] for c in columnas} for fila in EJEMPLOS[pregunta]["instancias"]]
    return pd.DataFrame(filas).astype(TIPOS[pregunta])


# =============================================================================
# Endpoints
# =============================================================================
def test_salud_reporta_los_tres_modelos(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert [
        (m["pregunta"], m["estado"], m["nombre"], m["version"]) for m in cuerpo["modelos"]
    ] == [(p, "ok", NOMBRES[p], VERSION) for p in PREGUNTAS]


def test_modelo_reporta_variables_y_linaje(cliente):
    cuerpos = {m["pregunta"]: m for m in cliente.get("/modelo").json()}
    assert set(cuerpos) == set(PREGUNTAS)
    for pregunta, cuerpo in cuerpos.items():
        assert cuerpo["variables"] == esquemas.COLUMNAS[pregunta]
        assert cuerpo["run_id"] == f"run-de-prueba-{pregunta}"
        assert cuerpo["etiquetas"]["data_version"] == "abc123def456"


def test_inicio_lleva_a_la_documentacion(cliente):
    respuesta = cliente.get("/", follow_redirects=False)
    assert respuesta.status_code in (302, 307)
    assert respuesta.headers["location"] == "/docs"
    rutas = cliente.get("/openapi.json").json()["paths"]
    assert {f"/predicciones/{p}" for p in PREGUNTAS} <= set(rutas)


@pytest.mark.parametrize("pregunta", PREGUNTAS)
def test_respuesta_indica_el_modelo_y_la_version(cliente, pregunta):
    respuesta = cliente.post(f"/predicciones/{pregunta}", json=EJEMPLOS[pregunta])
    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == {
        "nombre": NOMBRES[pregunta],
        "version": VERSION,
        "alias": "champion",
        "run_id": f"run-de-prueba-{pregunta}",
    }
    assert cuerpo["n"] == len(EJEMPLOS[pregunta]["instancias"]) == len(cuerpo["predicciones"])


def test_p1_predice_lo_mismo_que_el_modelo_original(cliente, modelos_exportados):
    _, estimador = modelos_exportados["p1"]
    cuerpo = cliente.post("/predicciones/p1", json=esquemas.EJEMPLO_PETICION_P1).json()
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == [101, 202]
    esperado = np.clip(estimador.predict(filas_del_ejemplo("p1")), 0, None)
    obtenido = [p["salidas_esperadas"] for p in cuerpo["predicciones"]]
    assert obtenido == pytest.approx(esperado, abs=1e-3)
    assert all(valor >= 0 for valor in obtenido)


def test_p2_predice_lo_mismo_que_el_modelo_original(cliente, modelos_exportados):
    _, estimador = modelos_exportados["p2"]
    cuerpo = cliente.post("/predicciones/p2", json=esquemas.EJEMPLO_PETICION_P2).json()
    esperado = estimador.predict(filas_del_ejemplo("p2")).tolist()
    assert [p["clase"] for p in cuerpo["predicciones"]] == esperado
    assert [p["riesgo"] for p in cuerpo["predicciones"]] == [
        esquemas.CLASES_P2[c] for c in esperado
    ]


def test_p3_predice_lo_mismo_que_el_modelo_original(cliente, modelos_exportados):
    _, estimador = modelos_exportados["p3"]
    cuerpo = cliente.post("/predicciones/p3", json=esquemas.EJEMPLO_PETICION_P3).json()
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == [2, 3]
    esperado = estimador.predict(filas_del_ejemplo("p3")).tolist()
    assert [p["grupo"] for p in cuerpo["predicciones"]] == esperado


@pytest.mark.parametrize(
    ("pregunta", "cambios", "campo"),
    [
        ("p1", {"hora_dia": 24}, "hora_dia"),
        ("p1", {"es_festivo": "no"}, "es_festivo"),
        ("p1", {"latitud": 19.43}, "latitud"),
        ("p1", {"hora": 8}, "hora"),
        ("p2", {"bloque_dia": 4}, "bloque_dia"),
        ("p2", {"frac_llenado_mismo_bloque_7d": 1.5}, "frac_llenado_mismo_bloque_7d"),
        ("p2", {"capacidad": "grande"}, "capacidad"),
        ("p3", {"balance_neto": 2.0}, "balance_neto"),
        ("p3", {"sal_lab_h08": -0.1}, "sal_lab_h08"),
        ("p3", {"sal_lab_h08": "alto"}, "sal_lab_h08"),
    ],
)
def test_peticion_invalida_da_422_con_el_campo(cliente, pregunta, cambios, campo):
    peticion = copy.deepcopy(EJEMPLOS[pregunta])
    peticion["instancias"][1].update(cambios)
    respuesta = cliente.post(f"/predicciones/{pregunta}", json=peticion)
    assert respuesta.status_code == 422
    ubicaciones = [detalle["loc"] for detalle in respuesta.json()["detail"]]
    assert ["body", "instancias", 1, campo] in ubicaciones


@pytest.mark.parametrize("pregunta", PREGUNTAS)
@pytest.mark.parametrize("cuerpo", [{}, {"instancias": []}, {"instancias": "fila"}])
def test_cuerpo_mal_formado_da_422(cliente, pregunta, cuerpo):
    assert cliente.post(f"/predicciones/{pregunta}", json=cuerpo).status_code == 422


def test_texto_que_no_es_json_da_422(cliente):
    respuesta = cliente.post(
        "/predicciones/p2", content="no es json", headers={"Content-Type": "application/json"}
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


def cliente_con(pregunta: str, valor: float) -> TestClient:
    """API donde el modelo de ``pregunta`` predice siempre ``valor`` (los otros, 1.0 o clase 1)."""

    def cargador(nombre: str) -> ModeloServido:
        modelo = ModeloConstante(valor if nombre == pregunta else 1)
        return ModeloServido(
            modelo, Identidad(NOMBRES[nombre], "1"), TIPOS[nombre], "prueba", pregunta=nombre
        )

    return TestClient(crear_app(cargador))


def test_p1_prediccion_negativa_se_recorta_a_cero():
    with cliente_con("p1", -3.0) as cliente:
        cuerpo = cliente.post("/predicciones/p1", json=EJEMPLOS["p1"]).json()
    assert [p["salidas_esperadas"] for p in cuerpo["predicciones"]] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("pregunta", "valor", "mensaje"),
    [
        ("p1", float("nan"), "no finitos"),
        ("p2", float("nan"), "no finitos"),
        ("p2", 5, "fuera de"),  # P2 solo conoce las clases 0, 1 y 2
        ("p2", 0.5, "enteros"),
        ("p3", -1, "enteros no negativos"),
        ("p3", float("inf"), "no finitos"),
    ],
)
def test_salida_invalida_del_modelo_da_500(pregunta, valor, mensaje):
    with cliente_con(pregunta, valor) as cliente:
        respuesta = cliente.post(f"/predicciones/{pregunta}", json=EJEMPLOS[pregunta])
    assert respuesta.status_code == 500
    assert mensaje in respuesta.json()["detail"]


# =============================================================================
# Carga de los modelos
# =============================================================================
def test_lee_la_carpeta_de_modelo_uri_de_cada_pregunta(modelos_exportados, monkeypatch):
    for pregunta, (carpeta, _) in modelos_exportados.items():
        monkeypatch.setenv(f"MODELO_URI_{pregunta.upper()}", str(carpeta))
        servido = cargar_modelo(columnas=esquemas.COLUMNAS[pregunta], pregunta=pregunta)
        assert (servido.identidad.nombre, servido.identidad.version) == (
            NOMBRES[pregunta],
            VERSION,
        )
        assert servido.pregunta == pregunta
    # La firma de MLflow guarda los enteros de 8 y 32 bits como "integer" (int32)
    assert {c: str(t) for c, t in servido.tipos.items()} == {
        c: str(t) for c, t in TIPOS["p3"].items()
    }


def test_un_modelo_que_no_carga_da_503_y_los_demas_siguen(modelos_exportados, tmp_path):
    carpetas = {p: carpeta for p, (carpeta, _) in modelos_exportados.items()}
    carpetas["p2"] = tmp_path / "vacia"  # no hay modelo de MLflow aqui
    with TestClient(crear_app(cargador_de(carpetas))) as cliente:
        respuesta = cliente.post("/predicciones/p2", json=EJEMPLOS["p2"])
        assert respuesta.status_code == 503
        assert "No hay un modelo" in respuesta.json()["detail"]

        salud = cliente.get("/salud")
        assert salud.status_code == 503
        cuerpo = salud.json()
        assert cuerpo["estado"] == "degradado"
        assert [(m["pregunta"], m["estado"]) for m in cuerpo["modelos"]] == [
            ("p1", "ok"),
            ("p2", "error"),
            ("p3", "ok"),
        ]
        assert "No hay un modelo" in cuerpo["modelos"][1]["error"]

        assert [m["pregunta"] for m in cliente.get("/modelo").json()] == ["p1", "p3"]
        assert cliente.post("/predicciones/p1", json=EJEMPLOS["p1"]).status_code == 200
        assert cliente.post("/predicciones/p3", json=EJEMPLOS["p3"]).status_code == 200


def test_modelo_sin_registro_json_no_se_carga(tmp_path):
    X, y = tabla("p1")
    guardar_modelo(tmp_path / "p1", "p1", X, y, registro=None)
    with pytest.raises(ErrorModelo, match=r"registro\.json.*--pregunta p1"):
        cargar_modelo(tmp_path / "p1", pregunta="p1")


def test_modelo_con_otras_variables_no_se_carga(tmp_path):
    X, y = tabla("p2")
    guardar_modelo(tmp_path / "p2", "p2", X.drop(columns=["mes"]), y)
    with pytest.raises(ErrorModelo, match="mes"):
        cargar_modelo(tmp_path / "p2", columnas=esquemas.COLUMNAS_P2, pregunta="p2")


def test_carga_directa_del_model_registry_por_alias(mlflow_temporal):
    """MODELO_URI_P1=models:/<nombre>@champion resuelve la version exacta del registro."""
    X, y = tabla("p1")
    estimador = estimador_de("p1").fit(X, y)
    with mlflow.start_run() as run:
        uri = comun.registrar_modelo(estimador, X)
    version = mlflow.register_model(uri, NOMBRE_P1).version
    mlflow.MlflowClient().set_registered_model_alias(NOMBRE_P1, "champion", version)

    servido = cargar_modelo(f"models:/{NOMBRE_P1}@champion", columnas=esquemas.COLUMNAS_P1)
    assert servido.identidad.version == str(version)
    assert servido.identidad.alias == "champion"
    assert servido.identidad.run_id == run.info.run_id
