"""Pruebas de la base del experimento: catalogo, particiones, registro en MLflow, busquedas y
MLflow Project."""

import json

import mlflow
from mlflow.projects import _project_spec
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from typer.testing import CliRunner

from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import PARTICIONES
from tc5061_proyecto_equipo18.modeling import busqueda, comun, train

META = {
    "data_version": "abc123def456",
    "parametros": {"mes_inicio": "2024-10", "p1_rezagos_h": [24, 48]},
}


def artefactos(run_id: str, carpeta: str) -> set[str]:
    return {a.path for a in mlflow.MlflowClient().list_artifacts(run_id, carpeta)}


# =============================================================================
# Catalogo de modelos e hiperparametros
# =============================================================================
def test_catalogo_es_coherente():
    for pregunta, catalogo in config.MODELOS.items():
        assert sum(bool(spec.get("baseline")) for spec in catalogo.values()) == 1
        assert len(catalogo) >= 4, "baseline y al menos tres candidatos por pregunta"
        for spec in catalogo.values():
            assert spec["familia"] and isinstance(spec["hiperparametros"], dict)
        busqueda.validar_busqueda(pregunta)  # la rejilla solo usa hiperparametros del modelo
        metrica, sentido = config.METRICAS_SELECCION[pregunta]
        assert sentido in ("min", "max")
        assert metrica.split("_")[0] in PARTICIONES
    assert set(config.MODELOS_REGISTRADOS) == set(config.MODELOS)
    assert set(config.DESCRIPCIONES_REGISTRO) == set(config.MODELOS)


def test_normalizar_pregunta():
    assert comun.normalizar_pregunta("p2") == "P2"
    assert comun.baseline_de("p3") == "cuartiles_volumen"
    with pytest.raises(ValueError):
        comun.normalizar_pregunta("p4")


def test_resolver_hiperparametros_convierte_tipos_y_rechaza_desconocidos():
    hp = comun.resolver_hiperparametros(
        "p1",
        "hgb_poisson",
        {"learning_rate": "0.05", "max_leaf_nodes": "31", "early_stopping": "false"},
    )
    assert hp["learning_rate"] == 0.05 and isinstance(hp["learning_rate"], float)
    assert hp["max_leaf_nodes"] == 31 and isinstance(hp["max_leaf_nodes"], int)
    assert hp["early_stopping"] is False
    assert hp["loss"] == "poisson"  # lo que no se cambia sale de config.py
    assert comun.resolver_hiperparametros("p1", "hgb_poisson") == config.P1_HIPERPARAMETROS
    with pytest.raises(ValueError, match="learnig_rate"):
        comun.resolver_hiperparametros("p1", "hgb_poisson", {"learnig_rate": "0.1"})
    with pytest.raises(ValueError, match="tipo int"):
        comun.resolver_hiperparametros("p1", "hgb_poisson", {"max_leaf_nodes": "31.5"})
    with pytest.raises(ValueError, match="Modelo desconocido"):
        comun.resolver_hiperparametros("p1", "rf")
    with pytest.raises(ValueError, match="ninguno"):
        comun.resolver_hiperparametros("p1", "rezago_semanal", {"alpha": "1"})


def test_leer_opciones_extra():
    argumentos = ["--learning_rate", "0.05", "--max-leaf-nodes=31", "--alpha", "-1"]
    assert comun.leer_opciones_extra(argumentos) == {
        "learning_rate": "0.05",
        "max_leaf_nodes": "31",
        "alpha": "-1",
    }
    with pytest.raises(ValueError):
        comun.leer_opciones_extra(["0.05"])
    with pytest.raises(ValueError):
        comun.leer_opciones_extra(["--learning_rate"])


def test_nombre_y_comando_de_reproduccion():
    hp = comun.resolver_hiperparametros("p1", "hgb_poisson", {"learning_rate": "0.05"})
    assert comun.nombre_run("P1", "candidato", "hgb_poisson", hp) == (
        "p1-candidato-hgb_poisson-learning_rate0.05"
    )
    assert comun.nombre_run("P1", "baseline", "rezago_semanal", {}) == "p1-baseline-rezago_semanal"
    assert comun.comando_reproduccion("P1", "hgb_poisson", hp) == (
        "mlflow run . -e train -P pregunta=p1 -P modelo=hgb_poisson -P learning_rate=0.05 "
        "--env-manager local"
    )


# =============================================================================
# Datos
# =============================================================================
def test_separar_particiones_por_fecha_y_por_periodo():
    fechas = pd.to_datetime([config.FIN_ENTRENAMIENTO, config.FIN_VALIDACION, "2026-09-30"])
    partes = comun.separar_particiones(pd.DataFrame({"hora": fechas}), "hora")
    assert [len(partes[p]) for p in ("entrenamiento", "validacion", "prueba")] == [1, 1, 1]

    perfiles = pd.DataFrame({"periodo": ["entrenamiento", "entrenamiento", "prueba"]})
    partes = comun.separar_particiones(perfiles, columna_tiempo=None)
    assert [len(partes[p]) for p in ("entrenamiento", "validacion", "prueba")] == [2, 0, 1]


def test_muestrear_es_reproducible():
    df = pd.DataFrame({"x": range(100)})
    assert comun.muestrear(df, 10).equals(comun.muestrear(df, 10))
    assert comun.muestrear(df, 1000) is df


# =============================================================================
# Runs: etiquetas, parametros, artefactos, conjuntos de datos y modelo
# =============================================================================
def test_run_registra_etiquetas_parametros_artefactos_y_modelo(mlflow_temporal):
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.ones(20)})
    modelo = DummyRegressor().fit(X, np.arange(20.0))
    hp = comun.resolver_hiperparametros("P1", "hgb_poisson", {"learning_rate": "0.05"})

    with comun.iniciar_run(
        "P1", "hgb_poisson", hiperparametros=hp, meta_features=META, parametros={"n_variables": 2}
    ) as run:
        comun.registrar_metricas({"mae": 1.5}, "validacion")
        comun.registrar_conjuntos(
            {p: X.assign(y=1.0) for p in PARTICIONES},
            "P1",
            config.DATASET_P1,
            ["a", "b"],
            "y",
            META["data_version"],
        )
        uri = comun.registrar_modelo(modelo, X)

    datos = mlflow.get_run(run.info.run_id)
    tags, params = datos.data.tags, datos.data.params
    assert run.info.run_name == "p1-candidato-hgb_poisson-learning_rate0.05"
    assert tags["pregunta"] == "P1" and tags["etapa"] == "candidato"
    assert tags["familia_modelo"] == "gradient_boosting" and tags["modelo"] == "hgb_poisson"
    assert tags["data_version"] == "abc123def456" and "git_commit" in tags
    assert tags["mlflow.note.content"].startswith("Reproducir: mlflow run . -e train")
    assert params["learning_rate"] == "0.05" and params["loss"] == "poisson"
    assert params["semilla"] == str(config.SEMILLA)
    assert params["features.mes_inicio"] == "2024-10"
    assert params["data_version"] == "abc123def456" and params["modelo"] == "hgb_poisson"
    assert params["pregunta"] == "p1" and params["n_variables"] == "2"
    assert datos.data.metrics["validacion_mae"] == 1.5
    assert datos.data.metrics["costo_tamano_modelo_mb"] > 0

    assert {
        "config/corrida.json",
        "config/MLproject",
        "config/python_env.yaml",
        "config/requirements.txt",
        "config/config.py",
    } <= artefactos(run.info.run_id, "config")
    assert {"datos/features_meta.json", "datos/manifiesto_datos.json"} <= artefactos(
        run.info.run_id, "datos"
    )
    corrida = json.loads(
        mlflow.artifacts.load_text(f"{datos.info.artifact_uri}/config/corrida.json")
    )
    assert corrida["hiperparametros"]["learning_rate"] == 0.05
    assert corrida["comando_reproduccion"].endswith("-P learning_rate=0.05 --env-manager local")

    conjuntos = {d.dataset.name: d for d in datos.inputs.dataset_inputs}
    assert set(conjuntos) == {f"p1-{p}-abc123def456" for p in PARTICIONES}
    fuente = json.loads(conjuntos["p1-entrenamiento-abc123def456"].dataset.source)
    assert fuente["uri"] == "data/processed/p1_demanda_estacion_hora.parquet"  # ruta relativa

    cargado = mlflow.sklearn.load_model(uri)
    assert cargado.predict(X).shape == (20,)
    assert mlflow.models.get_model_info(uri).signature is not None


def test_registrar_parametros_respeta_los_de_mlflow_run(mlflow_temporal, monkeypatch):
    """Simula `mlflow run`: el proyecto crea el run y registra los -P antes de llamar al codigo."""
    cliente = mlflow.MlflowClient()
    experimento = mlflow.get_experiment_by_name("prueba").experiment_id
    proyecto = cliente.create_run(experimento)
    for clave, valor in {
        "pregunta": "p1",
        "modelo": "hgb_poisson",
        "learning_rate": ".05",
    }.items():
        cliente.log_param(proyecto.info.run_id, clave, valor)
    monkeypatch.setenv("MLFLOW_RUN_ID", proyecto.info.run_id)

    hp = comun.resolver_hiperparametros("p1", "hgb_poisson", {"learning_rate": ".05"})
    with comun.iniciar_run("P1", "hgb_poisson", hiperparametros=hp, meta_features=META) as run:
        assert run.info.run_id == proyecto.info.run_id  # retoma el run del proyecto
    params = mlflow.get_run(proyecto.info.run_id).data.params
    assert params["learning_rate"] == ".05"  # se conserva el texto que registro mlflow run
    assert params["max_leaf_nodes"] == str(config.P1_HIPERPARAMETROS["max_leaf_nodes"])
    assert mlflow.get_run(proyecto.info.run_id).info.run_name == (
        "p1-candidato-hgb_poisson-learning_rate0.05"
    )


def test_configurar_mlflow_detecta_run_de_otro_experimento(mlflow_temporal, monkeypatch, tmp_path):
    cliente = mlflow.MlflowClient()
    run = cliente.create_run(cliente.create_experiment("otro"))
    monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)
    with pytest.raises(RuntimeError, match="MLFLOW_EXPERIMENT_NAME"):
        comun.configurar_mlflow(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            experimento="prueba",
            artefactos_dir=tmp_path / "mlruns",
        )


def test_configurar_mlflow_sin_servidor_da_un_mensaje_claro():
    with pytest.raises(RuntimeError, match="mlflow server"):
        comun.configurar_mlflow(tracking_uri="http://127.0.0.1:9")  # nadie escucha ahi


def test_validaciones_de_run(mlflow_temporal):
    with pytest.raises(ValueError), comun.iniciar_run("P4", "x", meta_features=META):
        pass
    with (
        pytest.raises(ValueError),
        comun.iniciar_run("P1", "hgb_poisson", "prod", meta_features=META),
    ):
        pass
    with (
        comun.iniciar_run("P2", "hgb_balanceado", meta_features=META),
        pytest.raises(ValueError),
    ):
        comun.registrar_metricas({"f1": 0.5}, "test")


# =============================================================================
# train.py: un run por llamada, hiperparametros extra por linea de comandos
# =============================================================================
def test_entrenar_resuelve_hiperparametros_y_llama_al_modulo(monkeypatch):
    llamadas = []

    class ModuloFalso:
        @staticmethod
        def correr(modelo, hiperparametros, datos=None, meta_features=None):
            llamadas.append((modelo, hiperparametros))
            return "run-1"

    monkeypatch.setattr(train, "modulo_de", lambda pregunta: ModuloFalso)
    assert train.entrenar("p1", "hgb_poisson", {"max_iter": "50"}) == "run-1"
    assert llamadas[0][0] == "hgb_poisson" and llamadas[0][1]["max_iter"] == 50


def test_cli_de_train_acepta_hiperparametros_extra(monkeypatch):
    capturado = {}
    monkeypatch.setattr(train, "configurar_mlflow", lambda: "0")
    monkeypatch.setattr(
        train, "entrenar", lambda p, m, c: capturado.update(pregunta=p, modelo=m, cambios=c)
    )
    resultado = CliRunner().invoke(
        train.app, ["--pregunta", "p1", "--modelo", "hgb_poisson", "--learning_rate", "0.05"]
    )
    assert resultado.exit_code == 0, resultado.output
    assert capturado == {
        "pregunta": "p1",
        "modelo": "hgb_poisson",
        "cambios": {"learning_rate": "0.05"},
    }
    # Un hiperparametro mal escrito falla antes de tocar MLflow
    resultado = CliRunner().invoke(
        train.app, ["--pregunta", "p1", "--modelo", "hgb_poisson", "--learnig_rate", "0.05"]
    )
    assert resultado.exit_code != 0


# =============================================================================
# busqueda.py: una corrida padre y una hija por combinacion
# =============================================================================
def test_combinaciones_y_nombres_de_hijas():
    assert busqueda.combinaciones({"a": [1, 2], "b": [0.5]}) == [
        {"a": 1, "b": 0.5},
        {"a": 2, "b": 0.5},
    ]
    assert busqueda.nombre_hijo("P2", "hgb_ponderado", {"peso_riesgo": 10.0}) == (
        "p2-busqueda-hgb_ponderado-peso_riesgo10"
    )


def test_busqueda_crea_padre_e_hijas_y_elige_la_mejor(mlflow_temporal, monkeypatch):
    class ModuloFalso:
        @staticmethod
        def cargar_datos():
            raise AssertionError("la busqueda recibe los datos ya leidos")

        @staticmethod
        def correr(modelo, hiperparametros, datos=None, meta_features=None, etapa=None,
                   anidado=False, nombre=None):  # fmt: skip
            with comun.iniciar_run(
                "P1", modelo, etapa, hiperparametros, meta_features, nombre, anidado
            ) as run:
                error = abs(hiperparametros["learning_rate"] - 0.1)
                comun.registrar_metricas({"mae": error}, "validacion")
                comun.registrar_metricas({"mae": error + 0.01}, "prueba")
            return run.info.run_id

    monkeypatch.setattr(busqueda, "modulo_de", lambda pregunta: ModuloFalso)
    padre_id = busqueda.ejecutar_busqueda(
        "p1", datos={}, meta_features=META, rejilla={"learning_rate": [0.05, 0.1, 0.2]}
    )
    padre = mlflow.get_run(padre_id)
    hijas = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{padre_id}'", output_format="list"
    )
    assert len(hijas) == 3
    assert {h.data.tags["etapa"] for h in hijas} == {"busqueda"}
    assert padre.info.run_name == "p1-busqueda-hgb_poisson"
    assert padre.data.tags["etapa"] == "busqueda"
    assert padre.data.params["mejor_learning_rate"] == "0.1"
    assert padre.data.params["n_combinaciones"] == "3"
    assert padre.data.metrics["mejor_validacion_mae"] == pytest.approx(0)
    assert padre.data.tags["mejor_run_name"] == "p1-busqueda-hgb_poisson-learning_rate0.1"
    assert "tablas/resultados_busqueda.csv" in artefactos(padre_id, "tablas")
    assert "figuras/busqueda.png" in artefactos(padre_id, "figuras")


def test_hijas_de_la_busqueda_heredan_que_las_lanzo_mlflow_run(mlflow_temporal, monkeypatch):
    """Simula `mlflow run . -e search`: el proyecto crea el padre y las hijas se abren adentro."""
    cliente = mlflow.MlflowClient()
    experimento = mlflow.get_experiment_by_name("prueba").experiment_id
    origen = {"mlflow.source.type": "PROJECT", "mlflow.project.entryPoint": "search"}
    proyecto = cliente.create_run(experimento, tags=origen)
    monkeypatch.setenv("MLFLOW_RUN_ID", proyecto.info.run_id)

    hp = comun.resolver_hiperparametros("p1", "hgb_poisson")
    with (
        comun.iniciar_run("P1", "hgb_poisson", "busqueda", meta_features=META) as padre,
        comun.iniciar_run("P1", "hgb_poisson", "busqueda", hp, META, anidado=True) as hija,
    ):
        pass
    assert padre.info.run_id == proyecto.info.run_id
    for run_id in (padre.info.run_id, hija.info.run_id):
        etiquetas = mlflow.get_run(run_id).data.tags
        assert {clave: etiquetas[clave] for clave in origen} == origen
        assert comun.lanzado_por_proyecto(run_id)
        corrida = mlflow.artifacts.load_dict(
            f"{mlflow.get_run(run_id).info.artifact_uri}/config/corrida.json"
        )
        assert corrida["lanzado_por_mlflow_project"] is True

    # Fuera del proyecto (p. ej. en estas pruebas) el run queda como LOCAL y no se marca
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    with comun.iniciar_run("P1", "hgb_poisson", meta_features=META) as local:
        pass
    assert not comun.lanzado_por_proyecto(local.info.run_id)


# =============================================================================
# MLflow Project
# =============================================================================
def test_mlproject_declara_entorno_y_puntos_de_entrada_parametrizados():
    proyecto = _project_spec.load_project(str(config.PROJ_ROOT))
    assert proyecto.env_type == "python_env"
    assert proyecto.env_config_path.endswith("python_env.yaml")
    for nombre in ("ingest", "features", "train", "search", "register"):
        assert proyecto.get_entry_point(nombre) is not None

    entrenamiento = proyecto.get_entry_point("train")
    comando = entrenamiento.compute_command(
        {"pregunta": "p1", "modelo": "hgb_poisson", "learning_rate": "0.05"}, storage_dir=None
    )
    assert comando == (
        "python -m tc5061_proyecto_equipo18.modeling.train --pregunta p1 --modelo hgb_poisson "
        "--learning_rate 0.05"
    )
    with pytest.raises(Exception, match="modelo"):
        entrenamiento.compute_command({"pregunta": "p1"}, storage_dir=None)

    # Rutas, fechas y semillas viven solo en config.py (se revisa el YAML sin comentarios)
    lineas = (config.PROJ_ROOT / "MLproject").read_text(encoding="utf-8").splitlines()
    yaml = "\n".join(linea for linea in lineas if not linea.lstrip().startswith("#"))
    for valor in (config.MES_INICIO, config.FIN_ENTRENAMIENTO, "data/", "semilla", "default"):
        assert valor not in yaml
