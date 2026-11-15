"""Pruebas del A/B testing de P1: bootstrap pareado, regla de decision, fuga y flujo completo
contra un Model Registry temporal (sin servidor ni datos reales)."""

from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from monitoreo import sintetico
from tc5061_proyecto_equipo18 import config
from tc5061_proyecto_equipo18.features import COLUMNAS_P1, OBJETIVO_P1
from tc5061_proyecto_equipo18.modeling import ab_test, comun, registro

NOMBRE_P1 = config.MODELOS_REGISTRADOS["P1"]


@pytest.fixture(scope="module")
def p1():
    """P1 sintetico de oct 2024 a sep 2026 con estaciones conocidas y nuevas (un dia de cada 4)."""
    return sintetico.p1_sintetico(conocidas=10, nuevas=4, cada_dias=4)


def variante(nombre: str, version: str, tags: dict | None = None) -> ab_test.Variante:
    """Variante falsa para las pruebas que no necesitan el registro."""
    info = SimpleNamespace(version=version, tags=tags or {})
    return ab_test.Variante(nombre, "alias", info, None, ab_test.fin_entrenamiento_de(info))


def por_dia(error_a, error_b, filas=10) -> pd.DataFrame:
    """Errores absolutos sumados por dia, con el mismo numero de filas cada dia."""
    return pd.DataFrame(
        {
            "error_a": np.asarray(error_a, float),
            "error_b": np.asarray(error_b, float),
            "filas": filas,
        }
    )


def metricas_conocidas(mae_a: float, mae_b: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"grupo": "conocidas", "version": "A", "mae": mae_a},
            {"grupo": "conocidas", "version": "B", "mae": mae_b},
        ]
    )


# =============================================================================
# Bootstrap pareado y regla de decision
# =============================================================================
def test_bootstrap_de_errores_iguales_da_diferencia_cero():
    resultado = ab_test.bootstrap_pareado(por_dia([5, 7, 9, 4], [5, 7, 9, 4]), remuestreos=500)
    assert resultado["diferencia_mae"] == 0
    assert resultado["ic_inferior"] == 0 and resultado["ic_superior"] == 0


def test_bootstrap_detecta_b_mejor_todos_los_dias():
    resultado = ab_test.bootstrap_pareado(por_dia([8, 9, 10, 11], [6, 7, 8, 9]), remuestreos=500)
    assert resultado["diferencia_mae"] == pytest.approx(0.2)  # 2 de error por dia / 10 filas
    assert resultado["ic_inferior"] > 0 and resultado["fraccion_b_mejor"] == 1.0


def test_bootstrap_es_reproducible_y_pareado_por_dia():
    datos = por_dia([10, 2, 9, 3, 8], [9, 3, 8, 2, 9])
    assert ab_test.bootstrap_pareado(datos, 300) == ab_test.bootstrap_pareado(datos, 300)
    # Un dia bueno para B y otro malo: el intervalo debe cruzar el 0
    resultado = ab_test.bootstrap_pareado(datos, 2000)
    assert resultado["ic_inferior"] < 0 < resultado["ic_superior"]


def test_bootstrap_necesita_al_menos_dos_dias():
    with pytest.raises(ValueError, match="al menos 2 dias"):
        ab_test.bootstrap_pareado(por_dia([1], [1]))


def test_decide_promover_solo_si_cumple_las_dos_condiciones():
    gana = {"ic_inferior": 0.001}
    decision = ab_test.decidir(gana, metricas_conocidas(0.80, 0.79))
    assert decision["promover_b"] and decision["sin_regresion_en_conocidas"]


def test_decide_mantener_si_el_intervalo_toca_el_cero():
    decision = ab_test.decidir({"ic_inferior": -0.0001}, metricas_conocidas(0.80, 0.79))
    assert not decision["promover_b"] and not decision["b_mejor_en_mae"]
    assert "incluye el 0" in decision["motivo"]


def test_decide_mantener_si_b_empeora_en_las_conocidas_mas_de_la_tolerancia():
    peor = 0.80 * (1 + config.AB_TOLERANCIA_CONOCIDAS) + 0.001
    decision = ab_test.decidir({"ic_inferior": 0.01}, metricas_conocidas(0.80, peor))
    assert decision["b_mejor_en_mae"] and not decision["sin_regresion_en_conocidas"]
    assert not decision["promover_b"] and "conocidas" in decision["motivo"]


# =============================================================================
# Fuga y conjunto de evaluacion
# =============================================================================
def test_periodo_de_entrenamiento_sale_de_la_etiqueta_del_reentrenado():
    assert variante("A", "1").fin_entrenamiento == config.FIN_ENTRENAMIENTO
    reentrenado = variante("B", "3", {"reentrenamiento_fin": config.REENTRENAMIENTO_FIN})
    assert reentrenado.fin_entrenamiento == config.REENTRENAMIENTO_FIN
    ab_test.verificar_sin_fuga([variante("A", "1"), reentrenado])


def test_rechaza_una_version_que_entreno_con_el_mes_de_evaluacion():
    con_fuga = variante("B", "3", {"reentrenamiento_fin": config.AB_EVALUACION[0]})
    with pytest.raises(ValueError, match="fuga"):
        ab_test.verificar_sin_fuga([variante("A", "1"), con_fuga])


def test_rechaza_comparar_una_version_consigo_misma():
    with pytest.raises(ValueError, match="misma version"):
        ab_test.verificar_sin_fuga([variante("A", "1"), variante("B", "1")])


def test_conjunto_de_evaluacion_es_solo_el_periodo_del_ab(p1):
    datos = ab_test.filas_de(p1, config.AB_EVALUACION)
    assert datos["hora"].min() >= pd.Timestamp(config.AB_EVALUACION[0])
    assert datos["hora"].max() < pd.Timestamp(config.AB_EVALUACION[1]) + pd.Timedelta(days=1)


def test_metricas_separan_estaciones_conocidas_y_nuevas(p1):
    datos = ab_test.filas_de(p1, config.AB_EVALUACION)
    conocidas = set(range(1, 11))
    real = datos[OBJETIVO_P1].to_numpy(float)
    tabla = ab_test.tabla_metricas(datos, {"A": real + 1, "B": real}, conocidas)
    assert set(tabla["grupo"]) == {"todas", "conocidas", "nuevas"}
    mae = tabla.set_index(["grupo", "version"])["mae"]
    assert mae["todas", "A"] == pytest.approx(1.0) and mae["todas", "B"] == 0
    filas = tabla.drop_duplicates("grupo").set_index("grupo")["filas"]
    assert filas["conocidas"] + filas["nuevas"] == filas["todas"]


# =============================================================================
# Flujo completo contra un registro temporal
# =============================================================================
def registrar(estimador, datos: pd.DataFrame, alias: str, tags: dict | None = None):
    """Entrena, guarda y registra un modelo de P1 con el alias pedido."""
    estimador.fit(datos[COLUMNAS_P1], datos[OBJETIVO_P1])
    with mlflow.start_run() as run:
        comun.registrar_modelo(estimador, datos[COLUMNAS_P1].head(50))
    nombre = registro.asegurar_modelo_registrado("P1")
    model_id = registro.model_id_de(mlflow.get_run(run.info.run_id))
    version = mlflow.register_model(f"models:/{model_id}", nombre, tags=tags)
    registro.asignar_alias(nombre, alias, version)
    return version


@pytest.fixture
def registro_ab(mlflow_temporal, p1):
    """A = constante (malo) como @champion; B = lineal sobre la media reciente como @challenger."""
    entrenamiento = p1[p1["hora"] <= config.FIN_ENTRENAMIENTO]
    reciente = p1[p1["hora"] <= config.REENTRENAMIENTO_FIN]
    a = registrar(DummyRegressor(strategy="constant", constant=4.0), entrenamiento, "champion")
    b = registrar(
        LinearRegression(),
        reciente,
        "challenger",
        {"reentrenamiento_fin": config.REENTRENAMIENTO_FIN, "version_origen": a.version},
    )
    return a, b


def aliases() -> dict[str, str]:
    return mlflow.MlflowClient().get_registered_model(NOMBRE_P1).aliases


def test_ab_completo_escribe_salidas_y_no_mueve_aliases(registro_ab, p1, tmp_path):
    a, b = registro_ab
    resultado = ab_test.ejecutar(
        p1, carpeta=tmp_path / "ab", figuras=tmp_path / "fig", data_version="abc123def456"
    )
    assert resultado["decision"]["promover_b"]  # B es mucho mejor que una constante
    assert aliases() == {"champion": a.version, "challenger": b.version}  # sin --promover
    for archivo in ("ab/ab_test.md", "ab/ab_test.json", "fig/ab_mae_mensual.png"):
        assert (tmp_path / archivo).stat().st_size > 0
    texto = (tmp_path / "ab" / "ab_test.md").read_text(encoding="utf-8")
    assert "B gana" in texto and texto.isascii()
    meses = [m["mes"] for m in resultado["mae_mensual_champion"]]
    assert meses[0] == "2026-04" and meses[-1] == "2026-09"
    run = mlflow.get_run(resultado["run_id"])
    assert run.data.tags["etapa"] == "ab_test" and run.data.tags["decision"] == "promover_b"
    assert run.data.metrics["ic_inferior"] > 0


def test_promover_mueve_aliases_solo_si_la_regla_lo_indica(registro_ab, p1, tmp_path):
    a, b = registro_ab
    ab_test.ejecutar(p1, promover=True, carpeta=tmp_path, figuras=tmp_path, data_version="x")
    assert aliases() == {"champion": b.version, "challenger": a.version}
    # Ahora el champion es el bueno: la regla no promueve y los aliases no cambian
    resultado = ab_test.ejecutar(
        p1, promover=True, carpeta=tmp_path, figuras=tmp_path, data_version="x"
    )
    assert not resultado["decision"]["promover_b"]
    assert aliases() == {"champion": b.version, "challenger": a.version}


def test_falla_sin_challenger(mlflow_temporal, p1):
    registrar(DummyRegressor(), p1.head(200), "champion")
    with pytest.raises(RuntimeError, match="-e retrain"):
        ab_test.ejecutar(p1, data_version="x")
