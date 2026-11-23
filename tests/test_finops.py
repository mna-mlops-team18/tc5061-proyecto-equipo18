"""Pruebas del analisis FinOps: precios, demanda, costo por inferencia y grafica."""

import json

import pytest

from carga import finops
from tc5061_proyecto_equipo18 import config

PRECIOS = config.FINOPS_PRECIOS


def resumen_de_prueba(modelos_de_prueba: bool = False) -> dict:
    """Lo minimo de ``resumen_carga.json``: capacidad de 1000 predicciones/s."""
    escalon = {
        "usuarios": 50,
        "rps": 50.0,
        "predicciones_por_seg": 1000.0,
        "p50_ms": 10.0,
        "p95_ms": 30.0,
        "p99_ms": 40.0,
        "p95_lote_ms": 80.0,
        "tasa_error": 0.0,
        "cpu_api_pct": 20.0,
        "memoria_api_max_mib": 300.0,
        "cumple_slo": True,
    }
    saturado = {**escalon, "usuarios": 100, "p95_ms": 900.0, "cpu_api_pct": 99.0}
    saturado["cumple_slo"] = False
    endpoints = [
        {"endpoint": "/predicciones/p2", "peticiones": 900, "filas_por_peticion": 20.0},
        {"endpoint": "/predicciones/p1 [lote]", "peticiones": 3, "filas_por_peticion": 1000.0},
        {"endpoint": "/predicciones/p3 [lote]", "peticiones": 1, "filas_por_peticion": 470.0},
        {"endpoint": "/salud", "peticiones": 100, "filas_por_peticion": 0.0},
    ]
    tiempos = {"/predicciones/p2": 15.0, "/predicciones/p1 [lote]": 60.0}
    for e in endpoints:
        e.update(p50_ms=tiempos.get(e["endpoint"], 250.0), p95_ms=1.0, p99_ms=1.0, tasa_error=0)
    return {
        "duracion_seg": 100.0,
        "peticiones": 1004,
        "predicciones": 21470,
        "errores": 0,
        "tasa_error": 0.0,
        "p50_ms": 10.0,
        "p95_ms": 30.0,
        "p99_ms": 40.0,
        "slo": {"p95_ms": 500, "tasa_error": 0.01},
        "capacidad": {
            "usuarios": 50,
            "rps": 50.0,
            "predicciones_por_seg": 1000.0,
            "p95_ms": 30.0,
            "tasa_error": 0.0,
            "cpu_api_pct": 20.0,
        },
        "por_endpoint": endpoints,
        "por_endpoint_sin_saturar": endpoints,
        "escalones": [escalon, saturado],
        "recursos": {
            "mibici-api": {
                "muestras": 50,
                "cpu_promedio_pct": 40.0,
                "cpu_max_pct": 99.0,
                "memoria_max_mib": 300.0,
                "limite_memoria_mib": 1024.0,
            }
        },
        "entorno": {"modelos_de_prueba": modelos_de_prueba, "escalones": [[50, 40], [100, 40]]},
        "figuras": {},
    }


def test_precio_por_segundo_de_las_referencias():
    assert finops.precio_por_seg(PRECIOS["cloud_run_instancia"]) == pytest.approx(0.00002)
    # Fargate: 1 vCPU y 2 GB por hora
    assert finops.precio_por_seg(PRECIOS["fargate"]) * 3600 == pytest.approx(
        0.04048 + 2 * 0.004445
    )


def test_demanda_mensual_segun_la_estrategia_de_despliegue():
    demanda = finops.demanda_mensual(estaciones=470, despachadores=10, horas_operacion=16, dias=30)
    assert demanda["p1"] == {
        "peticiones": 12 * 30,
        "predicciones": 470 * 24 * 30,
        "modo": "lote nocturno",
    }
    assert demanda["p2"]["peticiones"] == 10 * 60 * 16 * 30
    assert demanda["p2"]["predicciones"] == 10 * 60 * 16 * 30 * config.CARGA_FILAS["p2"]
    assert demanda["total"]["predicciones"] == sum(
        demanda[p]["predicciones"] for p in ("p1", "p2", "p3")
    )
    assert demanda["pico_predicciones_por_seg"] == pytest.approx(
        10 / 60 * config.CARGA_FILAS["p2"]
    )


@pytest.mark.parametrize(
    ("ms", "segundos"), [(1, 0.1), (15, 0.1), (100, 0.1), (101, 0.2), (250, 0.3)]
)
def test_tiempo_facturable_redondea_a_100_ms(ms, segundos):
    assert finops.tiempo_facturable(ms) == pytest.approx(segundos)


def test_costos_por_hora_por_inferencia_y_al_mes():
    demanda = finops.demanda_mensual()
    costos = finops.costos(resumen_de_prueba(), demanda)
    segundos_mes = config.FINOPS_DIAS_MES * 24 * 3600

    encendida = costos["cloud_run_instancia"]
    assert encendida["usd_hora"] == pytest.approx(0.072)
    assert encendida["usd_mes"] == pytest.approx(0.00002 * segundos_mes, abs=0.01)
    # A plena capacidad (1000 predicciones/s): 0.00002 USD/s / 1000 * 1e6
    assert encendida["usd_millon_predicciones_capacidad"] == pytest.approx(0.02)
    assert (
        encendida["usd_millon_predicciones_demanda"]
        > encendida["usd_millon_predicciones_capacidad"]
    )  # con la demanda real hay tiempo ocioso

    # Cobro por peticion: P2 paga 0.1 s por peticion (15 ms redondeados), P1 0.1 s y P3 0.3 s
    por_seg = finops.precio_por_seg(PRECIOS["cloud_run_peticion"])
    esperado = (
        demanda["p2"]["peticiones"] * 0.1 * por_seg
        + demanda["p1"]["peticiones"] * 0.1 * por_seg
        + demanda["p3"]["peticiones"] * 0.3 * por_seg
        + demanda["total"]["peticiones"] / 1e6 * 0.40
    )
    assert costos["cloud_run_peticion"]["usd_mes"] == pytest.approx(esperado, abs=0.01)
    assert costos["cloud_run_peticion"]["usd_mes"] < encendida["usd_mes"]


def test_sin_capacidad_da_error_claro():
    resumen = resumen_de_prueba() | {"capacidad": None}
    with pytest.raises(ValueError, match="Ningun escalon"):
        finops.costos(resumen, finops.demanda_mensual())


def test_medidas_para_optimizar():
    analisis_ = finops.analizar(resumen_de_prueba())
    assert analisis_["ms_por_prediccion"]["/predicciones/p2"] == pytest.approx(0.75)
    assert analisis_["ms_por_prediccion"]["/predicciones/p1 [lote]"] == pytest.approx(0.06)
    assert "/salud" not in analisis_["ms_por_prediccion"]  # no da predicciones
    # Solo el escalon que cumple y no satura: 20 % de vCPU / 1000 predicciones/s
    assert analisis_["cpu_seg_por_prediccion"] == pytest.approx(0.0002)
    assert analisis_["cpu_saturada"] is True
    assert analisis_["memoria_sugerida_mib"] == 512
    assert analisis_["utilizacion_pico"] == pytest.approx(
        analisis_["demanda"]["pico_predicciones_por_seg"] / 1000, rel=1e-3
    )


@pytest.mark.parametrize(
    ("usada", "sugerida"), [(None, None), (100, 512), (400, 768), (900, 1536)]
)
def test_memoria_sugerida(usada, sugerida):
    assert finops.memoria_sugerida(usada) == sugerida


def test_grafica_de_costo_mensual(tmp_path):
    analisis_ = finops.analizar(resumen_de_prueba())
    figura = tmp_path / "costo_mensual.png"
    finops.grafica_costo_mensual(analisis_, figura)
    assert figura.stat().st_size > 10_000
    json.dumps(analisis_)  # el analisis se puede guardar como JSON
