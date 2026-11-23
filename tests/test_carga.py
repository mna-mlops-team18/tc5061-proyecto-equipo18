"""Pruebas de la prueba de carga: peticiones validas, lectura de docker stats, resumen por escalon
y archivos que genera. No corren Locust ni Docker (eso lo hace ``make carga`` con el sistema
arriba)."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from api import esquemas
from carga import analisis, ejecutar, peticiones, recursos
from tc5061_proyecto_equipo18 import config

MODELOS_PETICION = {
    "p1": esquemas.PeticionP1,
    "p2": esquemas.PeticionP2,
    "p3": esquemas.PeticionP3,
}


# =============================================================================
# Peticiones del escenario
# =============================================================================
@pytest.mark.parametrize("pregunta", peticiones.PREGUNTAS)
def test_las_peticiones_al_azar_son_validas_para_la_api(pregunta):
    azar = random.Random(0)
    for _ in range(200):
        cuerpo = peticiones.peticion(pregunta, azar, instancias=3)
        MODELOS_PETICION[pregunta].model_validate(cuerpo)  # falla si alguna fila es invalida
        assert len(cuerpo["instancias"]) == 3


def test_las_peticiones_varian_y_no_modifican_los_ejemplos():
    originales = json.dumps(peticiones.EJEMPLOS, sort_keys=True)
    azar = random.Random(1)
    filas = [peticiones.fila("p1", azar) for _ in range(50)]
    assert len({(f["estacion_id"], f["hora_dia"], f["dia_semana"]) for f in filas}) > 40
    assert json.dumps(peticiones.EJEMPLOS, sort_keys=True) == originales


def test_escalones_de_config():
    assert config.CARGA_ESCALONES and all(u > 0 and s > 0 for u, s in config.CARGA_ESCALONES)
    usuarios = [u for u, _ in config.CARGA_ESCALONES]
    assert usuarios == sorted(usuarios)  # la carga solo sube
    assert all(s > 2 * config.CARGA_CALENTAMIENTO_SEG for _, s in config.CARGA_ESCALONES)
    # P2 en tiempo real; P1 y P3 por lotes (README > Estrategia de despliegue)
    assert set(config.CARGA_PESOS) == {"p2", "salud"}
    assert set(config.CARGA_PESOS_LOTES) == {"p1", "p3"}
    assert all(1 <= filas <= esquemas.MAX_INSTANCIAS for filas in config.CARGA_FILAS.values())


# =============================================================================
# docker stats
# =============================================================================
@pytest.mark.parametrize(
    ("texto", "bytes_"),
    [
        ("0B", 0),
        ("512KiB", 512 * 1024),
        ("206.5MiB", 206.5 * 2**20),
        ("1GiB", 2**30),
        ("1.2kB", 1200),
        ("3.4MB", 3.4e6),
        ("2.5GB", 2.5e9),
        ("10.0MB", 10e6),
    ],
)
def test_tamanos_de_docker_stats(texto, bytes_):
    assert recursos.a_bytes(texto) == pytest.approx(bytes_)


def test_tamano_invalido():
    with pytest.raises(ValueError, match="no reconocid"):
        recursos.a_bytes("mucho")
    with pytest.raises(ValueError, match="Unidad"):
        recursos.a_bytes("3 parsecs")


def test_linea_de_docker_stats():
    linea = "mibici-api\t98.03%\t238.3MiB / 1GiB\t23.27%\t12.1MB / 30.5MB"
    fila = recursos.leer_linea(linea, tiempo=100.0)
    assert fila == {
        "tiempo": 100.0,
        "contenedor": "mibici-api",
        "cpu_pct": 98.03,
        "memoria_mib": 238.3,
        "limite_mib": 1024.0,
        "memoria_pct": 23.27,
        "red_entrada_mb": 12.1,
        "red_salida_mb": 30.5,
    }
    assert list(fila) == recursos.COLUMNAS


def test_porcentaje_sin_dato_de_podman():
    assert np.isnan(recursos.porcentaje("--"))


# =============================================================================
# Resumen por escalon
# =============================================================================
def corrida_sintetica(semilla: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tres escalones de 20 s: 10 y 50 usuarios rapidos; 100 usuarios lentos y con errores.

    En tiempo real, P2 con 20 filas; cada segundo, ademas, un lote de P1 de 1000 filas que tarda
    5 s (no debe contar en la latencia en tiempo real)."""
    azar = np.random.default_rng(semilla)
    inicio, filas, recursos_ = 1_000.0, [], []
    escalones = [(10, 10, 20.0, 0.0), (50, 50, 30.0, 0.0), (100, 60, 900.0, 0.05)]
    for i, (usuarios, rps, ms, error) in enumerate(escalones):
        for segundo in range(20):
            t = inicio + 20 * i + segundo
            for _ in range(rps):
                fallo = int(azar.random() < error)
                filas.append(
                    {
                        "tiempo": t + azar.random(),
                        "endpoint": azar.choice(["/predicciones/p2", "/salud"], p=[0.9, 0.1]),
                        "ms": azar.normal(ms, ms / 10),
                        "fallo": fallo,
                        "usuarios": usuarios,
                        "filas": 0 if fallo else 20,
                    }
                )
            filas.append(
                {
                    "tiempo": t + 0.5,
                    "endpoint": "/predicciones/p1 [lote]",
                    "ms": 5000.0,
                    "fallo": 0,
                    "usuarios": usuarios,
                    "filas": 1000,
                }
            )
            if segundo % 2 == 0:
                for nombre, cpu in (("mibici-api", 10.0 * (i + 1)), ("mibici-mlflow", 1.0)):
                    recursos_.append(
                        {
                            "tiempo": t,
                            "contenedor": nombre,
                            "cpu_pct": cpu,
                            "memoria_mib": 200.0 + i,
                            "limite_mib": 1024.0,
                            "memoria_pct": 20.0,
                            "red_entrada_mb": 0.0,
                            "red_salida_mb": 0.0,
                        }
                    )
    return pd.DataFrame(filas), pd.DataFrame(recursos_)


@pytest.fixture
def corrida(tmp_path):
    filas, muestras = corrida_sintetica()
    filas.to_csv(tmp_path / "peticiones.csv", index=False)
    muestras.to_csv(tmp_path / "docker_stats.csv", index=False)
    return tmp_path


def test_escalones_rps_latencia_errores_y_cpu(corrida):
    datos = analisis.leer_peticiones(corrida / "peticiones.csv")
    muestras = analisis.leer_recursos(corrida / "docker_stats.csv", datos["tiempo"].min())
    escalones = analisis.por_escalon(datos, muestras).set_index("usuarios")

    assert list(escalones.index) == [10, 50, 100]
    assert escalones.loc[10, "rps"] == pytest.approx(11, rel=0.1)  # 10 + 1 lote por segundo
    assert escalones.loc[50, "rps"] == pytest.approx(51, rel=0.1)
    # Predicciones: 20 por peticion en tiempo real (las /salud no cuentan) + 1000 del lote
    assert escalones.loc[10, "predicciones_por_seg"] == pytest.approx(10 * 0.9 * 20 + 1000, 0.1)
    # Los lotes de 5 s no entran en la latencia en tiempo real, pero tienen su propio p95
    assert escalones.loc[10, "p95_ms"] < 30 and escalones.loc[100, "p95_ms"] > 900
    assert escalones["p95_lote_ms"].eq(5000).all()
    assert escalones.loc[10, "tasa_error"] == 0 and escalones.loc[100, "tasa_error"] > 0.02
    # El calentamiento no cuenta: cada escalon dura ~20 - 5 s
    assert escalones["segundos"].between(13, 15).all()
    # CPU de la API (no de MLflow) en cada escalon
    assert list(escalones["cpu_api_pct"]) == [10.0, 20.0, 30.0]
    assert list(escalones["cumple_slo"]) == [True, True, False]


def test_capacidad_es_el_mayor_rps_que_cumple(corrida):
    datos = analisis.leer_peticiones(corrida / "peticiones.csv")
    escalones = analisis.por_escalon(datos, analisis.leer_recursos(corrida / "x.csv", 0))
    capacidad = analisis.capacidad(escalones)
    assert capacidad["usuarios"] == 50  # el de 100 tiene mas RPS pero no cumple
    assert np.isnan(escalones["cpu_api_pct"]).all()  # sin docker stats no hay CPU
    assert analisis.capacidad(escalones.assign(cumple_slo=False)) is None


def test_por_segundo_sin_el_ultimo_incompleto(corrida):
    serie = analisis.por_segundo(analisis.leer_peticiones(corrida / "peticiones.csv"))
    assert len(serie) == 59  # 60 s menos el ultimo
    assert {"usuarios", "rps", "p50_ms", "p95_ms", "p99_ms", "tasa_error"} <= set(serie.columns)


def test_genera_resumen_tablas_y_graficas(corrida):
    entorno = {"modelos_de_prueba": True, "modelos_texto": "P1 x v1", "fecha": "2026-11-22"}
    destino, figuras = corrida / "reporte", corrida / "figuras"
    resumen = analisis.generar(
        corrida / "peticiones.csv",
        corrida / "docker_stats.csv",
        entorno,
        destino_dir=destino,
        figuras_dir=figuras,
    )

    guardado = json.loads((destino / "resumen_carga.json").read_text(encoding="utf-8"))
    assert guardado["peticiones"] == resumen["peticiones"] == 2460
    assert guardado["p99_ms"] < 5000  # sin los lotes
    assert guardado["capacidad"]["usuarios"] == 50
    endpoints = {e["endpoint"]: e for e in guardado["por_endpoint"]}
    assert set(endpoints) == {"/predicciones/p1 [lote]", "/predicciones/p2", "/salud"}
    assert endpoints["/predicciones/p1 [lote]"]["filas_por_peticion"] == 1000
    # Tiempo de servicio sin cola: solo el escalon de 10 usuarios (sin los lentos de 100)
    sin_saturar = {e["endpoint"]: e for e in guardado["por_endpoint_sin_saturar"]}
    assert sin_saturar["/predicciones/p2"]["p95_ms"] < 30
    assert (
        sin_saturar["/predicciones/p2"]["peticiones"] < endpoints["/predicciones/p2"]["peticiones"]
    )
    assert set(guardado["recursos"]) == {"mibici-api", "mibici-mlflow"}
    assert len(pd.read_csv(destino / "escalones.csv")) == 3
    assert len(pd.read_csv(destino / "recursos.csv")) == 60
    for nombre in ("carga_en_el_tiempo", "capacidad_por_escalon", "recursos_contenedores"):
        assert (figuras / f"{nombre}.png").stat().st_size > 10_000

    texto = (destino / "resumen_carga.md").read_text(encoding="utf-8")
    assert "modelos de prueba" in texto and "../figuras/capacidad_por_escalon.png" in texto
    assert texto.isascii()


def test_json_sin_nan():
    assert analisis.sin_nan({"a": float("nan"), "b": [np.float64(1.5), np.int64(2)]}) == {
        "a": None,
        "b": [1.5, 2],
    }


def test_sin_peticiones_da_error_claro(tmp_path):
    (tmp_path / "peticiones.csv").write_text("tiempo,endpoint,ms,fallo,usuarios,filas\n")
    with pytest.raises(ValueError, match="no tiene peticiones"):
        analisis.leer_peticiones(tmp_path / "peticiones.csv")


# =============================================================================
# Entorno de la corrida
# =============================================================================
def test_detecta_modelos_de_prueba():
    reales = [{"pregunta": "p1", "nombre": "m", "version": "3", "run_id": "abc"}]
    prueba = [{"pregunta": "p1", "nombre": "m", "version": "7", "run_id": "run-de-prueba-p1"}]
    assert ejecutar.describir_modelos(reales) == ("P1 m v3", False)
    assert ejecutar.describir_modelos(prueba)[1] is True


def test_texto_de_limites():
    assert ejecutar.texto_limites({"cpus": 1.0, "memoria_mib": 1024, "workers": 2}) == (
        "1 vCPU, 1024 MiB, 2 worker(s) de uvicorn"
    )
    assert "sin limite" in ejecutar.texto_limites({"cpus": None, "memoria_mib": None})


def test_url_no_http_se_rechaza():
    with pytest.raises(ValueError, match="no soportada"):
        ejecutar.consultar("file:///etc/passwd")
