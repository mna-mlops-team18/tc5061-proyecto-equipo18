"""Pruebas del reporte de drift con Evidently (ambiente .venv-monitoreo).

Corre con ``make test-monitoreo`` o, desde la raiz del repo,
``.venv-monitoreo/bin/python -m pytest monitoreo -q`` (Windows:
``.venv-monitoreo\\Scripts\\python -m pytest monitoreo -q``).

Usan datos sinteticos con la forma de P1 (``monitoreo/sintetico.py``), asi que no necesitan
los datos reales: comprueban que el reporte detecta un drift que se sabe que existe, que no lo
inventa donde no lo hay y que escribe todas sus salidas.
"""

import pytest

# En el ambiente principal (Python 3.14) Evidently no esta instalado: omitir, no fallar
pytest.importorskip("evidently", reason="Evidently vive en .venv-monitoreo (ver README)")

import pandas as pd

from monitoreo import drift, sintetico


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    """Corre las tres comparaciones sobre datos sinteticos una sola vez para todo el modulo."""
    carpeta = tmp_path_factory.mktemp("monitoreo")
    tabla = drift.ejecutar(
        sintetico.p1_sintetico(),
        muestra=20_000,
        carpeta=carpeta,
        figuras=carpeta / "figuras",
        resumen=carpeta / "resumen_drift.md",
    )
    return carpeta, tabla


def test_escribe_reportes_resumen_y_figuras(salida):
    carpeta, _ = salida
    for nombre in drift.ORDEN_COMPARACIONES:
        assert (carpeta / f"drift_{nombre}.html").stat().st_size > 100_000
        assert (carpeta / f"drift_{nombre}.json").exists()
    assert (carpeta / "drift_puntajes.csv").exists()
    assert (carpeta / "figuras" / "drift_puntajes.png").exists()
    assert (carpeta / "figuras" / "drift_distribuciones.png").exists()
    resumen = (carpeta / "resumen_drift.md").read_text(encoding="utf-8")
    assert "## Drift por variable" in resumen and "## Calidad de los datos" in resumen
    assert resumen.isascii(), "El resumen se versiona: debe ser ASCII (convencion del repo)"


def test_puntajes_cubren_todas_las_variables_en_cada_comparacion(salida):
    carpeta, _ = salida
    puntajes = pd.read_csv(carpeta / "drift_puntajes.csv")
    for _, grupo in puntajes.groupby("comparacion"):
        assert sorted(grupo["variable"]) == sorted(drift.conjuntos.VARIABLES)
    metodos = set(puntajes["metodo"])
    assert metodos == {"wasserstein", "jensenshannon"}


def test_detecta_el_drift_que_se_sabe_que_existe(salida):
    _, tabla = salida
    # Las estaciones nuevas estan mas al norte: latitud deriva solo si se incluyen
    assert tabla.loc["latitud", "deriva_principal"]
    assert not tabla.loc["latitud", "deriva_sin_nuevas"]
    assert tabla.loc["latitud", "causa_probable"] == "estaciones nuevas"
    # La referencia cubre los 12 meses y lo actual solo jul-sep: mes deriva por la temporada
    assert tabla.loc["mes", "deriva_sin_nuevas"]
    assert not tabla.loc["mes", "deriva_misma_temporada"]
    assert tabla.loc["mes", "causa_probable"] == "temporada"
    # La red conocida crece 25 % en 2026: aparece contra la misma temporada de 2025
    assert tabla.loc["salidas", "deriva_misma_temporada"]


def test_no_inventa_drift_donde_no_lo_hay(salida):
    _, tabla = salida
    for variable in ("hora_dia", "es_fin_semana"):
        assert not tabla.loc[variable, ["deriva_principal", "deriva_sin_nuevas"]].any()
        assert tabla.loc[variable, "causa_probable"] == "estable"


def test_puntajes_entiende_distancias_y_p_valores():
    resultado = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "a",
                    "method": "wasserstein",
                    "threshold": 0.1,
                },
                "value": 0.25,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "b",
                    "method": "K-S p_value",
                    "threshold": 0.05,
                },
                "value": 0.30,
            },
            {"config": {"type": "evidently:metric_v2:RowCount"}, "value": 10},
        ]
    }
    tabla = drift.puntajes(resultado, "principal").set_index("variable")
    assert tabla.loc["a", "deriva"]  # distancia alta: deriva
    assert not tabla.loc["b", "deriva"]  # p-valor alto: no deriva


def test_puntajes_falla_si_evidently_no_trae_drift():
    with pytest.raises(RuntimeError, match="ValueDrift"):
        drift.puntajes({"metrics": []}, "principal")


@pytest.mark.parametrize(
    ("principal", "sin_nuevas", "misma_temporada", "esperada"),
    [
        (False, False, False, "estable"),
        (True, False, False, "estaciones nuevas"),
        (True, True, False, "temporada"),
        (True, True, True, "cambio de un anio en la red existente"),
        (True, False, True, "estaciones nuevas + cambio de un anio en la red existente"),
        (False, False, True, "cambio de un anio en la red existente"),
    ],
)
def test_causa_probable(principal, sin_nuevas, misma_temporada, esperada):
    fila = pd.Series(
        {"principal": principal, "sin_nuevas": sin_nuevas, "misma_temporada": misma_temporada}
    )
    assert drift.causa_probable(fila) == esperada


def test_texto_de_evidently_queda_en_ascii():
    assert drift.ascii_simple("Equal 2.000 \u00b1 0.200 | x") == "Equal 2.000 +- 0.200 / x"
