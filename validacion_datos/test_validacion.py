"""Validacion de datos con DeepChecks como prueba de pytest (ambiente .venv-validacion).

Corre con ``make validate`` o, desde la raiz del repo,
``.venv-validacion/bin/python -m pytest validacion_datos -rs``.

Que pasa segun el estado de los datos (los datos no estan en Git):
- Faltan: SKIP con el comando a correr; con ``VALIDACION_REQUIERE_DATOS=1`` (``make validate``)
  falla, porque decir "validado" sin datos nunca debe pasar en silencio.
- No coinciden con el manifiesto: falla siempre; se validaria algo que no es el linaje versionado.
- Estan: corre y falla solo si hay hallazgos bloqueantes. Las fallas esperadas quedan
  documentadas en ``suites.ESPERADAS`` y en ``reports/validacion/resumen_validacion.md``.
"""

import os

import pytest

# En el ambiente principal DeepChecks no esta instalado: omitir con un mensaje, no un ImportError
pytest.importorskip("deepchecks", reason="DeepChecks vive en .venv-validacion (ver README)")

from validacion_datos import contrato, suites


@pytest.fixture(scope="session")
def resultados():
    estado = contrato.estado_datos()
    if estado == "desactualizados":
        pytest.fail("features_meta.json no coincide con el manifiesto: corre `make features`")
    if estado == "faltan":
        if os.getenv("VALIDACION_REQUIERE_DATOS") == "1":
            pytest.fail(
                "Faltan datos en data/processed (modo estricto): `make data && make features`"
            )
        pytest.skip("Faltan datos en data/processed: corre `make data && make features`")
    return suites.ejecutar_todo(muestra=contrato.MUESTRA_POR_DEFECTO)


@pytest.mark.parametrize("pregunta", sorted(contrato.ESPECIFICACIONES))
def test_sin_hallazgos_bloqueantes(resultados, pregunta):
    """Ningun check falla sin explicacion ni deja de correr; el mensaje dice cual y donde ver."""
    bloqueantes = suites.bloqueantes(resultados[pregunta])
    lineas = [
        f"- [{h.estado}] {h.suite}/{h.particion}: {h.comprobacion}: {h.detalle}"
        + (f" (reporte: {h.html})" if h.html else "")
        for h in bloqueantes
    ]
    assert not bloqueantes, (
        f"{len(bloqueantes)} hallazgos bloqueantes en {pregunta.upper()}:\n" + "\n".join(lineas)
    )
