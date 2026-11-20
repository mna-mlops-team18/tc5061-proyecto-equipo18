"""Pruebas de humo del sistema levantado con Docker Compose (API + MLflow).

Despues de ``docker compose up -d --build``, desde la raiz del repo::

    python -m operacion.verificar

Revisa, desde fuera de los contenedores, lo mismo que un usuario:

1. ``GET /salud`` de la API responde 200 con los tres modelos cargados.
2. ``GET /modelo`` reporta P1, P2 y P3 con su version.
3. ``POST /predicciones/p1``, ``p2`` y ``p3`` con las peticiones de ``api/ejemplos`` responden 200
   con una prediccion por fila.
4. Una peticion invalida (``hora_dia`` = 30) responde 422.
5. El servidor de MLflow responde en ``/health``.

Termina con codigo 1 si algo falla (asi lo usa el CI). Solo usa la biblioteca estandar.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

EJEMPLOS_DIR = Path(__file__).resolve().parents[1] / "api" / "ejemplos"


def pedir(url: str, cuerpo: dict | None = None) -> tuple[int, str]:
    """Codigo HTTP y texto de la respuesta (GET, o POST con JSON si hay cuerpo)."""
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(url, data=datos, method="GET" if datos is None else "POST")
    peticion.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(peticion, timeout=30) as respuesta:  # nosec B310  # http local
            return respuesta.status, respuesta.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as error:
        return 0, str(error)


def revisar(api: str, mlflow: str) -> list[tuple[str, bool, str]]:
    """Lista de (prueba, paso, detalle)."""
    resultados = []

    codigo, texto = pedir(f"{api}/salud")
    modelos = json.loads(texto).get("modelos", []) if codigo in (200, 503) else []
    detalle = ", ".join(f"{m['pregunta']} v{m.get('version')} {m['estado']}" for m in modelos)
    resultados.append(("API /salud", codigo == 200, f"HTTP {codigo}; {detalle or texto[:120]}"))

    codigo, texto = pedir(f"{api}/modelo")
    preguntas = sorted(m["pregunta"] for m in json.loads(texto)) if codigo == 200 else []
    resultados.append(
        ("API /modelo", preguntas == ["p1", "p2", "p3"], f"HTTP {codigo}; {preguntas}")
    )

    for pregunta in ("p1", "p2", "p3"):
        ejemplo = json.loads((EJEMPLOS_DIR / f"peticion_{pregunta}.json").read_text("utf-8"))
        codigo, texto = pedir(f"{api}/predicciones/{pregunta}", ejemplo)
        n = json.loads(texto).get("n") if codigo == 200 else None
        esperado = len(ejemplo["instancias"])
        resultados.append(
            (f"API /predicciones/{pregunta}", n == esperado, f"HTTP {codigo}; {n} de {esperado}")
        )

    invalida = json.loads((EJEMPLOS_DIR / "peticion_p1.json").read_text("utf-8"))
    invalida["instancias"][0]["hora_dia"] = 30
    codigo, _ = pedir(f"{api}/predicciones/p1", invalida)
    resultados.append(("API rechaza hora_dia = 30", codigo == 422, f"HTTP {codigo}"))

    codigo, texto = pedir(f"{mlflow}/health")
    resultados.append(("MLflow /health", codigo == 200, f"HTTP {codigo}; {texto.strip()[:40]}"))
    return resultados


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default=os.getenv("CARGA_URL_API", "http://127.0.0.1:8000"))
    parser.add_argument("--mlflow", default=os.getenv("URL_MLFLOW", "http://127.0.0.1:5000"))
    args = parser.parse_args(argumentos)
    resultados = revisar(args.api.rstrip("/"), args.mlflow.rstrip("/"))
    print("| Prueba | Resultado | Detalle |\n|---|---|---|")
    for prueba, paso, detalle in resultados:
        print(f"| {prueba} | {'ok' if paso else 'FALLA'} | {detalle} |")
    return 0 if all(paso for _, paso, _ in resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
