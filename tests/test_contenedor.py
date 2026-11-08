"""Chequeos de la imagen de la API sin construirla: Containerfile, dependencias y contexto.

Construir la imagen requiere podman o docker y los modelos exportados, asi que no corre con
``make test``; estas pruebas atrapan antes los errores que la romperian o la harian distinta de
lo que se probo:

- ``requirements-api.txt`` fija sus versiones y son las mismas de ``requirements.txt``;
- todo lo que importa ``api/`` esta instalado en la imagen;
- la imagen base esta fijada por digest y es la version de Python del proyecto;
- el contexto de construccion solo deja pasar lo que el Containerfile copia;
- el contenedor no corre como root.
"""

import ast
import re
import sys

import pytest

from tc5061_proyecto_equipo18 import config

RAIZ = config.PROJ_ROOT
CONTAINERFILE = (RAIZ / "Containerfile").read_text(encoding="utf-8")
IGNORAR = (RAIZ / ".dockerignore").read_text(encoding="utf-8")

# Paquete de la imagen -> paquete de requirements.txt con la misma version
EQUIVALENTES = {"mlflow-skinny": "mlflow"}
# Modulo importado -> paquete que lo instala (cuando el nombre difiere)
MODULO_A_PAQUETE = {"sklearn": "scikit-learn", "mlflow": "mlflow-skinny"}
# Llegan como dependencia de otro paquete de la lista
TRANSITIVOS = {"pydantic": "fastapi", "starlette": "fastapi"}


def fijados(archivo: str) -> dict[str, str]:
    """Paquete -> version de un archivo de requirements (ignora comentarios y ``-e .``)."""
    versiones = {}
    for linea in (RAIZ / archivo).read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if not linea or linea.startswith("-"):
            continue
        assert "==" in linea, f"{archivo}: version sin fijar en '{linea}'"
        nombre, version = linea.split("==", 1)
        versiones[nombre.strip().lower()] = version.strip()
    return versiones


def instrucciones(nombre: str) -> list[str]:
    """Argumentos de cada instruccion ``nombre`` del Containerfile (une lineas con ``\\``)."""
    texto = CONTAINERFILE.replace("\\\n", " ")
    return [
        linea.split(None, 1)[1].strip()
        for linea in texto.splitlines()
        if linea.strip().upper().startswith(nombre + " ")
    ]


def test_requirements_api_usa_las_versiones_del_proyecto():
    proyecto = fijados("requirements.txt")
    distintas = {
        paquete: (version, proyecto.get(EQUIVALENTES.get(paquete, paquete)))
        for paquete, version in fijados("requirements-api.txt").items()
        if proyecto.get(EQUIVALENTES.get(paquete, paquete)) != version
    }
    assert not distintas, f"Versiones distintas (imagen, proyecto): {distintas}"


def test_lo_que_importa_la_api_esta_en_la_imagen():
    instalados = set(fijados("requirements-api.txt"))
    importados = set()
    for archivo in (RAIZ / "api").glob("*.py"):
        for nodo in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
            if isinstance(nodo, ast.Import):
                importados |= {alias.name.split(".")[0] for alias in nodo.names}
            elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
                importados.add(nodo.module.split(".")[0])
    externos = importados - set(sys.stdlib_module_names) - {"api"}
    faltan = sorted(
        modulo
        for modulo in externos
        if MODULO_A_PAQUETE.get(modulo, modulo) not in instalados
        and TRANSITIVOS.get(modulo) not in instalados
    )
    assert not faltan, f"api/ importa modulos que no instala requirements-api.txt: {faltan}"


def test_imagen_base_fijada_y_con_el_python_del_proyecto():
    (base,) = instrucciones("FROM")
    assert re.search(r"@sha256:[0-9a-f]{64}$", base), "La imagen base debe fijarse por digest"
    requerido = re.search(r"~=\s*(\d+\.\d+)", (RAIZ / "pyproject.toml").read_text()).group(1)
    assert f"python:{requerido}." in base, f"La imagen base no es Python {requerido}: {base}"


def test_el_contexto_solo_deja_pasar_lo_que_se_copia():
    reglas = [
        linea.strip()
        for linea in IGNORAR.splitlines()
        if linea.strip() and not linea.startswith("#")
    ]
    assert reglas[0] == "*", ".dockerignore debe empezar excluyendo todo"
    permitidos = {regla[1:].rstrip("/") for regla in reglas if regla.startswith("!")}
    copiados = {
        origen.rstrip("/")
        for argumentos in instrucciones("COPY")
        for origen in argumentos.split()[:-1]
    }
    assert copiados == permitidos, f"COPY {copiados} y .dockerignore {permitidos} difieren"
    for ruta in ("requirements-api.txt", "api"):
        assert (RAIZ / ruta).exists(), f"El Containerfile copia {ruta} y no existe"


def test_el_contenedor_no_corre_como_root():
    usuarios = instrucciones("USER")
    assert usuarios and usuarios[-1] not in {"root", "0"}


@pytest.mark.parametrize("ruta", ["data", "mlflow.db", "mlruns", "mlartifacts", ".env", ".venv"])
def test_contexto_excluye_datos_y_credenciales(ruta):
    """Con la lista de permitidos, nada de esto entra aunque exista en la carpeta."""
    reglas = {linea.strip() for linea in IGNORAR.splitlines()}
    assert "*" in reglas and f"!{ruta}" not in reglas and f"!{ruta}/" not in reglas


def test_arranca_la_api_en_todas_las_interfaces():
    (cmd,) = instrucciones("CMD")
    assert "api.main:app" in cmd and "0.0.0.0" in cmd  # nosec B104  # dentro del contenedor
