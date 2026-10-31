"""Chequeos de seguridad del repositorio: credenciales, archivos sensibles y .gitignore.

Complementan a ``pip-audit`` (vulnerabilidades en dependencias) y ``bandit`` (codigo), que se
corren con ``make security``. Estas pruebas revisan lo que esas herramientas no ven:

- que Git ignore los archivos que pueden traer credenciales o datos (``.env``, la base de
  MLflow, artefactos, datos de MiBici y modelos serializados);
- que ningun archivo versionado sea de un tipo sensible (llaves, ``.env``, bases de datos);
- que ningun archivo versionado contenga algo con forma de credencial;
- que ``.env.example`` solo tenga variables comentadas o vacias.

Si el proyecto no esta en un repositorio de Git (por ejemplo, descargado como zip), se omiten.
"""

from pathlib import Path
import re
import shutil
import subprocess  # nosec B404: solo git con argumentos fijos

import pytest

from tc5061_proyecto_equipo18 import config

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(
    GIT is None or not (config.PROJ_ROOT / ".git").exists(),
    reason="el proyecto no esta en un repositorio de Git",
)

# Rutas de ejemplo que deben quedar fuera de Git (no hace falta que existan)
DEBEN_IGNORARSE = [
    ".env",
    ".env.local",
    "mlflow.db",
    "mlruns/0/meta.yaml",
    "mlartifacts/1/modelo.pkl",
    "data/raw/viajes/datos_abiertos_2026_09.csv",
    "data/processed/p1_demanda_estacion_hora.parquet",
    "models/p1-candidato-hgb_poisson.joblib",
    "modelo.pkl",
]

# Nombres de archivo que nunca deben versionarse (.env.example si: no lleva valores reales)
ARCHIVO_SENSIBLE = re.compile(
    r"(^|/)(\.env(\.(?!example$)[^/]*)?|id_rsa[^/]*|credentials\.json)$"
    r"|\.(pem|key|p12|pfx|db|sqlite3?|parquet|pkl|pickle|joblib)$",
    re.IGNORECASE,
)

# Formas conocidas de credenciales (llaves de nube, tokens de GitHub y Slack, llaves privadas)
PATRONES_CREDENCIAL = {
    "llave de AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "llave privada": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "token de GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}"),
    "token de Slack": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "llave de Google": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    # Variable de entorno con valor: MLFLOW_TRACKING_PASSWORD=abc123 (en mayusculas, sin llamadas
    # como os.getenv(...), que no son el valor)
    "credencial en variable de entorno": re.compile(
        r"\b[A-Z_]*(PASSWORD|PASSWD|SECRET|TOKEN|API_KEY)[A-Z_]*\s*=\s*[\"']?"
        r"[A-Za-z0-9/+_\-]{6,}[\"']?(?=\s|$)"
    ),
    # Texto literal asignado a un nombre de credencial: password = "abc123"
    "credencial en el codigo": re.compile(
        r"(password|passwd|secret|token|api_key)\w*[\"']?\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']",
        re.IGNORECASE,
    ),
}
VARIABLES_CREDENCIAL = re.compile(r"(PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|USERNAME)")


def git(*argumentos: str) -> str:
    resultado = subprocess.run(  # nosec B603: argumentos fijos, sin shell
        [GIT, *argumentos], cwd=config.PROJ_ROOT, capture_output=True, text=True, check=False
    )
    return resultado.stdout


def archivos_versionados() -> list[str]:
    return [ruta for ruta in git("ls-files").splitlines() if ruta]


def test_gitignore_excluye_credenciales_datos_y_modelos():
    ignorados = set(git("check-ignore", "--no-index", *DEBEN_IGNORARSE).splitlines())
    faltan = [ruta for ruta in DEBEN_IGNORARSE if ruta not in ignorados]
    assert not faltan, f".gitignore no excluye: {faltan}"


def test_no_se_versionan_archivos_sensibles():
    sensibles = [ruta for ruta in archivos_versionados() if ARCHIVO_SENSIBLE.search(ruta)]
    assert not sensibles, f"Archivos sensibles en Git: {sensibles}"


def test_archivos_versionados_sin_credenciales():
    hallazgos = []
    for ruta in archivos_versionados():
        archivo = config.PROJ_ROOT / ruta
        if not archivo.is_file() or ruta == "tests/test_seguridad.py":  # define los patrones
            continue
        try:
            texto = archivo.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # binarios (imagenes): no son texto
            continue
        for nombre, patron in PATRONES_CREDENCIAL.items():
            for coincidencia in patron.finditer(texto):
                linea = texto.count("\n", 0, coincidencia.start()) + 1
                hallazgos.append(f"{ruta}:{linea} ({nombre})")
    assert not hallazgos, f"Posibles credenciales versionadas: {hallazgos}"


def test_env_example_no_trae_valores_de_credenciales():
    ejemplo = Path(config.PROJ_ROOT / ".env.example").read_text(encoding="utf-8")
    con_valor = [
        linea
        for linea in ejemplo.splitlines()
        if not linea.lstrip().startswith("#")
        and "=" in linea
        and VARIABLES_CREDENCIAL.search(linea.split("=", 1)[0])
        and linea.split("=", 1)[1].strip()
    ]
    assert not con_valor, f".env.example tiene credenciales con valor: {con_valor}"


@pytest.mark.parametrize(
    ("texto", "nombre"),
    [
        ("aws = 'AKIA" + "ABCDEFGHIJKLMNOP'", "llave de AWS"),
        ("MLFLOW_TRACKING_PASSWORD=" + "s3cr3t-valor", "credencial en variable de entorno"),
        ('password = "' + 'mi-clave-123"', "credencial en el codigo"),
        ("-----BEGIN RSA " + "PRIVATE KEY-----", "llave privada"),
    ],
)
def test_los_patrones_detectan_credenciales(texto, nombre):
    """Control: si los patrones no detectaran nada, la prueba de arriba no demostraria nada."""
    assert PATRONES_CREDENCIAL[nombre].search(texto)


@pytest.mark.parametrize(
    "texto",
    [
        "MLFLOW_TRACKING_PASSWORD=",
        "token = argumentos[i]",
        'TOKEN = os.getenv("MLFLOW_TRACKING_TOKEN")',
        "password = None",
    ],
)
def test_los_patrones_no_marcan_variables_vacias_ni_codigo(texto):
    """Sin valor o con un valor que no es literal (variables, os.getenv): no es credencial."""
    for nombre in ("credencial en variable de entorno", "credencial en el codigo"):
        assert not PATRONES_CREDENCIAL[nombre].search(texto), nombre
