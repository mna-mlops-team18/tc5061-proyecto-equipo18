"""Chequeos de docker-compose.yml y de la imagen del servidor de MLflow, sin levantarlos.

Levantar el sistema requiere Docker o Podman (lo hace el CI en el job de despliegue); estas
pruebas atrapan antes los errores de configuracion que lo romperian o lo harian inseguro:

- los dos servicios (API y MLflow) se construyen desde archivos que existen;
- cada servicio tiene healthcheck, politica de reinicio y limites de recursos;
- la API espera a que MLflow este sano y lo llama por su nombre dentro de la red;
- los puertos solo se publican en 127.0.0.1 y los datos de MLflow viven en un volumen;
- la imagen de MLflow usa la misma base y la misma version de MLflow que el proyecto;
- cada variable ``${...}`` de docker-compose.yml esta documentada en .env.example.
"""

import re

import pytest
import yaml

from tc5061_proyecto_equipo18 import config

RAIZ = config.PROJ_ROOT
TEXTO = (RAIZ / "docker-compose.yml").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(TEXTO)
SERVICIOS = COMPOSE["services"]
PRINCIPALES = ("api", "mlflow")  # los que arranca `docker compose up` (sin perfiles)


def version_fijada(archivo: str, paquete: str) -> str:
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    return re.search(rf"^{paquete}==(\S+)", texto, re.MULTILINE).group(1)


def imagen_base(containerfile: str) -> str:
    texto = (RAIZ / containerfile).read_text(encoding="utf-8")
    return re.search(r"^FROM\s+(\S+)", texto, re.MULTILINE).group(1)


def test_up_levanta_la_api_y_mlflow():
    sin_perfil = {nombre for nombre, servicio in SERVICIOS.items() if "profiles" not in servicio}
    assert sin_perfil == set(PRINCIPALES)


@pytest.mark.parametrize("nombre", PRINCIPALES)
def test_cada_servicio_se_construye_desde_un_archivo_del_repo(nombre):
    build = SERVICIOS[nombre]["build"]
    assert (RAIZ / build["context"] / build["dockerfile"]).is_file()


@pytest.mark.parametrize("nombre", PRINCIPALES)
def test_cada_servicio_tiene_salud_reinicio_y_limites(nombre):
    servicio = SERVICIOS[nombre]
    assert servicio["healthcheck"]["test"][0] == "CMD"
    assert servicio["restart"] == "unless-stopped"
    limites = servicio["deploy"]["resources"]["limits"]
    assert limites["cpus"] and limites["memory"]
    assert "no-new-privileges:true" in servicio["security_opt"]
    assert servicio["networks"] == ["mibici"]


@pytest.mark.parametrize("nombre", PRINCIPALES)
def test_puertos_solo_en_localhost(nombre):
    for puerto in SERVICIOS[nombre]["ports"]:
        assert puerto.startswith("127.0.0.1:"), (
            f"{nombre} publica {puerto} en todas las interfaces"
        )


def test_la_api_espera_a_mlflow_y_lo_llama_por_su_nombre():
    api = SERVICIOS["api"]
    assert api["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert api["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    permitidos = SERVICIOS["mlflow"]["environment"]["MLFLOW_SERVER_ALLOWED_HOSTS"].split(",")
    assert "mlflow" in permitidos and "mlflow:*" in permitidos  # si no, MLflow rechaza a la API
    assert "/salud" in api["healthcheck"]["test"][-1]
    assert SERVICIOS["mlflow"]["environment"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"


def test_mlflow_guarda_sus_datos_en_un_volumen_con_nombre():
    volumen, destino = SERVICIOS["mlflow"]["volumes"][0].split(":")
    assert volumen in COMPOSE["volumes"] and destino == "/mlflow"
    containerfile = (RAIZ / "servidor_mlflow" / "Containerfile").read_text(encoding="utf-8")
    assert "sqlite:////mlflow/mlflow.db" in containerfile
    assert "/mlflow/mlartifacts" in containerfile


def test_imagen_de_mlflow_con_la_base_y_la_version_del_proyecto():
    assert imagen_base("servidor_mlflow/Containerfile") == imagen_base("Containerfile")
    assert version_fijada("servidor_mlflow/requirements.txt", "mlflow") == version_fijada(
        "requirements.txt", "mlflow"
    )


def test_mlflow_no_corre_como_root():
    texto = (RAIZ / "servidor_mlflow" / "Containerfile").read_text(encoding="utf-8")
    usuarios = re.findall(r"^USER\s+(\S+)", texto, re.MULTILINE)
    assert usuarios and usuarios[-1] not in {"root", "0"}


def test_las_variables_de_compose_estan_en_env_example():
    variables = set(re.findall(r"\$\{([A-Z0-9_]+)", TEXTO))
    ejemplo = (RAIZ / ".env.example").read_text(encoding="utf-8")
    faltan = sorted(v for v in variables if not re.search(rf"^#?\s*{v}=", ejemplo, re.MULTILINE))
    assert not faltan, f"Documenta en .env.example: {faltan}"


def test_compose_sin_credenciales():
    assert not re.search(r"(PASSWORD|SECRET|TOKEN|API_KEY)\w*:\s*\S", TEXTO)
