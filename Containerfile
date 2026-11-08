# Imagen de la API de MiBici (equipo 18): FastAPI + modelos de P1, P2 y P3 exportados del
# Model Registry. Construccion y uso: README > Despliegue de la API, o make image / make run-api.
#
# Antes de construir hay que exportar los champions a models/servicio/:
#   python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1   (y p2, p3)
#
# Python fijo por digest: la misma version de requirements.txt (~=3.14) y el mismo sistema base
# en cada construccion.
FROM docker.io/library/python:3.14.10-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app

WORKDIR /app

# Dependencias primero: si solo cambia el codigo o el modelo, esta capa se reutiliza
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Usuario sin privilegios: el proceso no puede modificar el codigo ni los modelos de la imagen
RUN useradd --uid 10001 --home-dir /app --no-create-home --shell /usr/sbin/nologin api

COPY api/ api/
COPY models/servicio/ models/servicio/

USER api
EXPOSE 8000

# /salud responde 200 si los tres modelos cargaron y 503 si alguno no (estado "degradado")
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salud', timeout=4)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
