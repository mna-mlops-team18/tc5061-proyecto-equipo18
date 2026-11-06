"""Servicio de prediccion de MiBici (equipo 18): API REST con FastAPI sobre el modelo de P1.

- ``esquemas.py``: contrato de entrada y salida (Pydantic).
- ``modelo.py``: carga del modelo exportado del Model Registry y prediccion.
- ``main.py``: la aplicacion FastAPI (``uvicorn api.main:app``).

Este paquete no importa ``tc5061_proyecto_equipo18``: la imagen solo lleva la API y el modelo,
no el codigo ni las dependencias de entrenamiento.
"""
