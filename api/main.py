"""API REST de MiBici (equipo 18): demanda por estacion y hora (P1), servida con FastAPI.

Endpoints:

- ``POST /predicciones/p1``: salidas esperadas de una o varias estaciones-hora.
- ``GET /salud``: el servicio esta arriba y con el modelo cargado (para pruebas y monitoreo).
- ``GET /modelo``: modelo registrado, version, run de origen, variables y linaje.
- ``GET /docs``: documentacion interactiva generada por FastAPI (Swagger UI).

Arranque local, desde la raiz del repo y con el ambiente del proyecto (despues de exportar el
modelo con ``python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1``)::

    uvicorn api.main:app --reload

En el contenedor lo arranca el ``CMD`` del Containerfile. El modelo se carga una sola vez, al
arrancar; si no se puede cargar o no coincide con el contrato, el servicio no arranca.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from api import esquemas
from api.modelo import ErrorPrediccion, ModeloServido, cargar_modelo

VERSION_API = "1.0.0"
logger = logging.getLogger("uvicorn.error")  # sale en la misma consola que los logs de uvicorn


def cargar_modelo_p1() -> ModeloServido:
    """Modelo de ``MODELO_URI`` (o de ``models/servicio/p1``) con las variables de P1."""
    return cargar_modelo(columnas=esquemas.COLUMNAS_P1)


def crear_app(cargador: Callable[[], ModeloServido] = cargar_modelo_p1) -> FastAPI:
    """Crea la aplicacion; las pruebas pasan otro ``cargador`` (por ejemplo, un modelo falso)."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        servido = cargador()
        app.state.modelo = servido
        identidad = servido.identidad
        logger.info(
            f"Modelo {identidad.nombre} version {identidad.version} cargado desde "
            f"{servido.origen} ({len(servido.tipos)} variables)"
        )
        yield

    app = FastAPI(
        title="API MiBici - equipo 18",
        version=VERSION_API,
        summary="Prediccion de la demanda de MiBici por estacion y hora (P1).",
        description=(
            "Sirve el modelo de P1 del Model Registry de MLflow (alias `champion`). "
            "Cada respuesta indica el modelo y la version que la produjo."
        ),
        lifespan=ciclo_de_vida,
    )

    def modelo_actual(request: Request) -> ModeloServido:
        return request.app.state.modelo

    @app.get("/", include_in_schema=False)
    def inicio() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/salud", response_model=esquemas.Salud, tags=["servicio"])
    def salud(request: Request) -> esquemas.Salud:
        """El servicio responde y tiene un modelo cargado."""
        identidad = modelo_actual(request).identidad
        return esquemas.Salud(estado="ok", modelo=identidad.nombre, version=identidad.version)

    @app.get("/modelo", response_model=esquemas.InfoModelo, tags=["servicio"])
    def modelo(request: Request) -> esquemas.InfoModelo:
        """Modelo registrado, version, run de origen, variables de entrada y linaje."""
        servido = modelo_actual(request)
        identidad = servido.identidad
        return esquemas.InfoModelo(
            nombre=identidad.nombre,
            version=identidad.version,
            alias=identidad.alias,
            run_id=identidad.run_id,
            origen=servido.origen,
            variables=list(servido.tipos),
            exportado_en=identidad.exportado_en,
            etiquetas=identidad.etiquetas,
        )

    @app.post(
        "/predicciones/p1",
        response_model=esquemas.RespuestaP1,
        tags=["P1 - demanda"],
        responses={
            422: {"description": "Peticion invalida: el detalle dice que campo y por que."},
            500: {"description": "El modelo no pudo producir una prediccion valida."},
        },
    )
    def predecir_p1(peticion: esquemas.PeticionP1, request: Request) -> esquemas.RespuestaP1:
        """Salidas esperadas de cada estacion en la hora indicada del dia siguiente.

        Acepta de 1 a 1000 filas; la respuesta conserva el orden de ``instancias``.
        """
        servido = modelo_actual(request)
        filas = [
            instancia.model_dump(include=set(esquemas.COLUMNAS_P1))
            for instancia in peticion.instancias
        ]
        try:
            valores = servido.predecir(filas)
        except ErrorPrediccion as error:
            logger.error(f"Prediccion invalida: {error}")
            raise HTTPException(status_code=500, detail=str(error)) from error
        except Exception as error:  # cualquier otra falla del modelo: sin detalles internos
            logger.exception("El modelo fallo al predecir")
            raise HTTPException(
                status_code=500, detail="El modelo no pudo predecir; revisa los logs del servicio"
            ) from error

        identidad = servido.identidad
        return esquemas.RespuestaP1(
            modelo=esquemas.ModeloUsado(
                nombre=identidad.nombre,
                version=identidad.version,
                alias=identidad.alias,
                run_id=identidad.run_id,
            ),
            n=len(valores),
            predicciones=[
                esquemas.PrediccionP1(
                    estacion_id=instancia.estacion_id, salidas_esperadas=round(float(valor), 4)
                )
                for instancia, valor in zip(peticion.instancias, valores, strict=True)
            ],
        )

    return app


app = crear_app()
