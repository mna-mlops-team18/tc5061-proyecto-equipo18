"""API REST de MiBici (equipo 18): P1, P2 y P3 servidas con FastAPI.

Endpoints:

- ``POST /predicciones/p1``: salidas esperadas de una o varias estaciones-hora.
- ``POST /predicciones/p2``: riesgo de vaciado o llenado de una o varias estaciones-bloque.
- ``POST /predicciones/p3``: grupo de una o varias estaciones segun su perfil de uso.
- ``GET /salud``: el servicio esta arriba y con sus modelos cargados (para pruebas y monitoreo).
- ``GET /modelo``: modelos registrados, versiones, run de origen, variables y linaje.
- ``GET /docs``: documentacion interactiva generada por FastAPI (Swagger UI).

Arranque local, desde la raiz del repo y con el ambiente del proyecto (despues de exportar los
modelos con ``python -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1``, ``p2`` y
``p3``)::

    uvicorn api.main:app --reload

En el contenedor lo arranca el ``CMD`` del Containerfile. Cada modelo se carga una sola vez, al
arrancar. Un modelo que no se puede cargar o no coincide con el contrato no tumba el servicio:
queda registrado con su error, ``GET /salud`` responde 503 y solo su endpoint de prediccion
responde 503; los demas siguen sirviendo.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
import numpy as np

from api import esquemas
from api.modelo import PREGUNTAS, ErrorPrediccion, ModeloServido, cargar_modelo

VERSION_API = "1.0.0"
logger = logging.getLogger("uvicorn.error")  # sale en la misma consola que los logs de uvicorn

Cargador = Callable[[str], ModeloServido]


def cargar_modelo_de(pregunta: str) -> ModeloServido:
    """Modelo de ``MODELO_URI_<PREGUNTA>`` (o de ``models/servicio/<pregunta>``) con su contrato."""
    return cargar_modelo(columnas=esquemas.COLUMNAS[pregunta], pregunta=pregunta)


def crear_app(cargador: Cargador = cargar_modelo_de) -> FastAPI:
    """Crea la aplicacion; las pruebas pasan otro ``cargador`` (por ejemplo, un modelo falso)."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        app.state.modelos = {}
        app.state.errores = {}
        for pregunta in PREGUNTAS:
            try:
                servido = cargador(pregunta)
            except Exception as error:  # noqa: BLE001 - un modelo roto no tumba a los otros dos
                app.state.errores[pregunta] = f"{type(error).__name__}: {error}"
                logger.error(f"No se pudo cargar el modelo de {pregunta}: {error}")
                continue
            app.state.modelos[pregunta] = servido
            identidad = servido.identidad
            logger.info(
                f"Modelo {pregunta} {identidad.nombre} version {identidad.version} cargado desde "
                f"{servido.origen} ({len(servido.tipos)} variables)"
            )
        yield

    app = FastAPI(
        title="API MiBici - equipo 18",
        version=VERSION_API,
        summary="Demanda (P1), riesgo de vaciado o llenado (P2) y grupos de estaciones (P3).",
        description=(
            "Sirve los modelos de P1, P2 y P3 del Model Registry de MLflow (alias `champion`). "
            "Cada respuesta indica el modelo y la version que la produjo."
        ),
        lifespan=ciclo_de_vida,
    )

    def modelo_de(request: Request, pregunta: str) -> ModeloServido:
        """El modelo cargado de la pregunta, o 503 si no se pudo cargar."""
        modelos = request.app.state.modelos
        if pregunta not in modelos:
            motivo = request.app.state.errores.get(pregunta, "el modelo no se ha cargado")
            raise HTTPException(
                status_code=503, detail=f"El modelo de {pregunta} no esta disponible: {motivo}"
            )
        return modelos[pregunta]

    def predecir(
        request: Request, pregunta: str, instancias: list
    ) -> tuple[ModeloServido, np.ndarray]:
        """Predice las instancias ya validadas; 500 si el modelo no da una salida valida."""
        servido = modelo_de(request, pregunta)
        columnas = set(esquemas.COLUMNAS[pregunta])
        filas = [instancia.model_dump(include=columnas) for instancia in instancias]
        try:
            return servido, servido.predecir(filas)
        except ErrorPrediccion as error:
            logger.error(f"Prediccion invalida de {pregunta}: {error}")
            raise HTTPException(status_code=500, detail=str(error)) from error
        except Exception as error:  # cualquier otra falla del modelo: sin detalles internos
            logger.exception(f"El modelo de {pregunta} fallo al predecir")
            raise HTTPException(
                status_code=500, detail="El modelo no pudo predecir; revisa los logs del servicio"
            ) from error

    def usado(servido: ModeloServido) -> esquemas.ModeloUsado:
        identidad = servido.identidad
        return esquemas.ModeloUsado(
            nombre=identidad.nombre,
            version=identidad.version,
            alias=identidad.alias,
            run_id=identidad.run_id,
        )

    @app.get("/", include_in_schema=False)
    def inicio() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get(
        "/salud",
        response_model=esquemas.Salud,
        responses={503: {"model": esquemas.Salud, "description": "Falta cargar algun modelo."}},
        tags=["servicio"],
    )
    def salud(request: Request):
        """200 si los tres modelos estan cargados; 503 (con el detalle de cada uno) si falta alguno."""
        modelos, errores = request.app.state.modelos, request.app.state.errores
        estados = []
        for pregunta in PREGUNTAS:
            if pregunta in modelos:
                identidad = modelos[pregunta].identidad
                estados.append(
                    esquemas.EstadoModelo(
                        pregunta=pregunta,
                        estado="ok",
                        nombre=identidad.nombre,
                        version=identidad.version,
                    )
                )
            else:
                estados.append(
                    esquemas.EstadoModelo(
                        pregunta=pregunta,
                        estado="error",
                        error=errores.get(pregunta, "el modelo no se ha cargado"),
                    )
                )
        completo = all(estado.estado == "ok" for estado in estados)
        respuesta = esquemas.Salud(estado="ok" if completo else "degradado", modelos=estados)
        if not completo:
            return JSONResponse(status_code=503, content=respuesta.model_dump())
        return respuesta

    @app.get("/modelo", response_model=list[esquemas.InfoModelo], tags=["servicio"])
    def modelo(request: Request) -> list[esquemas.InfoModelo]:
        """Modelo registrado, version, run de origen, variables de entrada y linaje de cada pregunta."""
        informes = []
        for pregunta, servido in request.app.state.modelos.items():
            identidad = servido.identidad
            informes.append(
                esquemas.InfoModelo(
                    pregunta=pregunta,
                    nombre=identidad.nombre,
                    version=identidad.version,
                    alias=identidad.alias,
                    run_id=identidad.run_id,
                    origen=servido.origen,
                    variables=list(servido.tipos),
                    exportado_en=identidad.exportado_en,
                    etiquetas=identidad.etiquetas,
                )
            )
        return informes

    respuestas_error = {
        422: {"description": "Peticion invalida: el detalle dice que campo y por que."},
        500: {"description": "El modelo no pudo producir una prediccion valida."},
        503: {"description": "El modelo de esta pregunta no se pudo cargar al arrancar."},
    }

    @app.post(
        "/predicciones/p1",
        response_model=esquemas.RespuestaP1,
        tags=["P1 - demanda"],
        responses=respuestas_error,
    )
    def predecir_p1(peticion: esquemas.PeticionP1, request: Request) -> esquemas.RespuestaP1:
        """Salidas esperadas de cada estacion en la hora indicada del dia siguiente.

        Acepta de 1 a 1000 filas; la respuesta conserva el orden de ``instancias``.
        """
        servido, valores = predecir(request, "p1", peticion.instancias)
        return esquemas.RespuestaP1(
            modelo=usado(servido),
            n=len(valores),
            predicciones=[
                esquemas.PrediccionP1(
                    estacion_id=instancia.estacion_id, salidas_esperadas=round(float(valor), 4)
                )
                for instancia, valor in zip(peticion.instancias, valores, strict=True)
            ],
        )

    @app.post(
        "/predicciones/p2",
        response_model=esquemas.RespuestaP2,
        tags=["P2 - riesgo"],
        responses=respuestas_error,
    )
    def predecir_p2(peticion: esquemas.PeticionP2, request: Request) -> esquemas.RespuestaP2:
        """Riesgo de cada estacion en el bloque de 3 h indicado.

        Devuelve 0 = sin riesgo, 1 = riesgo de vaciado o 2 = riesgo de llenado, con su nombre.
        Acepta de 1 a 1000 filas; la respuesta conserva el orden de ``instancias``.
        """
        servido, clases = predecir(request, "p2", peticion.instancias)
        return esquemas.RespuestaP2(
            modelo=usado(servido),
            n=len(clases),
            predicciones=[
                esquemas.PrediccionP2(
                    estacion_id=instancia.estacion_id,
                    clase=int(clase),
                    riesgo=esquemas.CLASES_P2[int(clase)],
                )
                for instancia, clase in zip(peticion.instancias, clases, strict=True)
            ],
        )

    @app.post(
        "/predicciones/p3",
        response_model=esquemas.RespuestaP3,
        tags=["P3 - grupos"],
        responses=respuestas_error,
    )
    def predecir_p3(peticion: esquemas.PeticionP3, request: Request) -> esquemas.RespuestaP3:
        """Grupo al que pertenece cada estacion segun su perfil de uso.

        Acepta de 1 a 1000 filas; la respuesta conserva el orden de ``instancias``.
        """
        servido, grupos = predecir(request, "p3", peticion.instancias)
        return esquemas.RespuestaP3(
            modelo=usado(servido),
            n=len(grupos),
            predicciones=[
                esquemas.PrediccionP3(estacion_id=instancia.estacion_id, grupo=int(grupo))
                for instancia, grupo in zip(peticion.instancias, grupos, strict=True)
            ],
        )

    return app


app = crear_app()
