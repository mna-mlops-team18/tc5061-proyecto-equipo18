"""Esquemas de entrada y salida de la API (Pydantic): el contrato del servicio.

FastAPI valida cada peticion contra estas clases antes de llamar al modelo, y cada respuesta
antes de enviarla. Una peticion invalida recibe el codigo 422 con la lista de errores (campo,
valor recibido y motivo) y nunca llega al modelo.

Reglas de entrada (``InstanciaP1``):

- **Tipos estrictos:** un entero no acepta ``"8"``, ``8.5`` ni ``true``; un numero no acepta
  texto. Asi un error del cliente no se convierte en silencio en otro valor.
- **Sin campos desconocidos:** un nombre mal escrito (``hora`` en lugar de ``hora_dia``) se
  rechaza en lugar de ignorarse.
- **Sin NaN ni infinito**, y cada variable dentro de su rango (hora 0 a 23, mes 1 a 12,
  conteos no negativos, coordenadas dentro del Area Metropolitana de Guadalajara).
- **Coherencia:** ``es_fin_semana`` debe corresponder a ``dia_semana``.
- **Tamano del lote:** de 1 a ``MAX_INSTANCIAS`` filas por peticion.

Las variables son las de ``features.COLUMNAS_P1``, con el mismo nombre y orden. No se importan
de ``features.py`` para que la imagen no necesite el paquete de entrenamiento; la prueba
``tests/test_api_esquemas.py`` falla si las dos listas dejan de coincidir.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Limites de dominio: fuera de ellos el modelo no vio datos o el valor no tiene sentido
LATITUD_MIN, LATITUD_MAX = 20.4, 20.9  # Area Metropolitana de Guadalajara, con margen
LONGITUD_MIN, LONGITUD_MAX = -103.6, -103.1
CAPACIDAD_MAX = 200  # anclajes; las estaciones de MiBici tienen decenas
VIAJES_MAX = 1000.0  # viajes por estacion y hora; muy por encima de lo observado
MAX_INSTANCIAS = 1000  # filas por peticion; volumenes mayores, en varias peticiones o por lotes

# Tipos estrictos, sin campos desconocidos y sin NaN ni infinito (ver el docstring del modulo)
ESTRICTO = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

EJEMPLO_PETICION_P1 = {
    "instancias": [
        {
            "estacion_id": 101,
            "hora_dia": 8,
            "dia_semana": 2,
            "es_fin_semana": 0,
            "es_festivo": 0,
            "mes": 9,
            "capacidad": 24,
            "latitud": 20.6745,
            "longitud": -103.3612,
            "salidas_lag_24h": 7,
            "salidas_lag_48h": 6,
            "salidas_lag_168h": 8,
            "salidas_lag_336h": 7,
            "llegadas_lag_24h": 3,
            "salidas_media_misma_hora_7d": 6.4,
            "salidas_media_24h_previa": 2.3,
            "salidas_media_7d": 2.1,
        },
        {
            "estacion_id": 202,
            "hora_dia": 17,
            "dia_semana": 5,
            "es_fin_semana": 1,
            "es_festivo": 0,
            "mes": 9,
            "capacidad": 16,
            "latitud": 20.6597,
            "longitud": -103.392,
            "salidas_lag_24h": 2,
            "salidas_lag_48h": 1,
            "salidas_lag_168h": 3,
            "salidas_lag_336h": 2,
            "llegadas_lag_24h": 2,
            "salidas_media_misma_hora_7d": 1.7,
            "salidas_media_24h_previa": 0.8,
            "salidas_media_7d": 0.9,
        },
    ]
}


def _conteo(descripcion: str):
    """Campo de viajes: numero finito entre 0 y ``VIAJES_MAX``."""
    return Field(ge=0, le=VIAJES_MAX, description=descripcion)


# =============================================================================
# Entrada
# =============================================================================
class InstanciaP1(BaseModel):
    """Una estacion en una hora del dia siguiente: las 16 variables del modelo de P1.

    Todas se conocen al cierre del dia anterior (horizonte de 24 h), igual que en el
    entrenamiento: los rezagos y medias usan solo viajes de hace 24 h o mas.
    """

    model_config = ESTRICTO

    estacion_id: int | None = Field(
        default=None,
        ge=0,
        description="Opcional. No entra al modelo: se devuelve tal cual para ubicar la fila.",
    )
    hora_dia: int = Field(ge=0, le=23, description="Hora que se predice, de 0 a 23.")
    dia_semana: int = Field(ge=0, le=6, description="Dia de la semana: 0 = lunes ... 6 = domingo.")
    es_fin_semana: int = Field(ge=0, le=1, description="1 si es sabado o domingo, 0 si no.")
    es_festivo: int = Field(ge=0, le=1, description="1 si es dia festivo oficial, 0 si no.")
    mes: int = Field(ge=1, le=12, description="Mes de la fecha que se predice, de 1 a 12.")
    capacidad: int = Field(gt=0, le=CAPACIDAD_MAX, description="Anclajes de la estacion.")
    latitud: float = Field(
        ge=LATITUD_MIN, le=LATITUD_MAX, description="Latitud de la estacion (grados)."
    )
    longitud: float = Field(
        ge=LONGITUD_MIN, le=LONGITUD_MAX, description="Longitud de la estacion (grados)."
    )
    salidas_lag_24h: float = _conteo("Salidas de la misma hora, hace 1 dia.")
    salidas_lag_48h: float = _conteo("Salidas de la misma hora, hace 2 dias.")
    salidas_lag_168h: float = _conteo("Salidas de la misma hora, hace 7 dias.")
    salidas_lag_336h: float = _conteo("Salidas de la misma hora, hace 14 dias.")
    llegadas_lag_24h: float = _conteo("Llegadas de la misma hora, hace 1 dia.")
    salidas_media_misma_hora_7d: float = _conteo(
        "Media de las salidas a la misma hora en los 7 dias anteriores."
    )
    salidas_media_24h_previa: float = _conteo(
        "Media por hora de las salidas en las 24 h que terminan 24 h antes de la hora predicha."
    )
    salidas_media_7d: float = _conteo(
        "Media por hora de las salidas en los 7 dias que terminan 24 h antes de la hora predicha."
    )

    @model_validator(mode="after")
    def _fin_de_semana_coherente(self):
        """``es_fin_semana`` se deriva de ``dia_semana``; si no coinciden, la fila esta mal."""
        esperado = 1 if self.dia_semana >= 5 else 0
        if self.es_fin_semana != esperado:
            raise ValueError(
                f"es_fin_semana debe ser {esperado} cuando dia_semana es {self.dia_semana} "
                "(1 solo para 5 = sabado y 6 = domingo)"
            )
        return self


# Variables que recibe el modelo, en el orden de features.COLUMNAS_P1 (sin estacion_id)
COLUMNAS_P1 = [nombre for nombre in InstanciaP1.model_fields if nombre != "estacion_id"]


class PeticionP1(BaseModel):
    """Cuerpo de ``POST /predicciones/p1``: una o varias filas en ``instancias``."""

    # El ejemplo aparece en /docs y llena el boton "Try it out"
    model_config = ConfigDict(**ESTRICTO, json_schema_extra={"examples": [EJEMPLO_PETICION_P1]})

    instancias: list[InstanciaP1] = Field(
        min_length=1,
        max_length=MAX_INSTANCIAS,
        description=f"De 1 a {MAX_INSTANCIAS} filas; la respuesta conserva el mismo orden.",
    )


# =============================================================================
# Salida
# =============================================================================
class ModeloUsado(BaseModel):
    """De que modelo y version del Model Registry sale la respuesta (trazabilidad)."""

    nombre: str = Field(description="Modelo registrado en MLflow.")
    version: str = Field(description="Version del Model Registry.")
    alias: str | None = Field(default=None, description="Alias con que se eligio (champion).")
    run_id: str | None = Field(default=None, description="Run de MLflow que entreno el modelo.")


class PrediccionP1(BaseModel):
    """Prediccion de una fila, en el mismo orden de la peticion."""

    model_config = ConfigDict(allow_inf_nan=False)

    estacion_id: int | None = Field(default=None, description="El mismo de la peticion.")
    salidas_esperadas: float = Field(
        ge=0, description="Viajes que se espera salgan de la estacion en esa hora."
    )


class RespuestaP1(BaseModel):
    """Respuesta de ``POST /predicciones/p1``."""

    model_config = ConfigDict(allow_inf_nan=False)

    modelo: ModeloUsado
    n: int = Field(ge=1, description="Numero de predicciones (igual al de instancias).")
    predicciones: list[PrediccionP1]

    @model_validator(mode="after")
    def _una_por_instancia(self):
        if self.n != len(self.predicciones):
            raise ValueError(f"n = {self.n}, pero hay {len(self.predicciones)} predicciones")
        return self


class Salud(BaseModel):
    """Respuesta de ``GET /salud``: el servicio esta arriba y con el modelo cargado."""

    estado: str = Field(description="ok si la API cargo el modelo y responde.")
    modelo: str
    version: str


class InfoModelo(BaseModel):
    """Respuesta de ``GET /modelo``: detalle y linaje del modelo cargado."""

    nombre: str
    version: str
    alias: str | None = None
    run_id: str | None = None
    origen: str = Field(description="Carpeta de la imagen o URI del Model Registry.")
    variables: list[str] = Field(description="Columnas de la firma del modelo, en orden.")
    exportado_en: str | None = Field(default=None, description="Fecha de la exportacion (UTC).")
    etiquetas: dict[str, str] = Field(
        default_factory=dict,
        description="Linaje de la version: run, commit, version de datos, metrica, comando.",
    )
