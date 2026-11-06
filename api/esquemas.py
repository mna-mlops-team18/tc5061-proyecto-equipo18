"""Esquemas de entrada y salida de la API (Pydantic): el contrato del servicio.

FastAPI valida cada peticion contra estas clases antes de llamar al modelo, y cada respuesta
antes de enviarla. Una peticion invalida recibe el codigo 422 con la lista de errores (campo,
valor recibido y motivo) y nunca llega al modelo.

Una pregunta, un contrato: ``InstanciaP1`` (demanda por estacion y hora), ``InstanciaP2`` (riesgo
de vaciado o llenado por bloque de 3 h) e ``InstanciaP3`` (grupo de una estacion segun su perfil de
uso). Reglas de entrada comunes:

- **Tipos estrictos:** un entero no acepta ``"8"``, ``8.5`` ni ``true``; un numero no acepta
  texto. Asi un error del cliente no se convierte en silencio en otro valor.
- **Sin campos desconocidos:** un nombre mal escrito (``hora`` en lugar de ``hora_dia``) se
  rechaza en lugar de ignorarse.
- **Sin NaN ni infinito**, y cada variable dentro de su rango (hora 0 a 23, mes 1 a 12,
  conteos no negativos, coordenadas dentro del Area Metropolitana de Guadalajara).
- **Coherencia:** ``es_fin_semana`` debe corresponder a ``dia_semana`` (P1 y P2); en P3, cada
  perfil horario suma 1 (o 0 si la estacion no tuvo viajes de ese tipo).
- **Tamano del lote:** de 1 a ``MAX_INSTANCIAS`` filas por peticion.

Las variables son las de ``features.COLUMNAS_P1`` y ``COLUMNAS_P2`` y las de
``features.perfiles_estacion`` (P3), con el mismo nombre. No se importan de ``features.py`` para
que la imagen no necesite el paquete de entrenamiento; ``tests/test_api_esquemas.py`` falla si las
listas dejan de coincidir. Los tipos de cada columna salen de la firma del modelo registrado
(``capacidad``, ``latitud`` y ``longitud`` son ``double``; los conteos, ``float``).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

# Limites de dominio: fuera de ellos el modelo no vio datos o el valor no tiene sentido
LATITUD_MIN, LATITUD_MAX = 20.4, 20.9  # Area Metropolitana de Guadalajara, con margen
LONGITUD_MIN, LONGITUD_MAX = -103.6, -103.1
CAPACIDAD_MAX = 200  # anclajes; las estaciones de MiBici tienen decenas
VIAJES_MAX = 1000.0  # viajes por estacion y hora; muy por encima de lo observado
BLOQUE_MAX = 3 * VIAJES_MAX  # viajes por estacion en un bloque de 3 h (P2)
DIA_MAX = 24 * VIAJES_MAX  # viajes por estacion en 24 h (P2)
NETO_REL_MAX = 50.0  # flujo neto / capacidad; lo observado no pasa de 7
RATIO_FDS_MAX = 50.0  # viajes diarios en fin de semana / en dias laborales; lo observado, < 3
LOG_VIAJES_MAX = 11.0  # log1p de los viajes diarios de una estacion (observado: 6.5)
MAX_INSTANCIAS = 1000  # filas por peticion; volumenes mayores, en varias peticiones o por lotes
TOLERANCIA_PERFIL = 1e-3  # los perfiles horarios suman 1 con error de redondeo

EJEMPLOS_DIR = Path(__file__).resolve().parent / "ejemplos"

# Clases de P2; las mismas de config.P2_CLASES (una prueba las compara)
CLASES_P2 = {0: "sin_riesgo", 1: "riesgo_vaciado", 2: "riesgo_llenado"}

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


def _conteo(descripcion: str, maximo: float = VIAJES_MAX):
    """Campo de viajes: numero finito entre 0 y ``maximo``."""
    return Field(ge=0, le=maximo, description=descripcion)


def _neto(descripcion: str, maximo: float = BLOQUE_MAX):
    """Campo de flujo neto (salidas - llegadas): puede ser negativo."""
    return Field(ge=-maximo, le=maximo, description=descripcion)


def _fraccion(descripcion: str):
    """Campo entre 0 y 1."""
    return Field(ge=0, le=1, description=descripcion)


def _fin_de_semana_coherente(instancia) -> None:
    """``es_fin_semana`` se deriva de ``dia_semana``; si no coinciden, la fila esta mal."""
    esperado = 1 if instancia.dia_semana >= 5 else 0
    if instancia.es_fin_semana != esperado:
        raise ValueError(
            f"es_fin_semana debe ser {esperado} cuando dia_semana es {instancia.dia_semana} "
            "(1 solo para 5 = sabado y 6 = domingo)"
        )


def _estacion_id():
    """Campo opcional que no entra al modelo: se devuelve tal cual para ubicar la fila."""
    return Field(
        default=None,
        ge=0,
        description="Opcional. No entra al modelo: se devuelve tal cual para ubicar la fila.",
    )


# =============================================================================
# Entrada
# =============================================================================
class InstanciaP1(BaseModel):
    """Una estacion en una hora del dia siguiente: las 16 variables del modelo de P1.

    Todas se conocen al cierre del dia anterior (horizonte de 24 h), igual que en el
    entrenamiento: los rezagos y medias usan solo viajes de hace 24 h o mas.
    """

    model_config = ESTRICTO

    estacion_id: int | None = _estacion_id()
    hora_dia: int = Field(ge=0, le=23, description="Hora que se predice, de 0 a 23.")
    dia_semana: int = Field(ge=0, le=6, description="Dia de la semana: 0 = lunes ... 6 = domingo.")
    es_fin_semana: int = Field(ge=0, le=1, description="1 si es sabado o domingo, 0 si no.")
    es_festivo: int = Field(ge=0, le=1, description="1 si es dia festivo oficial, 0 si no.")
    mes: int = Field(ge=1, le=12, description="Mes de la fecha que se predice, de 1 a 12.")
    # float, no int: la firma del modelo registrado la declara ``double`` (11.0 a 39.0 en los datos)
    capacidad: float = Field(gt=0, le=CAPACIDAD_MAX, description="Anclajes de la estacion.")
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
    def _coherente(self):
        _fin_de_semana_coherente(self)
        return self


class InstanciaP2(BaseModel):
    """Una estacion al inicio de un bloque de 3 h: las 19 variables del modelo de P2.

    Todas usan solo bloques ya cerrados (el anterior, el mismo bloque de ayer y de hace una
    semana, y la frecuencia de riesgo de los ultimos 7 dias). ``neto`` es salidas - llegadas del
    bloque: positivo si la estacion pierde bicicletas. Las variables ``neto_rel_*`` lo dividen
    entre la capacidad.
    """

    model_config = ESTRICTO

    estacion_id: int | None = _estacion_id()
    bloque_dia: int = Field(
        ge=0, le=21, multiple_of=3, description="Hora de inicio del bloque de 3 h: 0, 3, ..., 21."
    )
    dia_semana: int = Field(ge=0, le=6, description="Dia de la semana: 0 = lunes ... 6 = domingo.")
    es_fin_semana: int = Field(ge=0, le=1, description="1 si es sabado o domingo, 0 si no.")
    es_festivo: int = Field(ge=0, le=1, description="1 si es dia festivo oficial, 0 si no.")
    mes: int = Field(ge=1, le=12, description="Mes de la fecha que se predice, de 1 a 12.")
    capacidad: float = Field(gt=0, le=CAPACIDAD_MAX, description="Anclajes de la estacion.")
    latitud: float = Field(
        ge=LATITUD_MIN, le=LATITUD_MAX, description="Latitud de la estacion (grados)."
    )
    longitud: float = Field(
        ge=LONGITUD_MIN, le=LONGITUD_MAX, description="Longitud de la estacion (grados)."
    )
    salidas_lag_1b: float = _conteo("Salidas del bloque anterior.", BLOQUE_MAX)
    llegadas_lag_1b: float = _conteo("Llegadas del bloque anterior.", BLOQUE_MAX)
    neto_lag_1b: float = _neto("Flujo neto del bloque anterior.")
    neto_lag_dia: float = _neto("Flujo neto del mismo bloque, ayer.")
    neto_lag_semana: float = _neto("Flujo neto del mismo bloque, hace 7 dias.")
    neto_media_mismo_bloque_7d: float = _neto("Media del flujo neto del mismo bloque, 7 dias.")
    neto_rel_media_mismo_bloque_7d: float = _neto(
        "La media anterior dividida entre la capacidad.", NETO_REL_MAX
    )
    neto_rel_lag_dia: float = _neto(
        "El flujo neto de ayer dividido entre la capacidad.", NETO_REL_MAX
    )
    frac_vaciado_mismo_bloque_7d: float = _fraccion(
        "Fraccion de los ultimos 7 dias en que el mismo bloque termino en riesgo de vaciado."
    )
    frac_llenado_mismo_bloque_7d: float = _fraccion(
        "Fraccion de los ultimos 7 dias en que el mismo bloque termino en riesgo de llenado."
    )
    salidas_previas_24h: float = _conteo("Salidas en las 24 h anteriores.", DIA_MAX)

    @model_validator(mode="after")
    def _coherente(self):
        _fin_de_semana_coherente(self)
        return self


# P3: perfil horario de salidas (sal) y llegadas (lle), en dias laborales (lab) y en fines de
# semana y festivos (fds). Cada hora es la fraccion de los viajes del perfil que ocurre en ella.
PERFILES_P3 = tuple(f"{m}_{t}" for m in ("sal", "lle") for t in ("lab", "fds"))
_NOMBRES_PERFIL_P3 = {
    "sal_lab": "salidas en dias laborales",
    "sal_fds": "salidas en fines de semana y festivos",
    "lle_lab": "llegadas en dias laborales",
    "lle_fds": "llegadas en fines de semana y festivos",
}


class _BaseP3(BaseModel):
    """Reglas de ``InstanciaP3`` que no dependen de cada columna."""

    model_config = ESTRICTO

    estacion_id: int | None = _estacion_id()

    @model_validator(mode="after")
    def _perfiles_suman_uno(self):
        """Cada perfil horario es una distribucion: suma 1 (o 0 si no hubo viajes de ese tipo)."""
        for perfil in PERFILES_P3:
            suma = sum(getattr(self, f"{perfil}_h{hora:02d}") for hora in range(24))
            if min(abs(suma - 1.0), abs(suma)) > TOLERANCIA_PERFIL:
                raise ValueError(
                    f"las 24 fracciones de {perfil}_hHH deben sumar 1 (suman {suma:.4f})"
                )
        return self


_campos_p3: dict = {
    f"{perfil}_h{hora:02d}": (
        float,
        _fraccion(f"Fraccion de las {_NOMBRES_PERFIL_P3[perfil]} que ocurre a las {hora:02d} h."),
    )
    for perfil in PERFILES_P3
    for hora in range(24)
}
_campos_p3["log_viajes_dia"] = (
    float,
    Field(ge=0, le=LOG_VIAJES_MAX, description="log(1 + viajes por dia) de la estacion."),
)
_campos_p3["ratio_fin_semana"] = (
    float,
    Field(
        ge=0,
        le=RATIO_FDS_MAX,
        description="Viajes diarios en fin de semana entre viajes diarios en dias laborales.",
    ),
)
_campos_p3["balance_neto"] = (
    float,
    Field(ge=-1, le=1, description="(llegadas - salidas) / viajes: entre -1 y 1."),
)

InstanciaP3 = create_model(
    "InstanciaP3",
    __base__=_BaseP3,
    __doc__=(
        "Perfil de uso de una estacion: las 99 variables del modelo de P3 (96 fracciones "
        "horarias y 3 agregados). La ubicacion y la capacidad no entran: los grupos se forman "
        "por uso."
    ),
    **_campos_p3,
)


# Variables que recibe cada modelo, en el orden de la firma (sin estacion_id)
def _columnas(instancia: type[BaseModel]) -> list[str]:
    return [nombre for nombre in instancia.model_fields if nombre != "estacion_id"]


COLUMNAS_P1 = _columnas(InstanciaP1)
COLUMNAS_P2 = _columnas(InstanciaP2)
COLUMNAS_P3 = _columnas(InstanciaP3)
COLUMNAS = {"p1": COLUMNAS_P1, "p2": COLUMNAS_P2, "p3": COLUMNAS_P3}


def _cargar_ejemplo(nombre: str) -> dict:
    """Peticion de ejemplo de ``api/ejemplos`` (filas reales de la particion de prueba)."""
    return json.loads((EJEMPLOS_DIR / nombre).read_text(encoding="utf-8"))


EJEMPLO_PETICION_P2 = _cargar_ejemplo("peticion_p2.json")
EJEMPLO_PETICION_P3 = _cargar_ejemplo("peticion_p3.json")


def _peticion(
    nombre: str, instancia: type[BaseModel], ruta: str, ejemplo: dict
) -> type[BaseModel]:
    """Cuerpo de ``POST /predicciones/<pregunta>``: una o varias filas en ``instancias``."""
    return create_model(
        nombre,
        __config__=ConfigDict(**ESTRICTO, json_schema_extra={"examples": [ejemplo]}),
        __doc__=f"Cuerpo de ``POST {ruta}``: una o varias filas en ``instancias``.",
        instancias=(
            list[instancia],
            Field(
                min_length=1,
                max_length=MAX_INSTANCIAS,
                description=f"De 1 a {MAX_INSTANCIAS} filas; la respuesta conserva el mismo orden.",
            ),
        ),
    )


# El ejemplo aparece en /docs y llena el boton "Try it out"
PeticionP1 = _peticion("PeticionP1", InstanciaP1, "/predicciones/p1", EJEMPLO_PETICION_P1)
PeticionP2 = _peticion("PeticionP2", InstanciaP2, "/predicciones/p2", EJEMPLO_PETICION_P2)
PeticionP3 = _peticion("PeticionP3", InstanciaP3, "/predicciones/p3", EJEMPLO_PETICION_P3)


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


class PrediccionP2(BaseModel):
    """Riesgo de una fila, en el mismo orden de la peticion."""

    estacion_id: int | None = Field(default=None, description="El mismo de la peticion.")
    clase: int = Field(ge=0, le=2, description="0 = sin riesgo, 1 = vaciado, 2 = llenado.")
    riesgo: str = Field(
        description="Nombre de la clase: sin_riesgo, riesgo_vaciado o riesgo_llenado."
    )

    @model_validator(mode="after")
    def _nombre_de_la_clase(self):
        if self.riesgo != CLASES_P2[self.clase]:
            raise ValueError(
                f"la clase {self.clase} es {CLASES_P2[self.clase]!r}, no {self.riesgo!r}"
            )
        return self


class PrediccionP3(BaseModel):
    """Grupo de una fila, en el mismo orden de la peticion."""

    estacion_id: int | None = Field(default=None, description="El mismo de la peticion.")
    grupo: int = Field(ge=0, description="Grupo (cluster) de la estacion segun su perfil de uso.")


class _Respuesta(BaseModel):
    """Lo comun de las respuestas: modelo usado, numero de filas y una prediccion por fila."""

    modelo: ModeloUsado
    n: int = Field(ge=1, description="Numero de predicciones (igual al de instancias).")

    @model_validator(mode="after")
    def _una_por_instancia(self):
        if self.n != len(self.predicciones):
            raise ValueError(f"n = {self.n}, pero hay {len(self.predicciones)} predicciones")
        return self


class RespuestaP1(_Respuesta):
    """Respuesta de ``POST /predicciones/p1``."""

    model_config = ConfigDict(allow_inf_nan=False)

    predicciones: list[PrediccionP1]


class RespuestaP2(_Respuesta):
    """Respuesta de ``POST /predicciones/p2``."""

    predicciones: list[PrediccionP2]


class RespuestaP3(_Respuesta):
    """Respuesta de ``POST /predicciones/p3``."""

    predicciones: list[PrediccionP3]


class EstadoModelo(BaseModel):
    """Estado de carga del modelo de una pregunta."""

    pregunta: str
    estado: Literal["ok", "error"]
    nombre: str | None = None
    version: str | None = None
    error: str | None = Field(default=None, description="Por que no cargo, si estado es error.")


class Salud(BaseModel):
    """Respuesta de ``GET /salud``: el servicio esta arriba y con sus modelos cargados."""

    estado: Literal["ok", "degradado"] = Field(
        description="ok si cargaron los tres modelos; degradado si falta alguno (503)."
    )
    modelos: list[EstadoModelo]


class InfoModelo(BaseModel):
    """Detalle y linaje de un modelo cargado (``GET /modelo``)."""

    pregunta: str
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
