"""Ingesta reproducible y versionada de los datos abiertos de MiBici.

Comandos (desde la raiz del repositorio, con el ambiente activado)::

    python -m tc5061_proyecto_equipo18.dataset ingest    # descarga, normaliza, Parquet y manifiesto
    python -m tc5061_proyecto_equipo18.dataset verify    # compara los datos locales con el manifiesto
    python -m tc5061_proyecto_equipo18.dataset resumen   # tamanio real de los datos para el reporte

``ingest`` no requiere pasos manuales:

1. Descarga los meses de ``config.MES_INICIO`` a ``config.MES_FIN`` y la nomenclatura de
   estaciones a ``data/raw/``. Si un archivo ya existe, no lo vuelve a bajar.
2. Detecta la codificacion (UTF-8 o Windows-1252), el delimitador y el formato de fecha de
   cada archivo, y lleva los nombres de columna a un esquema unico en ``snake_case``.
3. Descarta desde la lectura las columnas con datos personales (usuario, genero y anio de
   nacimiento): nunca llegan a los datos procesados ni a MLflow.
4. Guarda un Parquet por mes en ``data/processed/viajes/`` y las estaciones en
   ``data/processed/estaciones.parquet``.
5. Escribe ``references/manifiesto_datos.json`` con el numero de registros y el hash SHA-256
   de cada archivo crudo y de su contenido limpio. La version de los datos es un hash de esos
   hashes; es el valor que se registra en MLflow como ``data_version``.

Si el manifiesto ya existe (porque esta versionado en Git), ``ingest`` comprueba que los datos
reconstruidos sean identicos y solo lo reescribe con ``--actualizar-manifiesto``.
"""

import codecs
import csv
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as version_paquete
import json
from pathlib import Path
import platform
import re
import unicodedata

from loguru import logger
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import typer
from urllib3.util.retry import Retry

from tc5061_proyecto_equipo18 import config

app = typer.Typer(
    help="Ingesta reproducible y versionada de los datos abiertos de MiBici.",
    no_args_is_help=True,
    add_completion=False,
)

# =============================================================================
# Esquema de salida
# =============================================================================
COLUMNAS_VIAJES = ("viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id")
COLUMNAS_PERSONALES = ("usuario_id", "genero", "anio_nacimiento")
TIPOS_VIAJES = {
    "viaje_id": "int64",
    "inicio_viaje": "datetime64[s]",
    "fin_viaje": "datetime64[s]",
    "origen_id": "int32",
    "destino_id": "int32",
    "duracion_seg": "int64",
}
TIPOS_ESTACIONES = {
    "estacion_id": "int32",
    "nombre": "str",
    "latitud": "float64",
    "longitud": "float64",
    "capacidad": "int32",
    "codigo": "str",
    "municipio": "str",
    "retirada": "bool",
}

# Alias de columnas, aplicados sobre el nombre ya normalizado (sin acentos, minusculas).
# Cubren las variantes vistas en una muestra de archivos de 2014 a 2026 (incluida la enie mal
# codificada de "Anio_de_nacimiento", que aparece como "A?o") y algunos nombres alternativos
# por si el portal los cambia.
ALIAS_VIAJES = (
    (r"^(viaje_id|id_viaje|trip_id)$", "viaje_id"),
    (r"^(usuario_id|id_usuario|user_id)$", "usuario_id"),
    (r"^(genero|sexo|gender|sex)$", "genero"),
    (r"nacimiento|birth", "anio_nacimiento"),
    (r"^(inicio_del_viaje|inicio_viaje|fecha_inicio|trip_start|start_time)$", "inicio_viaje"),
    (r"^(fin_del_viaje|fin_viaje|fecha_fin|trip_end|end_time)$", "fin_viaje"),
    (r"^(origen_id|id_origen|estacion_origen|origin_id)$", "origen_id"),
    (r"^(destino_id|id_destino|estacion_destino|destination_id)$", "destino_id"),
    (r"duracion|duration", "duracion_viaje"),
)
ALIAS_ESTACIONES = (
    (r"^(id|estacion_id|id_estacion|station_id)$", "estacion_id"),
    (r"^(name|nombre|nombre_estacion)$", "nombre"),
    (r"^(latitude|latitud|lat)$", "latitud"),
    (r"^(longitude|longitud|lon|lng)$", "longitud"),
    (r"capacity|capacidad", "capacidad"),
    (r"^(obcn|codigo)$", "codigo_original"),
    (r"^(location|ubicacion)$", "ubicacion"),
    (r"^(status|estatus)$", "estatus"),
)

# Formatos de fecha vistos en el portal ("2025-06-01 00:00:08" y "01/10/2024 0:00") y
# variantes cercanas. Se elige, por archivo, el que deja mas viajes dentro del mes esperado.
FORMATOS_FECHA = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)
VALORES_NULOS = ["", "NULL", "null", "NA", "N/A", "None", "nan"]


# =============================================================================
# Utilidades de normalizacion
# =============================================================================
def meses_en_rango(mes_inicio: str, mes_fin: str) -> list[str]:
    """Devuelve los meses ``AAAA-MM`` entre ``mes_inicio`` y ``mes_fin``, ambos inclusive."""
    if pd.Period(mes_inicio, "M") > pd.Period(mes_fin, "M"):
        raise ValueError(f"El rango esta invertido: {mes_inicio} > {mes_fin}")
    return [str(periodo) for periodo in pd.period_range(mes_inicio, mes_fin, freq="M")]


def normalizar_nombre(nombre: str) -> str:
    """Lleva un nombre de columna a ``snake_case`` ASCII.

    Quita acentos y caracteres que no son letras o digitos, de modo que las variantes del
    encabezado del anio de nacimiento (con enie, con la enie rota o mal decodificada) terminen
    en nombres reconocibles por los alias (todas contienen "nacimiento").
    """
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^0-9A-Za-z]+", "_", texto).strip("_").lower()


def mapear_columnas(columnas: list[str], alias: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """Relaciona cada columna original con su nombre canonico segun ``alias``.

    Las columnas que no coinciden con ningun alias se omiten del resultado.
    """
    mapa: dict[str, str] = {}
    for original in columnas:
        normal = normalizar_nombre(original)
        for patron, canonica in alias:
            if re.search(patron, normal):
                if canonica in mapa.values():
                    raise ValueError(f"Dos columnas se interpretan como '{canonica}': {columnas}")
                mapa[original] = canonica
                break
    return mapa


def detectar_codificacion(ruta: Path, tam_bloque: int = 1 << 20) -> str:
    """Detecta si un archivo es UTF-8 (con o sin BOM) o Windows-1252.

    Algunos meses de MiBici vienen en UTF-8 y otros en Windows-1252 (por eso la enie se ve
    rota en ciertos encabezados). Se valida el archivo completo con un decodificador
    incremental para no cargarlo entero en memoria.
    """
    for codificacion in ("utf-8-sig", "cp1252"):
        decodificador = codecs.getincrementaldecoder(codificacion)()
        try:
            with ruta.open("rb") as archivo:
                while bloque := archivo.read(tam_bloque):
                    decodificador.decode(bloque)
                decodificador.decode(b"", final=True)
            return codificacion
        except UnicodeDecodeError:
            continue
    return "latin-1"  # nunca falla: cada byte es un caracter


def leer_encabezado(ruta: Path, codificacion: str) -> tuple[list[str], str]:
    """Lee la primera linea, detecta el delimitador y devuelve los nombres de columna."""
    with ruta.open(encoding=codificacion, newline="") as archivo:
        primera_linea = archivo.readline()
    delimitador = max((",", ";", "\t", "|"), key=primera_linea.count)
    columnas = next(csv.reader([primera_linea], delimiter=delimitador))
    return [columna.strip() for columna in columnas], delimitador


def elegir_formato_fecha(valores: pd.Series, mes: str | None = None) -> str:
    """Elige el formato de fecha que mejor interpreta una columna.

    Prueba cada formato de ``FORMATOS_FECHA`` sobre una muestra sistematica (determinista).
    Si se conoce el mes del archivo, gana el formato que deja mas fechas dentro de ese mes;
    asi se distingue dia/mes de mes/dia. Exige al menos 95 % de fechas validas.
    """
    muestra = valores.dropna().astype(str).str.strip()
    if muestra.empty:
        raise ValueError("La columna de fechas esta vacia")
    muestra = muestra.iloc[:: max(1, len(muestra) // 5000)]
    periodo = pd.Period(mes, "M") if mes else None

    mejor_formato, mejor_puntaje = None, -1.0
    for formato in FORMATOS_FECHA:
        fechas = pd.to_datetime(muestra, format=formato, errors="coerce")
        if periodo is not None:
            puntaje = float((fechas.dt.to_period("M") == periodo).mean())
        else:
            puntaje = float(fechas.notna().mean())
        if puntaje > mejor_puntaje:
            mejor_formato, mejor_puntaje = formato, puntaje
    if mejor_formato is None or mejor_puntaje < 0.95:
        ejemplos = muestra.head(3).tolist()
        raise ValueError(
            f"No se reconoce el formato de fecha (mejor acierto {mejor_puntaje:.1%}). "
            f"Ejemplos: {ejemplos}. Agrega el formato a FORMATOS_FECHA."
        )
    return mejor_formato


def tipos_canonicos(df: pd.DataFrame, tipos: dict[str, str]) -> pd.DataFrame:
    """Aplica los tipos del esquema (los mismos al escribir y al verificar el hash)."""
    return df.astype({columna: tipo for columna, tipo in tipos.items() if columna in df.columns})


def hash_archivo(ruta: Path) -> str:
    """SHA-256 del archivo tal como esta en disco (huella del CSV publicado)."""
    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1 << 20):
            huella.update(bloque)
    return huella.hexdigest()


def hash_contenido(df: pd.DataFrame) -> str:
    """SHA-256 del contenido de una tabla, independiente del formato de archivo.

    Usa el hash por fila de pandas mas los nombres de columna. A diferencia del hash del
    Parquet, no cambia por metadatos del escritor, asi que sirve para verificar en otra
    computadora que los datos limpios son identicos.
    """
    filas = pd.util.hash_pandas_object(df, index=False).to_numpy()
    huella = hashlib.sha256(filas.tobytes())
    huella.update("|".join(df.columns).encode("utf-8"))
    return huella.hexdigest()


def ruta_relativa(ruta: Path) -> str:
    """Ruta relativa a ``data/`` con separador ``/`` (igual en Windows y Linux)."""
    return ruta.relative_to(config.DATA_DIR).as_posix()


# =============================================================================
# Lectura y limpieza
# =============================================================================
@dataclass
class LecturaViajes:
    """Resultado de leer un CSV mensual: los datos limpios y lo que se observo del archivo."""

    datos: pd.DataFrame
    codificacion: str
    delimitador: str
    formato_fecha: str
    columnas_descartadas: list[str]
    filas_crudas: int
    descartadas_nulos: int
    descartadas_duplicados: int


def leer_viajes_crudos(ruta: Path, mes: str | None = None) -> LecturaViajes:
    """Lee un CSV mensual de viajes y lo devuelve con el esquema normalizado.

    - Normaliza los nombres de columna (acentos, mayusculas, enie mal codificada).
    - Carga solo las columnas del esquema: usuario, genero y anio de nacimiento no se leen.
    - Convierte identificadores a enteros y fechas con el formato detectado para el archivo.
    - Elimina filas sin fecha o sin estacion y viajes con identificador repetido.
    - Agrega ``duracion_seg`` y ordena por inicio para que el contenido sea canonico.
    """
    codificacion = detectar_codificacion(ruta)
    encabezado, delimitador = leer_encabezado(ruta, codificacion)
    mapa = mapear_columnas(encabezado, ALIAS_VIAJES)
    faltantes = [columna for columna in COLUMNAS_VIAJES if columna not in mapa.values()]
    if faltantes:
        raise ValueError(
            f"{ruta.name}: no se reconocen las columnas {faltantes}. "
            f"Encabezado leido: {encabezado}. Agrega el alias en ALIAS_VIAJES."
        )
    usar = [original for original, canonica in mapa.items() if canonica in COLUMNAS_VIAJES]
    descartadas = [columna for columna in encabezado if columna not in usar]

    df = pd.read_csv(
        ruta,
        sep=delimitador,
        encoding=codificacion,
        usecols=usar,
        dtype=str,
        na_values=VALORES_NULOS,
        keep_default_na=True,
    )
    df = df.rename(columns=mapa)[list(COLUMNAS_VIAJES)]
    filas_crudas = len(df)

    formato_inicio = elegir_formato_fecha(df["inicio_viaje"], mes)
    # El fin suele venir en el mismo formato; solo se busca otro si ese no funciona.
    fin_valido = df["fin_viaje"].dropna()
    tasa_mismo_formato = (
        pd.to_datetime(fin_valido.str.strip(), format=formato_inicio, errors="coerce")
        .notna()
        .mean()
    )
    formato_fin = (
        formato_inicio if tasa_mismo_formato >= 0.99 else elegir_formato_fecha(df["fin_viaje"])
    )
    df["inicio_viaje"] = pd.to_datetime(
        df["inicio_viaje"].str.strip(), format=formato_inicio, errors="coerce"
    )
    df["fin_viaje"] = pd.to_datetime(
        df["fin_viaje"].str.strip(), format=formato_fin, errors="coerce"
    )
    for columna in ("viaje_id", "origen_id", "destino_id"):
        df[columna] = pd.to_numeric(df[columna].str.strip(), errors="coerce")

    nulos = df.isna().any(axis=1)
    df = df.loc[~nulos]
    duplicados = df.duplicated(subset="viaje_id", keep="first")
    df = df.loc[~duplicados].copy()
    df["duracion_seg"] = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds().round()
    df = tipos_canonicos(df, TIPOS_VIAJES)
    df = df.sort_values(["inicio_viaje", "viaje_id"], kind="stable").reset_index(drop=True)

    formato = (
        formato_inicio if formato_inicio == formato_fin else f"{formato_inicio} | {formato_fin}"
    )
    return LecturaViajes(
        datos=df,
        codificacion=codificacion,
        delimitador=delimitador,
        formato_fecha=formato,
        columnas_descartadas=descartadas,
        filas_crudas=filas_crudas,
        descartadas_nulos=int(nulos.sum()),
        descartadas_duplicados=int(duplicados.sum()),
    )


def leer_estaciones_crudas(ruta: Path) -> tuple[pd.DataFrame, str]:
    """Lee la nomenclatura de estaciones y la normaliza.

    Ademas de identificador, nombre, coordenadas y capacidad, extrae del nombre el codigo
    de la estacion y el municipio, p. ej. "(GDL-001) C. Epigmenio..." -> ``GDL-001`` y
    ``GDL``. Marca como ``retirada`` la estacion con prefijo "ex-" o capacidad 0.
    """
    codificacion = detectar_codificacion(ruta)
    encabezado, delimitador = leer_encabezado(ruta, codificacion)
    mapa = mapear_columnas(encabezado, ALIAS_ESTACIONES)
    faltantes = [
        c for c in ("estacion_id", "nombre", "latitud", "longitud") if c not in mapa.values()
    ]
    if faltantes:
        raise ValueError(f"{ruta.name}: faltan columnas {faltantes}. Encabezado: {encabezado}")

    df = pd.read_csv(
        ruta,
        sep=delimitador,
        encoding=codificacion,
        usecols=list(mapa),
        dtype=str,
        na_values=VALORES_NULOS,
    ).rename(columns=mapa)
    for columna in ("estacion_id", "latitud", "longitud"):
        df[columna] = pd.to_numeric(df[columna].str.strip(), errors="coerce")
    if "capacidad" in df.columns:
        df["capacidad"] = pd.to_numeric(df["capacidad"].str.strip(), errors="coerce").fillna(0)
    else:
        logger.warning("La nomenclatura no trae capacidad; P2 no podra etiquetar el riesgo.")
        df["capacidad"] = 0

    df["nombre"] = df["nombre"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    partes = df["nombre"].str.extract(
        r"^\(\s*(?P<ex>ex-)?(?P<municipio>[A-Za-z]{2,4})-(?P<numero>\d+)\s*\)",
        flags=re.IGNORECASE,
    )
    municipio = partes["municipio"].str.upper()
    df["municipio"] = municipio.fillna("ND")
    df["codigo"] = (municipio + "-" + partes["numero"]).fillna("ND")
    df["retirada"] = (
        partes["ex"].notna()
        | df["nombre"].str.contains(r"(?:^|\W)ex-", case=False, regex=True)
        | (df["capacidad"] <= 0)
    )

    df = df.dropna(subset=["estacion_id", "latitud", "longitud"])
    df = df.drop_duplicates(subset="estacion_id", keep="first")
    df = tipos_canonicos(df[list(TIPOS_ESTACIONES)], TIPOS_ESTACIONES)
    return df.sort_values("estacion_id").reset_index(drop=True), codificacion


# =============================================================================
# Descarga
# =============================================================================
def crear_sesion_http() -> requests.Session:
    """Sesion HTTP con reintentos automaticos ante errores temporales del servidor."""
    reintentos = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    sesion = requests.Session()
    sesion.mount("https://", HTTPAdapter(max_retries=reintentos))
    sesion.mount("http://", HTTPAdapter(max_retries=reintentos))
    sesion.headers["User-Agent"] = config.USER_AGENT
    return sesion


def descargar(url: str, destino: Path, sesion: requests.Session, forzar: bool = False) -> bool:
    """Descarga ``url`` en ``destino`` (via archivo temporal). Devuelve si hubo descarga."""
    if destino.exists() and not forzar:
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".part")
    with sesion.get(url, stream=True, timeout=(15, 300)) as respuesta:
        respuesta.raise_for_status()
        if "text/html" in respuesta.headers.get("content-type", ""):
            raise RuntimeError(
                f"{url} devolvio una pagina HTML en lugar de un CSV. "
                f"Revisa el enlace en config.RUTAS_VIAJES contra {config.URL_PORTAL_DATOS}"
            )
        total = int(respuesta.headers.get("content-length", 0)) or None
        with (
            temporal.open("wb") as archivo,
            tqdm(total=total, unit="B", unit_scale=True, desc=destino.name, leave=False) as barra,
        ):
            for bloque in respuesta.iter_content(chunk_size=1 << 20):
                archivo.write(bloque)
                barra.update(len(bloque))
    temporal.replace(destino)
    return True


# =============================================================================
# Manifiesto (versionado de datos)
# =============================================================================
@dataclass
class RegistroMes:
    """Huella de un mes de viajes: de donde vino, que se leyo y como quedo."""

    mes: str
    url: str
    archivo_crudo: str
    sha256_crudo: str
    bytes_crudo: int
    codificacion: str
    delimitador: str
    formato_fecha: str
    columnas_descartadas: list[str]
    filas_crudas: int
    descartadas_nulos: int
    descartadas_duplicados: int
    filas_limpias: int
    primer_inicio: str
    ultimo_inicio: str
    estaciones_con_viajes: int
    archivo_procesado: str
    sha256_contenido: str


@dataclass
class RegistroEstaciones:
    """Huella de la nomenclatura de estaciones."""

    url: str
    archivo_crudo: str
    sha256_crudo: str
    bytes_crudo: int
    codificacion: str
    estaciones: int
    estaciones_activas: int
    estaciones_retiradas: int
    archivo_procesado: str
    sha256_contenido: str


@dataclass
class Manifiesto:
    """Manifiesto versionado en Git que identifica exactamente los datos usados.

    La ``version`` es un hash del rango y del contenido limpio de cada archivo: si cambia
    un solo viaje, cambia la version. Ese valor se registra en cada run de MLflow.
    """

    mes_inicio: str
    mes_fin: str
    estaciones: RegistroEstaciones
    meses: list[RegistroMes]
    meses_sin_archivo: list[str] = field(default_factory=list)
    estaciones_con_viajes: int = 0
    dataset: str = "mibici-viajes"
    fuente: str = config.URL_PORTAL_DATOS
    entorno: dict[str, str] = field(default_factory=dict)
    version: str = ""

    @property
    def total_viajes(self) -> int:
        return sum(registro.filas_limpias for registro in self.meses)

    def calcular_version(self) -> str:
        huella = {
            "rango": [self.mes_inicio, self.mes_fin],
            "estaciones": self.estaciones.sha256_contenido,
            "meses": {registro.mes: registro.sha256_contenido for registro in self.meses},
        }
        texto = json.dumps(huella, sort_keys=True).encode("utf-8")
        return hashlib.sha256(texto).hexdigest()[:12]

    def registro(self, mes: str) -> RegistroMes | None:
        return next((registro for registro in self.meses if registro.mes == mes), None)

    def diferencias(self, otro: "Manifiesto") -> list[str]:
        """Diferencias de datos (no de entorno) entre dos manifiestos."""
        difs = []
        if (self.mes_inicio, self.mes_fin) != (otro.mes_inicio, otro.mes_fin):
            difs.append(
                f"rango {self.mes_inicio}..{self.mes_fin} vs {otro.mes_inicio}..{otro.mes_fin}"
            )
        if self.estaciones.sha256_contenido != otro.estaciones.sha256_contenido:
            difs.append("nomenclatura de estaciones con contenido distinto")
        propios = {registro.mes: registro for registro in self.meses}
        ajenos = {registro.mes: registro for registro in otro.meses}
        for mes in sorted(set(propios) | set(ajenos)):
            a, b = propios.get(mes), ajenos.get(mes)
            if a is None or b is None:
                difs.append(f"{mes}: aparece solo en uno de los manifiestos")
            elif a.sha256_crudo != b.sha256_crudo:
                difs.append(f"{mes}: el CSV del portal cambio (SHA-256 crudo distinto)")
            elif a.sha256_contenido != b.sha256_contenido or a.filas_limpias != b.filas_limpias:
                difs.append(
                    f"{mes}: contenido limpio distinto ({a.filas_limpias} vs {b.filas_limpias})"
                )
        return difs

    def a_dict(self) -> dict:
        datos = asdict(self)
        datos["total_viajes"] = self.total_viajes
        orden = [
            "dataset", "version", "fuente", "mes_inicio", "mes_fin", "total_viajes",
            "estaciones_con_viajes", "meses_sin_archivo", "entorno", "estaciones", "meses",
        ]  # fmt: skip
        return {clave: datos[clave] for clave in orden}

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        texto = json.dumps(self.a_dict(), indent=2)  # ASCII: lo que no lo es va como \uXXXX
        ruta.write_text(texto + "\n", encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path = config.MANIFIESTO_DATOS) -> "Manifiesto":
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. Ejecuta: python -m tc5061_proyecto_equipo18.dataset ingest"
            )
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        datos.pop("total_viajes", None)
        datos["estaciones"] = RegistroEstaciones(**datos["estaciones"])
        datos["meses"] = [RegistroMes(**registro) for registro in datos["meses"]]
        return cls(**datos)


def describir_entorno() -> dict[str, str]:
    """Versiones con las que se generaron los datos (informativo, no entra en la version)."""
    entorno = {"python": platform.python_version()}
    for paquete in ("pandas", "numpy", "pyarrow"):
        try:
            entorno[paquete] = version_paquete(paquete)
        except PackageNotFoundError:
            entorno[paquete] = "no instalado"
    return entorno


# =============================================================================
# Pasos de la ingesta
# =============================================================================
def procesar_estaciones(sesion: requests.Session, forzar_descarga: bool) -> RegistroEstaciones:
    """Descarga, normaliza y guarda la nomenclatura de estaciones."""
    crudo = config.RAW_DATA_DIR / Path(config.RUTA_ESTACIONES).name
    descargar(config.URL_BASE_ARCHIVOS + config.RUTA_ESTACIONES, crudo, sesion, forzar_descarga)
    estaciones, codificacion = leer_estaciones_crudas(crudo)
    config.ESTACIONES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    estaciones.to_parquet(config.ESTACIONES_PARQUET, index=False, compression="zstd")
    retiradas = int(estaciones["retirada"].sum())
    return RegistroEstaciones(
        url=config.URL_BASE_OFICIAL + config.RUTA_ESTACIONES,
        archivo_crudo=ruta_relativa(crudo),
        sha256_crudo=hash_archivo(crudo),
        bytes_crudo=crudo.stat().st_size,
        codificacion=codificacion,
        estaciones=len(estaciones),
        estaciones_activas=len(estaciones) - retiradas,
        estaciones_retiradas=retiradas,
        archivo_procesado=ruta_relativa(config.ESTACIONES_PARQUET),
        sha256_contenido=hash_contenido(estaciones),
    )


def procesar_mes(
    mes: str, sesion: requests.Session, forzar_descarga: bool, previo: RegistroMes | None
) -> tuple[RegistroMes, set[int]]:
    """Descarga y procesa un mes. Si el CSV y el Parquet no cambiaron, reutiliza el registro."""
    ruta_portal = config.RUTAS_VIAJES[mes]
    crudo = config.RAW_VIAJES_DIR / Path(ruta_portal).name
    procesado = config.ruta_viajes_mes(mes)
    descargar(config.URL_BASE_ARCHIVOS + ruta_portal, crudo, sesion, forzar_descarga)
    sha_crudo = hash_archivo(crudo)

    if previo is not None and previo.sha256_crudo == sha_crudo and procesado.exists():
        df = tipos_canonicos(pd.read_parquet(procesado), TIPOS_VIAJES)
        if hash_contenido(df) == previo.sha256_contenido:
            estaciones = set(df["origen_id"].unique()) | set(df["destino_id"].unique())
            return previo, {int(e) for e in estaciones}

    lectura = leer_viajes_crudos(crudo, mes)
    df = lectura.datos
    procesado.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(procesado, index=False, compression="zstd")
    estaciones = {int(e) for e in set(df["origen_id"].unique()) | set(df["destino_id"].unique())}
    registro = RegistroMes(
        mes=mes,
        url=config.URL_BASE_OFICIAL + ruta_portal,
        archivo_crudo=ruta_relativa(crudo),
        sha256_crudo=sha_crudo,
        bytes_crudo=crudo.stat().st_size,
        codificacion=lectura.codificacion,
        delimitador=lectura.delimitador,
        formato_fecha=lectura.formato_fecha,
        columnas_descartadas=lectura.columnas_descartadas,
        filas_crudas=lectura.filas_crudas,
        descartadas_nulos=lectura.descartadas_nulos,
        descartadas_duplicados=lectura.descartadas_duplicados,
        filas_limpias=len(df),
        primer_inicio=str(df["inicio_viaje"].min()) if len(df) else "",
        ultimo_inicio=str(df["inicio_viaje"].max()) if len(df) else "",
        estaciones_con_viajes=len(estaciones),
        archivo_procesado=ruta_relativa(procesado),
        sha256_contenido=hash_contenido(df),
    )
    return registro, estaciones


# =============================================================================
# Comandos
# =============================================================================
@app.command()
def ingest(
    mes_inicio: str = typer.Option(config.MES_INICIO, help="Primer mes AAAA-MM (inclusive)."),
    mes_fin: str = typer.Option(config.MES_FIN, help="Ultimo mes AAAA-MM (inclusive)."),
    forzar_descarga: bool = typer.Option(False, help="Vuelve a bajar los CSV aunque existan."),
    actualizar_manifiesto: bool = typer.Option(
        False, help="Reescribe el manifiesto aunque los datos difieran del versionado."
    ),
):
    """Descarga, normaliza y versiona los datos del rango configurado."""
    meses = meses_en_rango(mes_inicio, mes_fin)
    anterior = Manifiesto.cargar() if config.MANIFIESTO_DATOS.exists() else None
    logger.info(f"Ingesta MiBici {mes_inicio} a {mes_fin}: {len(meses)} meses")

    sesion = crear_sesion_http()
    registro_estaciones = procesar_estaciones(sesion, forzar_descarga)
    logger.info(f"Estaciones: {registro_estaciones.estaciones} en la nomenclatura")

    registros, sin_archivo, estaciones_vistas = [], [], set()
    for mes in tqdm(meses, desc="Meses", unit="mes"):
        if config.RUTAS_VIAJES.get(mes) is None:
            sin_archivo.append(mes)
            logger.warning(f"{mes}: el portal no publica archivo; se omite")
            continue
        previo = anterior.registro(mes) if anterior else None
        registro, estaciones = procesar_mes(mes, sesion, forzar_descarga, previo)
        registros.append(registro)
        estaciones_vistas |= estaciones

    manifiesto = Manifiesto(
        mes_inicio=mes_inicio,
        mes_fin=mes_fin,
        estaciones=registro_estaciones,
        meses=registros,
        meses_sin_archivo=sin_archivo,
        estaciones_con_viajes=len(estaciones_vistas),
        entorno=describir_entorno(),
    )
    manifiesto.version = manifiesto.calcular_version()

    if anterior is not None:
        diferencias = anterior.diferencias(manifiesto)
        if not diferencias:
            logger.success(
                f"Datos reconstruidos identicos al manifiesto versionado "
                f"(version {manifiesto.version}, {manifiesto.total_viajes:,} viajes)."
            )
            return
        if not actualizar_manifiesto:
            detalle = "\n  - ".join(diferencias)
            logger.error(
                f"Los datos difieren del manifiesto versionado:\n  - {detalle}\n"
                "Si el cambio es intencional (p. ej. un nuevo rango de fechas), vuelve a correr "
                "con --actualizar-manifiesto y sube el manifiesto nuevo en un Pull Request."
            )
            raise typer.Exit(code=1)

    manifiesto.guardar(config.MANIFIESTO_DATOS)
    logger.success(
        f"Manifiesto guardado en {config.MANIFIESTO_DATOS.relative_to(config.PROJ_ROOT)}: "
        f"version {manifiesto.version}, {len(registros)} meses, "
        f"{manifiesto.total_viajes:,} viajes, {manifiesto.estaciones_con_viajes} estaciones con viajes."
    )


@app.command()
def verify(
    revisar_crudos: bool = typer.Option(True, help="Tambien compara el hash de los CSV crudos."),
):
    """Comprueba que los datos locales coinciden exactamente con el manifiesto versionado."""
    manifiesto = Manifiesto.cargar()
    problemas: list[str] = []

    def revisar(nombre, archivo_crudo, sha_crudo, archivo_procesado, sha_contenido, tipos, filas):
        crudo = config.DATA_DIR / archivo_crudo
        if revisar_crudos:
            if not crudo.exists():
                problemas.append(f"{nombre}: falta {archivo_crudo}")
            elif hash_archivo(crudo) != sha_crudo:
                problemas.append(f"{nombre}: el CSV crudo no coincide (SHA-256)")
        procesado = config.DATA_DIR / archivo_procesado
        if not procesado.exists():
            problemas.append(f"{nombre}: falta {archivo_procesado}")
            return
        df = tipos_canonicos(pd.read_parquet(procesado), tipos)
        if filas is not None and len(df) != filas:
            problemas.append(f"{nombre}: {len(df)} filas en lugar de {filas}")
        elif hash_contenido(df) != sha_contenido:
            problemas.append(f"{nombre}: el contenido limpio no coincide (SHA-256)")

    est = manifiesto.estaciones
    revisar(
        "estaciones", est.archivo_crudo, est.sha256_crudo, est.archivo_procesado,
        est.sha256_contenido, TIPOS_ESTACIONES, est.estaciones,
    )  # fmt: skip
    for registro in tqdm(manifiesto.meses, desc="Verificando", unit="mes"):
        revisar(
            registro.mes, registro.archivo_crudo, registro.sha256_crudo,
            registro.archivo_procesado, registro.sha256_contenido, TIPOS_VIAJES,
            registro.filas_limpias,
        )  # fmt: skip

    if problemas:
        logger.error(
            "Los datos locales NO coinciden con el manifiesto:\n  - " + "\n  - ".join(problemas)
        )
        raise typer.Exit(code=1)
    logger.success(
        f"Datos verificados: version {manifiesto.version}, {len(manifiesto.meses)} meses, "
        f"{manifiesto.total_viajes:,} viajes. Coinciden con el manifiesto."
    )


@app.command()
def resumen():
    """Escribe reports/resumen_datos.md con el tamanio real de los datos (desde el manifiesto)."""
    manifiesto = Manifiesto.cargar()
    est = manifiesto.estaciones
    filas = [
        f"| {r.mes} | {r.filas_crudas:,} | {r.descartadas_nulos:,} | {r.descartadas_duplicados:,} "
        f"| {r.filas_limpias:,} | {r.estaciones_con_viajes} | {r.codificacion} | `{r.formato_fecha}` |"
        for r in manifiesto.meses
    ]
    total_crudas = sum(r.filas_crudas for r in manifiesto.meses)
    sin_archivo = ", ".join(manifiesto.meses_sin_archivo) or "ninguno"
    texto = "\n".join(
        [
            "# Resumen de los datos de MiBici",
            "",
            (
                "Generado con `python -m tc5061_proyecto_equipo18.dataset resumen` a partir de "
                "`references/manifiesto_datos.json`. No editar a mano."
            ),
            "",
            f"- **Version de datos (`data_version`):** `{manifiesto.version}`",
            (
                f"- **Rango:** {manifiesto.mes_inicio} a {manifiesto.mes_fin} "
                f"({len(manifiesto.meses)} meses con archivo; sin archivo: {sin_archivo})"
            ),
            (
                f"- **Viajes:** {total_crudas:,} filas crudas y {manifiesto.total_viajes:,} viajes "
                "despues de la normalizacion"
            ),
            (
                f"- **Estaciones:** {est.estaciones} en la nomenclatura ({est.estaciones_activas} "
                f"activas; {est.estaciones_retiradas} retiradas o sin capacidad); "
                f"{manifiesto.estaciones_con_viajes} con al menos un viaje en el rango"
            ),
            f"- **Fuente:** {manifiesto.fuente}",
            "",
            (
                "| Mes | Filas crudas | Sin fecha o estacion | Duplicados | Viajes limpios "
                "| Estaciones con viajes | Codificacion | Formato de fecha |"
            ),
            "|---|---:|---:|---:|---:|---:|---|---|",
            *filas,
            f"| **Total** | **{total_crudas:,}** | | | **{manifiesto.total_viajes:,}** | | | |",
            "",
        ]
    )
    config.RESUMEN_DATOS_MD.parent.mkdir(parents=True, exist_ok=True)
    config.RESUMEN_DATOS_MD.write_text(texto, encoding="utf-8")
    logger.success(f"Resumen escrito en {config.RESUMEN_DATOS_MD.relative_to(config.PROJ_ROOT)}")
    print(texto)


if __name__ == "__main__":
    app()
