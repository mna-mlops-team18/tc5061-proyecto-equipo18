"""Suites de DeepChecks sobre los conjuntos de P1, P2 y P3 y su resumen.

Corre en el ambiente de validacion (Python 3.12, ver README), desde la raiz del repo. Cada
resultado se clasifica en ``pasa``, ``falla_esperada``, ``falla_bloqueante`` o ``no_corrio``:
una comprobacion que no corrio cuenta como bloqueante, nunca se ignora en silencio. Las que
no aplican a la pregunta (p. ej. las que piden etiqueta en P3) se quitan antes de correr y se
anotan como ``no_aplica``: una omision declarada, no un choque.

Implementa la integridad de cada particion (``data_integrity``), la unicidad de la llave sobre
el archivo completo y la comparacion entrenamiento vs prueba (``train_test_validation``: deriva
y fuga temporal). La particion es cronologica, asi que la prueba (futuro) debe diferir del
entrenamiento: la deriva se mide y se explica, la fuga temporal bloquea.
"""

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path

# DeepChecks consulta su ultima version al importarse; no queremos red en la validacion
os.environ.setdefault("DISABLE_LATEST_VERSION_CHECK", "True")

import deepchecks
from deepchecks.core.check_result import CheckFailure
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureLabelCorrelation,
    FeatureLabelCorrelationChange,
    IdentifierLabelCorrelation,
    IndexTrainTestLeakage,
    LabelDrift,
    NewLabelTrainTest,
    PercentOfNulls,
)
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

from tc5061_proyecto_equipo18 import config
from validacion_datos import contrato

PASA = "pasa"
FALLA_ESPERADA = "falla_esperada"
FALLA_BLOQUEANTE = "falla_bloqueante"
NO_CORRIO = "no_corrio"
INFORMATIVO = "informativo"  # el check no tiene condicion: solo muestra informacion
NO_APLICA = "no_aplica"  # el check no corresponde a la pregunta; se quita y se declara
BLOQUEANTES = (FALLA_BLOQUEANTE, NO_CORRIO)

# Checks de data_integrity que necesitan etiqueta: P3 (clustering) no tiene
REQUIEREN_ETIQUETA = (ConflictingLabels, FeatureLabelCorrelation, IdentifierLabelCorrelation)
# Las variables ya pasaron dropna al construirse: cualquier nulo es un defecto, no ruido
MAX_PORCENTAJE_NULOS = 0.0
# Columnas que estan en el conjunto solo para la validacion, no son entrada de ningun modelo
# (estacion_id sirve a NewCategoryTrainTest). Se excluyen de la suite de integridad: una
# estacion determina su capacidad y su ubicacion, y esa correlacion no dice nada de los datos.
# Siguen cubiertas por la revision de la llave (sin nulos y unica) en el archivo completo.
COLUMNAS_SOLO_VALIDACION = ("estacion_id",)
# Umbral de la condicion por defecto de FeatureFeatureCorrelation (debe coincidir con ella)
UMBRAL_CORRELACION = 0.9
# Nombre de la "particion" de la suite que compara entrenamiento con prueba
PARTICION_DERIVA = "entrenamiento_vs_prueba"

# Umbral de la condicion por defecto de FeatureDrift (debe coincidir con ella): falla desde 0.2
UMBRAL_DERIVA = 0.2
# Importancia minima en el clasificador de dominio de MultivariateDrift para contar a una
# variable como contribuyente principal. Es nuestro, solo delimita lo aceptado: no es de DeepChecks
IMPORTANCIA_MINIMA = 0.1

# La prueba es el futuro (jul a sep 2026), asi que difiere del entrenamiento por construccion
# (oct 2024 a mar 2026). Estos motivos citan valores medidos en la corrida del 2026-10-30.
_MOTIVO_ESTACIONES_NUEVAS = (
    "Hay estaciones que aparecen en la prueba y nunca en el entrenamiento (99 de 469 en el "
    "archivo completo, conteo con pandas): son el {porcentaje} de las filas de la prueba. La red "
    "crece con el tiempo, asi que se espera. Solo se aceptan en estacion_id; si otra variable "
    "categorica trae categorias nuevas, el hallazgo vuelve a ser bloqueante. Cuando se agregue "
    "una estacion, el modelo no la vio en el entrenamiento: va a la seccion de limitaciones."
)
_MOTIVO_MES = (
    "El entrenamiento cubre los 12 meses (oct 2024 a mar 2026) y la prueba solo jul a sep, asi "
    "que la distribucion de mes difiere por construccion (Cramer's V {mes}). Es estacionalidad "
    "de la particion cronologica, no un defecto de los datos. Es la unica variable que deriva: "
    "la siguiente tiene un puntaje menor a 0.1, y la etiqueta no deriva (Label Drift {etiqueta})."
)
_MOTIVO_MULTIVARIADA = (
    "El clasificador de dominio separa entrenamiento y prueba (AUC {auc}, puntaje {puntaje}) y "
    "la unica variable que contribuye es mes (importancia 1.0), por la misma razon que Feature "
    "Drift. Sin mes, el AUC baja a {auc_sin_mes} y el puntaje a {puntaje_sin_mes}, bajo el "
    "umbral de 0.25; lo que queda es estacion_id: las estaciones nuevas, ya cubiertas arriba."
)

# Hallazgos aceptados: (pregunta, comprobacion) -> por que se espera y no bloquea.
# Se llena DESPUES de abrir el HTML de cada falla y entender su causa (no antes), y el motivo
# se copia al resumen. No se aflojan umbrales para ponerlo todo en verde.
ESPERADAS: dict[tuple[str, str], str] = {
    ("p3", "Single Value in Column"): (
        "Columnas de perfil horario de las 02 a las 04 h valen 0.0 en todas las estaciones de la "
        "particion (entrenamiento: 7 columnas; prueba: 3). Son horas de madrugada sin viajes; "
        "la causa exacta (cierre del servicio o poca actividad) no esta verificada. El conjunto "
        "cambia entre particiones: 6 de las 7 columnas constantes en entrenamiento no lo son en "
        "prueba, y 2 de las 3 de prueba no lo son en entrenamiento. Impacto medido: en prueba "
        "esas 6 columnas valen como maximo 0.006 y el StandardScaler ajustado en entrenamiento "
        "las deja sin escalar (varianza 0); frente a variables estandarizadas, de desviacion 1, "
        "es despreciable."
    ),
    ("p1", "Feature-Feature Correlation"): (
        "Las medias moviles de las 24 h y de los 7 dias previos se traslapan (las 24 h estan "
        "dentro de las 168 h; ver construir_dataset_p1), asi que su correlacion de Spearman "
        "supera 0.9 por construccion. Es redundancia entre variables, no un defecto de los "
        "datos; importa sobre todo al interpretar los coeficientes de un modelo lineal. El "
        "check usa 10,000 filas de la muestra."
    ),
    ("p2", "Feature-Feature Correlation"): (
        "Las variables relativas neto_rel_* son la version absoluta dividida entre la "
        "capacidad de la estacion (ver construir_dataset_p2), asi que quedan muy correlacionadas "
        "con ella (Spearman > 0.9) por construccion. Es redundancia entre variables, no un "
        "defecto de los datos; importa sobre todo al interpretar los coeficientes de un modelo "
        "lineal. El check usa 10,000 filas de la muestra."
    ),
    ("p1", "New Category Train Test"): _MOTIVO_ESTACIONES_NUEVAS.format(porcentaje="17.73%"),
    ("p2", "New Category Train Test"): _MOTIVO_ESTACIONES_NUEVAS.format(porcentaje="18.52%"),
    ("p1", "Feature Drift"): _MOTIVO_MES.format(mes="0.84", etiqueta="0.03"),
    ("p2", "Feature Drift"): _MOTIVO_MES.format(mes="0.84", etiqueta="0.04"),
    ("p1", "Multivariate Drift"): _MOTIVO_MULTIVARIADA.format(
        auc="0.92", puntaje="0.85", auc_sin_mes="0.58", puntaje_sin_mes="0.15"
    ),
    ("p2", "Multivariate Drift"): _MOTIVO_MULTIVARIADA.format(
        auc="0.93", puntaje="0.86", auc_sin_mes="0.58", puntaje_sin_mes="0.17"
    ),
    ("p3", "Feature Drift"): (
        "Deriva en 8 de 99 columnas: siete de perfil horario (lle_fds_h01, lle_lab_h01, "
        "lle_fds_h05, lle_lab_h05, sal_fds_h05, sal_fds_h17, sal_fds_h18) y ratio_fin_semana, "
        "con puntaje de 0.21 a 0.42 (KS). Cinco son horas de madrugada, vecinas de las h02 a h04 constantes "
        "del hallazgo de integridad. La causa no esta verificada: la prueba tiene 459 estaciones "
        "contra 370 del entrenamiento (90 nuevas y 1 que ya no esta), y el cambio de estaciones "
        "podria explicar parte. Solo se aceptan esas 8 columnas; otra que derive vuelve a ser "
        "bloqueante."
    ),
    ("p3", "Multivariate Drift"): (
        "El clasificador de dominio separa los periodos (AUC 0.83, puntaje 0.67). Contribuyen "
        "lle_fds_h01 (0.65), sal_fds_h05 (0.18) y balance_neto (0.12). Las dos primeras estan "
        "entre las 8 con deriva; balance_neto no (su deriva univariada es menor a 0.2) y solo "
        "se distingue en conjunto. La causa no esta verificada (ver Feature Drift de P3). Solo "
        "se acepta con esas tres como contribuyentes."
    ),
}
# Una falla esperada solo cubre lo que se listo: si aparece otra columna constante (p. ej.
# log_viajes_dia) u otro par correlacionado, el hallazgo vuelve a ser bloqueante en vez de
# quedar aceptado en bloque. Los pares van en orden alfabetico, como los reporta DeepChecks.
COLUMNAS_CONSTANTES_ESPERADAS: dict[tuple[str, str], set[str]] = {
    ("p3", "Single Value in Column"): {
        f"{tipo}_{dia}_h{hora:02d}"
        for tipo in ("sal", "lle")
        for dia in ("lab", "fds")
        for hora in (2, 3, 4)
    },
}
PARES_CORRELACIONADOS_ESPERADOS: dict[tuple[str, str], set[tuple[str, str]]] = {
    ("p1", "Feature-Feature Correlation"): {("salidas_media_24h_previa", "salidas_media_7d")},
    ("p2", "Feature-Feature Correlation"): {
        ("neto_lag_dia", "neto_rel_lag_dia"),
        ("neto_media_mismo_bloque_7d", "neto_rel_media_mismo_bloque_7d"),
    },
}
# Lo mismo para la deriva: el hallazgo esperado cubre solo las columnas listadas.
COLUMNAS_CON_DERIVA_ESPERADAS: dict[tuple[str, str], set[str]] = {
    ("p1", "Feature Drift"): {"mes"},
    ("p2", "Feature Drift"): {"mes"},
    ("p3", "Feature Drift"): {
        "lle_fds_h01", "lle_lab_h01", "lle_fds_h05", "lle_lab_h05",
        "sal_fds_h05", "sal_fds_h17", "sal_fds_h18", "ratio_fin_semana",
    },
}  # fmt: skip
CATEGORIAS_NUEVAS_ESPERADAS: dict[tuple[str, str], set[str]] = {
    ("p1", "New Category Train Test"): {"estacion_id"},
    ("p2", "New Category Train Test"): {"estacion_id"},
}
# MultivariateDrift solo da un puntaje; se acota con las variables que mas contribuyen
CONTRIBUYENTES_ESPERADOS: dict[tuple[str, str], set[str]] = {
    ("p1", "Multivariate Drift"): {"mes"},
    ("p2", "Multivariate Drift"): {"mes"},
    ("p3", "Multivariate Drift"): {"lle_fds_h01", "sal_fds_h05", "balance_neto"},
}


@dataclass(frozen=True)
class Hallazgo:
    pregunta: str
    suite: str
    particion: str
    comprobacion: str
    estado: str
    detalle: str
    motivo: str | None = None  # solo en fallas esperadas
    html: Path | None = None


def bloqueantes(hallazgos: list[Hallazgo]) -> list[Hallazgo]:
    """Hallazgos que detienen el entrenamiento: fallas bloqueantes y checks que no corrieron."""
    return [h for h in hallazgos if h.estado in BLOQUEANTES]


def _crear_dataset(
    df: pd.DataFrame, esp: contrato.Especificacion, excluir: tuple[str, ...] = ()
) -> Dataset:
    """Dataset de DeepChecks con las columnas del contrato, menos las de ``excluir``.

    ``excluir`` quita la columna de los datos y de las listas de variables: ``ignore_columns``
    de DeepChecks no basta, porque algunos checks siguen buscando las categoricas declaradas.
    """
    df = df.drop(columns=[c for c in excluir if c in df.columns])
    return Dataset(
        df,
        label=esp.etiqueta,
        features=[c for c in esp.caracteristicas if c not in excluir],
        cat_features=[c for c in esp.categoricas if c not in excluir],
        datetime_name=esp.columna_tiempo,
        label_type=esp.tarea,
    )


def _guardar_reporte(resultado, pregunta: str, nombre: str) -> Path:
    """HTML estatico (sin widgets de notebook). Lleva filas de datos: no se versiona."""
    ruta = config.VALIDACION_DIR / pregunta / f"{nombre}.html"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resultado.save_as_html(str(ruta), as_widget=False)
    return ruta


def _fuera_de_lo_esperado(pregunta: str, nombre: str, r) -> list:
    """Columnas o pares que fallan pero no estan en la lista de lo esperado (vacia si cubre todo)."""
    clave = (pregunta, nombre)
    if clave in COLUMNAS_CONSTANTES_ESPERADAS:
        # IsSingleValue devuelve {columna: numero de valores distintos}
        constantes = {columna for columna, n in r.value.items() if n == 1}
        return sorted(constantes - COLUMNAS_CONSTANTES_ESPERADAS[clave])
    if clave in PARES_CORRELACIONADOS_ESPERADOS:
        # FeatureFeatureCorrelation devuelve la matriz de correlaciones; misma lectura que su
        # condicion: pares por encima del umbral, sin la diagonal ni el par repetido
        altos = r.value[r.value > UMBRAL_CORRELACION].stack().index.to_list()
        pares = {(i, j) for i, j in altos if i < j}
        return sorted(pares - PARES_CORRELACIONADOS_ESPERADOS[clave])
    if clave in COLUMNAS_CON_DERIVA_ESPERADAS:
        # FeatureDrift devuelve {columna: {"Drift score": ...}}; falla desde el umbral (>=)
        derivan = {c for c, v in r.value.items() if v["Drift score"] >= UMBRAL_DERIVA}
        return sorted(derivan - COLUMNAS_CON_DERIVA_ESPERADAS[clave])
    if clave in CATEGORIAS_NUEVAS_ESPERADAS:
        # NewCategoryTrainTest devuelve un DataFrame por variable con "# New Categories"
        con_nuevas = set(r.value.index[r.value["# New Categories"] > 0])
        return sorted(con_nuevas - CATEGORIAS_NUEVAS_ESPERADAS[clave])
    if clave in CONTRIBUYENTES_ESPERADOS:
        importancias = r.value["domain_classifier_feature_importance"]
        principales = {c for c, imp in importancias.items() if imp >= IMPORTANCIA_MINIMA}
        return sorted(principales - CONTRIBUYENTES_ESPERADOS[clave])
    return []


def _clasificar(
    resultado, pregunta: str, suite: str, particion: str, html: Path
) -> list[Hallazgo]:
    """Un hallazgo por check de la suite, con el estado que le toca."""
    hallazgos = []
    for r in resultado.results:
        # get_header: el encabezado del check o, si no lo trae, el nombre de su clase
        nombre = r.get_header()
        base = {
            "pregunta": pregunta,
            "suite": suite,
            "particion": particion,
            "comprobacion": nombre,
            "html": html,
        }
        if isinstance(r, CheckFailure):
            detalle = f"{type(r.exception).__name__}: {r.exception}"
            hallazgos.append(Hallazgo(**base, estado=NO_CORRIO, detalle=detalle))
            continue
        if not r.conditions_results:
            hallazgos.append(Hallazgo(**base, estado=INFORMATIVO, detalle="sin condicion"))
            continue
        # Advertencia (WARN) tambien cuenta como no aprobado, igual que get_not_passed_checks
        fallidas = [c for c in r.conditions_results if not c.is_pass()]
        if not fallidas:
            detalle = "; ".join(c.details for c in r.conditions_results)
            hallazgos.append(Hallazgo(**base, estado=PASA, detalle=detalle))
            continue
        detalle = "; ".join(f"{c.name}: {c.details}" for c in fallidas)
        motivo = ESPERADAS.get((pregunta, nombre))
        if motivo:
            inesperadas = _fuera_de_lo_esperado(pregunta, nombre, r)
            if inesperadas:
                motivo = None
                detalle += f"; fuera de lo esperado: {inesperadas}"
        estado = FALLA_ESPERADA if motivo else FALLA_BLOQUEANTE
        hallazgos.append(Hallazgo(**base, estado=estado, detalle=detalle, motivo=motivo))
    return hallazgos


def _checks_que_no_aplican(esp: contrato.Especificacion) -> tuple[tuple, str]:
    """Clases de check que no corresponden a la pregunta y el motivo."""
    if esp.etiqueta is None:
        return REQUIEREN_ETIQUETA, "requiere etiqueta y esta pregunta no tiene (clustering)"
    if esp.tarea == "regression":
        return (ConflictingLabels,), "no aplica a regresion (la etiqueta es continua)"
    return (), ""


def _suite_integridad(esp: contrato.Especificacion, filas: int):
    """``data_integrity`` mas el porcentaje de nulos, sin los checks que no aplican a la pregunta.

    ``data_integrity`` detecta nulos escritos de distintas formas (``MixedNulls``) pero no mide
    cuantos hay: por eso se agrega ``PercentOfNulls``. Devuelve la suite y los (nombre, motivo)
    de los checks quitados.
    """
    suite = data_integrity(random_state=config.SEMILLA)
    nulos = PercentOfNulls(n_samples=filas, random_state=config.SEMILLA)
    suite.add(nulos.add_condition_percent_of_nulls_not_greater_than(MAX_PORCENTAJE_NULOS))
    clases, motivo = _checks_que_no_aplican(esp)
    quitados = []
    for indice, check in list(suite.checks.items()):
        if isinstance(check, clases):
            suite.remove(indice)
            quitados.append((check.name(), motivo))
    return suite, quitados


def _integridad(pregunta: str, particion: str, df: pd.DataFrame) -> list[Hallazgo]:
    """Integridad de una particion: nulos, tipos, duplicados y columnas constantes."""
    esp = contrato.ESPECIFICACIONES[pregunta]
    suite, quitados = _suite_integridad(esp, len(df))
    resultado = suite.run(_crear_dataset(df, esp, excluir=COLUMNAS_SOLO_VALIDACION))
    html = _guardar_reporte(resultado, pregunta, f"integridad_{particion}")
    hallazgos = _clasificar(resultado, pregunta, "integridad", particion, html)
    for nombre, motivo in quitados:
        hallazgos.append(
            Hallazgo(pregunta, "integridad", particion, nombre, NO_APLICA, motivo, html=html)
        )
    return hallazgos


def _deriva_que_no_aplica(esp: contrato.Especificacion) -> dict[type, str]:
    """Clases de check de ``train_test_validation`` que no corresponden a la pregunta y el motivo."""
    no_aplican = {
        # Los conjuntos no llevan columna de indice: la llave se revisa en el archivo completo
        # y la fuga temporal la cubre DateTrainTestLeakageOverlap. No se inventa un indice:
        # tras reset_index ambas particiones valdrian 0..N y reportarian fuga total.
        IndexTrainTestLeakage: (
            "sin columna de indice; la llave se revisa en el archivo completo y la fuga "
            "temporal con Date Train-Test Leakage"
        ),
    }
    if esp.etiqueta is None:
        sin_etiqueta = "requiere etiqueta y esta pregunta no tiene (clustering)"
        for clase in (NewLabelTrainTest, FeatureLabelCorrelationChange, LabelDrift):
            no_aplican[clase] = sin_etiqueta
    elif esp.tarea == "regression":
        no_aplican[NewLabelTrainTest] = "no aplica a regresion (la etiqueta es continua)"
    if esp.columna_tiempo is None:
        sin_tiempo = "sin columna de tiempo; la particion viene de la columna periodo"
        for clase in (DateTrainTestLeakageDuplicates, DateTrainTestLeakageOverlap):
            no_aplican[clase] = sin_tiempo
    return no_aplican


def _suite_deriva(esp: contrato.Especificacion):
    """``train_test_validation`` sin los checks que no aplican; devuelve la suite y los quitados."""
    suite = train_test_validation(random_state=config.SEMILLA)
    no_aplican = _deriva_que_no_aplica(esp)
    quitados = []
    for indice, check in list(suite.checks.items()):
        for clase, motivo in no_aplican.items():
            if isinstance(check, clase):
                suite.remove(indice)
                quitados.append((check.name(), motivo))
                break
    return suite, quitados


def _deriva(pregunta: str, entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> list[Hallazgo]:
    """Entrenamiento contra prueba: deriva, categorias nuevas y fuga temporal.

    Conserva ``estacion_id`` (a diferencia de la integridad): es lo que permite a
    NewCategoryTrainTest mostrar las estaciones que solo aparecen en la prueba.
    """
    esp = contrato.ESPECIFICACIONES[pregunta]
    suite, quitados = _suite_deriva(esp)
    resultado = suite.run(_crear_dataset(entrenamiento, esp), _crear_dataset(prueba, esp))
    html = _guardar_reporte(resultado, pregunta, f"deriva_{PARTICION_DERIVA}")
    hallazgos = _clasificar(resultado, pregunta, "deriva", PARTICION_DERIVA, html)
    for nombre, motivo in quitados:
        hallazgos.append(
            Hallazgo(pregunta, "deriva", PARTICION_DERIVA, nombre, NO_APLICA, motivo, html=html)
        )
    return hallazgos


def _llave(pregunta: str) -> list[Hallazgo]:
    """La llave (estacion y tiempo o periodo) no tiene nulos y es unica en el archivo completo."""
    esp = contrato.ESPECIFICACIONES[pregunta]
    nulos, repetidas = contrato.revisar_llave(pregunta)
    revisiones = [
        (f"Llave {esp.llave} sin nulos (archivo completo)", nulos, "celdas vacias en la llave"),
        (f"Llave {esp.llave} unica (archivo completo)", repetidas, "filas con la llave repetida"),
    ]
    hallazgos = []
    for comprobacion, cantidad, descripcion in revisiones:
        detalle = f"{cantidad:,} {descripcion}"
        motivo = ESPERADAS.get((pregunta, comprobacion)) if cantidad else None
        if cantidad == 0:
            estado = PASA
        else:
            estado = FALLA_ESPERADA if motivo else FALLA_BLOQUEANTE
        hallazgos.append(
            Hallazgo(pregunta, "integridad", "completo", comprobacion, estado, detalle, motivo)
        )
    return hallazgos


def _celda(texto: str, maximo: int = 220) -> str:
    texto = " ".join(str(texto).split()).replace("|", "/")
    return texto if len(texto) <= maximo else texto[: maximo - 3] + "..."


def escribir_resumen(
    resultados: dict[str, list[Hallazgo]],
    tamanos: dict[str, dict[str, tuple[int, int]]],
    muestra: int,
) -> Path:
    """Escribe ``reports/validacion/resumen_validacion.md``: solo metadatos, sin filas de datos."""
    lineas = [
        "# Resumen de la validacion de datos",
        "",
        f"- Version de datos: `{contrato.version_datos()}`",
        f"- Fecha: {date.today().isoformat()}",
        f"- Herramienta: DeepChecks {deepchecks.__version__} (ambiente aparte, ver README)",
        (
            f"- Muestra: hasta {muestra:,} filas por particion en P1 y P2 "
            f"(semilla {config.SEMILLA}); P3 completo"
        ),
        "- Reportes HTML: `reports/validacion/<pregunta>/` (ignorados por Git: llevan filas)",
        "",
    ]
    for pregunta, hallazgos in resultados.items():
        lineas += [f"## {pregunta.upper()}", ""]
        for particion, (usadas, total) in tamanos[pregunta].items():
            lineas.append(f"- {particion}: {usadas:,} filas validadas de {total:,}")
        conteo = {
            e: sum(h.estado == e for h in hallazgos) for e in (PASA, FALLA_ESPERADA, NO_APLICA)
        }
        lineas += [
            "",
            (
                f"Pasan: {conteo[PASA]}. Fallas esperadas: {conteo[FALLA_ESPERADA]}. "
                f"Bloqueantes: {len(bloqueantes(hallazgos))}. No aplican: {conteo[NO_APLICA]}."
            ),
            "",
            "| Suite | Particion | Comprobacion | Estado | Detalle |",
            "|---|---|---|---|---|",
        ]
        for h in hallazgos:
            lineas.append(
                f"| {h.suite} | {h.particion} | {_celda(h.comprobacion)} | {h.estado} "
                f"| {_celda(h.detalle)} |"
            )
        esperadas = [h for h in hallazgos if h.estado == FALLA_ESPERADA]
        if esperadas:
            lineas += ["", f"Fallas esperadas de {pregunta.upper()} y por que:", ""]
            lineas += [f"- **{h.comprobacion}** ({h.particion}): {h.motivo}" for h in esperadas]
        lineas.append("")
    ruta = config.RESUMEN_VALIDACION_MD
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    return ruta


def ejecutar_todo(muestra: int = contrato.MUESTRA_POR_DEFECTO) -> dict[str, list[Hallazgo]]:
    """Valida P1, P2 y P3, guarda los HTML y el resumen, y devuelve los hallazgos por pregunta."""
    resultados: dict[str, list[Hallazgo]] = {}
    tamanos: dict[str, dict[str, tuple[int, int]]] = {}
    for pregunta in contrato.ESPECIFICACIONES:
        entrenamiento, prueba = contrato.cargar(pregunta, muestra)
        particiones = {
            contrato.PARTICION_ENTRENAMIENTO: entrenamiento,
            contrato.PARTICION_PRUEBA: prueba,
        }
        tamanos[pregunta] = {
            nombre: (len(df), df.attrs["filas_particion"]) for nombre, df in particiones.items()
        }
        hallazgos = []
        for nombre, df in particiones.items():
            hallazgos += _integridad(pregunta, nombre, df)
        hallazgos += _llave(pregunta)
        hallazgos += _deriva(pregunta, entrenamiento, prueba)
        resultados[pregunta] = hallazgos
    escribir_resumen(resultados, tamanos, muestra)
    return resultados
