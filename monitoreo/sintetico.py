"""Datos sinteticos con la forma de P1, solo para las pruebas del monitoreo y del A/B testing.

Nunca se usan en los reportes. Imitan lo que importa para el monitoreo:

- estaciones "conocidas" desde octubre de 2024 y estaciones "nuevas" desde junio de 2026, en
  otra zona (latitud mas al norte) y con menos demanda;
- un ciclo diario, fines de semana y una temporada de lluvias (jul-sep) con menos viajes;
- en 2026 la red conocida crece 25 % (cambio de un anio a otro).

Para que las pruebas sean rapidas, solo hay un dia de cada ``cada_dias``.
"""

import numpy as np
import pandas as pd

from validacion_datos import contrato


def p1_sintetico(
    conocidas: int = 30, nuevas: int = 10, cada_dias: int = 4, semilla: int = 0
) -> pd.DataFrame:
    """Tabla con las columnas del Parquet de P1: estacion_id, hora, variables y objetivo."""
    rng = np.random.default_rng(semilla)
    dias = pd.date_range("2024-10-15", "2026-09-30", freq=f"{cada_dias}D")
    horas = (dias.values[:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    partes = []
    for estacion in range(1, conocidas + nuevas + 1):
        es_nueva = estacion > conocidas
        tiempo = pd.DatetimeIndex(horas)
        if es_nueva:
            tiempo = tiempo[tiempo >= "2026-06-01"]
        hora = tiempo.hour.to_numpy()
        perfil = 0.3 + np.exp(-((hora - 8) ** 2) / 4) * 3 + np.exp(-((hora - 18) ** 2) / 6) * 2.5
        nivel = rng.uniform(0.4, 1.0) if es_nueva else rng.uniform(1.0, 3.0)
        lluvias = np.where(tiempo.month.isin([7, 8, 9]), 0.75, 1.0)
        crecimiento = np.where((tiempo.year == 2026) & (not es_nueva), 1.25, 1.0)
        fin_semana = np.where(tiempo.dayofweek >= 5, 0.6, 1.0)
        esperado = perfil * nivel * lluvias * crecimiento * fin_semana
        n = len(tiempo)

        def conteo(escala: float = 1.0, esperado=esperado, n=n) -> np.ndarray:
            return rng.poisson(esperado * escala, n).astype("float32")

        partes.append(
            pd.DataFrame(
                {
                    "estacion_id": np.full(n, estacion, dtype="int32"),
                    "hora": tiempo.astype("datetime64[s]"),
                    "hora_dia": hora.astype("int8"),
                    "dia_semana": tiempo.dayofweek.to_numpy().astype("int8"),
                    "es_fin_semana": (tiempo.dayofweek.to_numpy() >= 5).astype("int8"),
                    "es_festivo": np.zeros(n, dtype="int8"),
                    "mes": tiempo.month.to_numpy().astype("int8"),
                    "capacidad": np.full(n, 12 if es_nueva else 20, dtype="int32"),
                    "latitud": np.full(n, (20.72 if es_nueva else 20.67) + rng.normal(0, 0.005)),
                    "longitud": np.full(n, -103.37 + rng.normal(0, 0.01)),
                    "salidas_lag_24h": conteo(),
                    "salidas_lag_48h": conteo(),
                    "salidas_lag_168h": conteo(),
                    "salidas_lag_336h": conteo(),
                    "llegadas_lag_24h": conteo(),
                    "salidas_media_misma_hora_7d": (esperado * rng.uniform(0.9, 1.1, n)).astype(
                        "float32"
                    ),
                    "salidas_media_24h_previa": np.full(n, esperado.mean(), dtype="float32"),
                    "salidas_media_7d": np.full(n, esperado.mean(), dtype="float32"),
                    "salidas": rng.poisson(esperado, n).astype("int32"),
                }
            )
        )
    datos = pd.concat(partes, ignore_index=True)
    return datos[["estacion_id", "hora", *contrato.COLUMNAS_P1, contrato.OBJETIVO_P1]]
