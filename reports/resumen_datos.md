# Resumen de los datos de MiBici

Generado con `python -m tc5061_proyecto_equipo18.dataset resumen` a partir de `references/manifiesto_datos.json`. No editar a mano.

- **Version de datos (`data_version`):** `060bfc3714aa`
- **Rango:** 2024-10 a 2026-09 (24 meses con archivo; sin archivo: ninguno)
- **Viajes:** 9,043,925 filas crudas y 9,043,167 viajes despues de la normalizacion
- **Estaciones:** 484 en la nomenclatura (470 activas; 14 retiradas o sin capacidad); 472 con al menos un viaje en el rango
- **Fuente:** https://mibici.net/es/datos-abiertos/

| Mes | Filas crudas | Sin fecha o estacion | Duplicados | Viajes limpios | Estaciones con viajes | Codificacion | Formato de fecha |
|---|---:|---:|---:|---:|---:|---|---|
| 2024-10 | 418,690 | 0 | 0 | 418,690 | 360 | cp1252 | `%d/%m/%Y %H:%M` |
| 2024-11 | 391,089 | 0 | 0 | 391,089 | 360 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2024-12 | 340,049 | 0 | 0 | 340,049 | 368 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2025-01 | 380,184 | 0 | 0 | 380,184 | 368 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2025-02 | 372,529 | 0 | 0 | 372,529 | 367 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2025-03 | 405,110 | 0 | 0 | 405,110 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-04 | 366,193 | 0 | 0 | 366,193 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-05 | 395,891 | 0 | 0 | 395,891 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-06 | 356,611 | 0 | 0 | 356,611 | 368 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-07 | 368,186 | 0 | 0 | 368,186 | 367 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2025-08 | 374,727 | 0 | 0 | 374,727 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-09 | 376,789 | 0 | 0 | 376,789 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-10 | 420,064 | 0 | 0 | 420,064 | 368 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-11 | 380,328 | 0 | 0 | 380,328 | 367 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2025-12 | 335,420 | 0 | 0 | 335,420 | 368 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-01 | 357,180 | 0 | 0 | 357,180 | 368 | cp1252 | `%Y-%m-%d %H:%M:%S` |
| 2026-02 | 333,756 | 0 | 0 | 333,756 | 368 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-03 | 376,243 | 0 | 0 | 376,243 | 369 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-04 | 351,851 | 0 | 0 | 351,851 | 368 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-05 | 315,895 | 0 | 0 | 315,895 | 371 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-06 | 355,303 | 0 | 0 | 355,303 | 427 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-07 | 388,257 | 0 | 0 | 388,257 | 453 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-08 | 393,524 | 0 | 0 | 393,524 | 467 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| 2026-09 | 490,056 | 0 | 758 | 489,298 | 468 | utf-8-sig | `%Y-%m-%d %H:%M:%S` |
| **Total** | **9,043,925** | | | **9,043,167** | | | |
