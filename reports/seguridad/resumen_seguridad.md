# Resumen de seguridad

Revision del 31 de octubre de 2026 sobre la rama `chore/seguridad` (base `main` en `873536e`).
Se repite con `make security` desde el ambiente `.venv` (ver README > Seguridad).

| Chequeo | Herramienta | Resultado inicial | Accion | Resultado final |
|---|---|---|---|---|
| Vulnerabilidades en dependencias fijadas | `pip-audit` 2.10.1 | 1: `mlflow` 3.17.1, CVE-2026-53995 | `mlflow==3.17.2` | Sin vulnerabilidades |
| Vulnerabilidades en el ambiente completo (con transitivas) | `pip-audit --skip-editable --strict` | 12, todas en `pip` 25.1.1 | Actualizar `pip` (`make requirements`) | Sin vulnerabilidades |
| Analisis estatico del codigo | `bandit` 1.9.4 | 3 de severidad baja (B404, B603, B607) | Ruta absoluta de `git`; 2 excepciones justificadas | Sin hallazgos |
| Credenciales y archivos sensibles versionados | `tests/test_seguridad.py` | Sin hallazgos | Pruebas automaticas nuevas | 12 pruebas pasan |
| Credenciales en el historial completo de Git | `git log -p --all` con patrones | Sin hallazgos | Ninguna | Sin hallazgos |

## Dependencias (`pip-audit`)

**CVE-2026-53995 en `mlflow` 3.17.1.** Path traversal en el backend de tracking `FileStore`: un
atacante sin autenticacion podia escribir archivos fuera del directorio de trazas con un
identificador de traza manipulado en `POST /api/3.0/mlflow/traces`. Se corrige en 3.17.2.

- Impacto en el proyecto: bajo. El servidor del equipo usa SQLite
  (`--backend-store-uri sqlite:///mlflow.db`), no `FileStore`, y escucha solo en `127.0.0.1`.
- Accion: se actualizo `requirements.txt` a `mlflow==3.17.2` (version de parche, sin cambios de
  interfaz). `pyarrow==25.0.1` sigue siendo compatible. Las pruebas pasan con la nueva version.

**`pip` 25.1.1.** El `pip` que trae un ambiente recien creado tenia 12 vulnerabilidades
conocidas. No es una dependencia del proyecto, pero vive en el mismo ambiente: `make
requirements` lo actualiza antes de instalar y el README lo indica.

Despues de los cambios:

```text
$ pip-audit --skip-editable --strict
No known vulnerabilities found
```

## Codigo (`bandit`)

Los tres hallazgos estaban en `_git` (`modeling/comun.py`), que consulta el commit actual para
registrarlo en cada run de MLflow:

| Regla | Que avisa | Decision |
|---|---|---|
| B607 | Proceso iniciado con una ruta parcial (`"git"`), que depende del `PATH` | Corregido: se usa la ruta absoluta (`shutil.which("git")`) |
| B603 | `subprocess` sin shell: revisar que no reciba entrada no confiable | Excepcion `# nosec B603`: los argumentos son fijos del codigo, nunca texto del usuario |
| B404 | Importar `subprocess` | Excepcion `# nosec B404`: solo lo usa `_git` |

```text
$ bandit -r tc5061_proyecto_equipo18
Test results:
	No issues identified.
	Total potential issues skipped due to specifically being disabled (e.g., #nosec BXXX): 2
```

Las pruebas (`tests/`) no se analizan con `bandit`: usan `assert` a proposito (regla B101).

## Credenciales y archivos sensibles

`tests/test_seguridad.py` corre con `make test` y `make security`:

- `.gitignore` excluye `.env` y `.env.*`, `mlflow.db`, `mlruns/`, `mlartifacts/`, `data/`
  (datos de MiBici), Parquet y modelos serializados (`.pkl`, `.joblib`). Se comprueba con
  `git check-ignore`.
- Ningun archivo versionado es de un tipo sensible. El unico `.env*` es `.env.example`, que
  solo tiene variables comentadas.
- Ningun archivo versionado contiene llaves de nube, llaves privadas, tokens de GitHub o Slack,
  ni contrasenas asignadas. Las pruebas incluyen controles: los patrones si detectan ejemplos
  falsos y no marcan variables vacias ni codigo como `token = argumentos[i]`.

Historial completo de Git (todas las ramas), revisado una vez a mano:

```bash
git log -p --all | grep -E 'AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|github_pat_|xox[baprs]-|AIza[0-9A-Za-z_-]{35}'
git log --all --name-only --format= | sort -u | grep -iE '(^|/)\.env'
```

Sin coincidencias en el primero; el segundo solo encuentra `.env.example`.

## Datos personales

- Las columnas de usuario, genero y anio de nacimiento no se leen del CSV
  (`dataset.COLUMNAS_PERSONALES`; lo verifica `tests/test_dataset.py`). El manifiesto solo
  menciona sus nombres, no sus valores.
- El notebook de exploracion se versiona sin salidas (0 celdas con salida).
- Los artefactos de MLflow solo tienen agregados por estacion.

## Fuera de alcance

- Analisis continuo en GitHub Actions: los chequeos se corren localmente con `make security`.
- El ambiente de validacion de DeepChecks (`.venv-validacion`, PRs #23 y #24) tiene su propio
  archivo de dependencias; conviene correr `pip-audit -r requirements-validacion.txt` cuando se
  integre. `make security` ya incluye `validacion_datos/` en `bandit` cuando exista.
