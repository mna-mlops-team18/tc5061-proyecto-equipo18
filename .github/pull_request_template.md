## ¿Qué cambia y por qué?

<!-- 2 o 3 líneas. Si atiende una tarea o issue, enlázalo. -->

## Tipo de cambio

- [ ] `feat`: funcionalidad nueva
- [ ] `fix`: corrección
- [ ] `docs`: documentación
- [ ] `chore` / `refactor`: configuración, estructura o limpieza sin cambiar comportamiento

## ¿Cómo se probó?

<!-- Comandos que corriste y su resultado, por ejemplo:
`pytest`, `python -m tc5061_proyecto_equipo18.dataset verify`,
`python -m tc5061_proyecto_equipo18.modeling.train --pregunta P1 --modelo baseline` -->

## Lista de verificación

- [ ] La rama sigue la convención `tipo/descripcion-corta` y los commits siguen Conventional Commits.
- [ ] No agrega datos de MiBici, modelos serializados, carpetas de MLflow (`mlruns/`, `mlflow.db`) ni credenciales (`.env`).
- [ ] Las rutas, fechas y semillas nuevas están en `config.py`, sin repetirse en otros archivos.
- [ ] Si cambian dependencias, `requirements.txt` mantiene versiones fijas (`==`).
- [ ] Las funciones nuevas tienen docstring y las variables no usan información posterior al momento de la predicción.
- [ ] Si cambian los datos, se actualizó `references/manifiesto_datos.json` y aquí se explica por qué.
- [ ] Si cambian comandos o estructura, se actualizó el README.

## Revisión

<!-- Menciona a quién le pides revisión (@usuario). La rama main exige al menos 1 aprobación. -->
