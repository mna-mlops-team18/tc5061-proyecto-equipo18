## Que cambia y por que?

<!-- 2 o 3 lineas. Si atiende una tarea o issue, enlazalo. -->

## Tipo de cambio

- [ ] `feat`: funcionalidad nueva
- [ ] `fix`: correccion
- [ ] `docs`: documentacion
- [ ] `chore` / `refactor`: configuracion, estructura o limpieza sin cambiar comportamiento

## Como se probo?

<!-- Comandos que corriste y su resultado, por ejemplo:
`pytest`, `python -m tc5061_proyecto_equipo18.dataset verify`,
`python -m tc5061_proyecto_equipo18.modeling.train --pregunta P1 --modelo baseline` -->

## Lista de verificacion

- [ ] La rama sigue la convencion `tipo/descripcion-corta` y los commits siguen Conventional Commits.
- [ ] Codigo, documentacion, commits y esta descripcion van sin acentos ni enie (solo ASCII).
- [ ] No agrega datos de MiBici, modelos serializados, carpetas de MLflow (`mlruns/`, `mlflow.db`) ni credenciales (`.env`).
- [ ] Las rutas, fechas y semillas nuevas estan en `config.py`, sin repetirse en otros archivos.
- [ ] Si cambian dependencias, `requirements.txt` mantiene versiones fijas (`==`).
- [ ] Las funciones nuevas tienen docstring y las variables no usan informacion posterior al momento de la prediccion.
- [ ] Si cambian los datos, se actualizo `references/manifiesto_datos.json` y aqui se explica por que.
- [ ] Si cambian comandos o estructura, se actualizo el README.

## Revision

<!-- Menciona a quien le pides revision (@usuario). La rama main exige al menos 1 aprobacion. -->
