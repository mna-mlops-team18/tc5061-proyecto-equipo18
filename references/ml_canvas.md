# ML Canvas - Proyecto MiBici (equipo 18)

Formato basado en el *Machine Learning Canvas* de Louis Dorard ([ownml.co](https://www.ownml.co/machine-learning-canvas)).
La version visual esta en [`ml_canvas.png`](ml_canvas.png) (fuente editable: [`ml_canvas.html`](ml_canvas.html), se abre en el navegador). Las decisiones detalladas y su
justificacion estan en la [propuesta](../reports/propuesta_entrega1.md).

## Propuesta de valor

Ayudar a que MiBici (Area Metropolitana de Guadalajara) tenga **bicicletas y anclajes donde y
cuando se necesitan**:

- **Operacion de la flota (P1):** anticipar cuantos viajes saldran de cada estacion el dia
  siguiente, hora por hora, para dimensionar y posicionar bicicletas.
- **Rebalanceo (P2):** alertar al inicio de cada turno que estaciones van a quedarse sin
  bicicletas o sin anclajes, para priorizar las rutas de los camiones.
- **Planeacion urbana (P3):** describir que tipos de estaciones existen (residenciales,
  de destino laboral, recreativas...) y donde se ubican, para orientar la expansion del sistema.

Usuarios: operador del sistema, equipo de rebalanceo y planeacion urbana de la AMIM.

## Tarea de prediccion

| | P1 - Regresion | P2 - Clasificacion | P3 - Clustering |
|---|---|---|---|
| **Entidad** | Estacion x hora | Estacion x bloque de 3 h | Estacion |
| **Resultado** | Numero de viajes que inician (conteo >= 0) | 0 sin riesgo - 1 riesgo de vaciarse - 2 riesgo de llenarse | Grupo de uso (k grupos) |
| **Horizonte** | 24 h (al cierre del dia anterior) | El bloque que inicia (0-3 h) | Descriptivo; se recalcula cada trimestre |
| **Espera para observar el resultado** | 1 dia | 3 h | Un trimestre (estabilidad) |

## Decisiones

- **P1:** cuantas bicicletas tener en servicio y en que estaciones dejarlas al inicio del
  dia; programar mantenimiento en horas de baja demanda.
- **P2:** a que estaciones mandar camion primero y si hay que **llevar** bicicletas
  (riesgo de vaciarse) o **retirarlas** (riesgo de llenarse).
- **P3:** donde abrir estaciones nuevas (zonas parecidas a grupos exitosos o sin cobertura),
  donde ampliar capacidad y como definir zonas de rebalanceo.

## Simulacion de impacto (evaluacion offline)

- Particion cronologica: entrenamiento oct-2024 a mar-2026, validacion abr-jun 2026,
  prueba jul-sep 2026. La prueba se usa una sola vez para reportar.
- Cada modelo se compara contra un **baseline** sencillo: rezago semanal (P1), clase mas
  frecuente del mismo bloque en los ultimos 7 dias (P2) y cuartiles de volumen (P3).
- Costo de errores: en P1 subestimar picos deja estaciones sin bicicletas (por eso RMSE
  ademas de MAE); en P2 un riesgo no detectado cuesta viajes perdidos (recall) y una falsa
  alarma cuesta un viaje de camion (precision); en P3 un grupo inestable no sirve para planear.

## Hacer predicciones

- **P1:** en lote, una vez al dia al cierre de operacion, para las 24 horas siguientes.
- **P2:** en lote al inicio de cada bloque de 3 h (8 veces al dia), en segundos.
- **P3:** en lote, al inicio de cada trimestre, con los perfiles del periodo anterior.

## Construir modelos

- Reentrenamiento mensual (P1, P2) y trimestral (P3) con la ventana mas reciente.
- Cada entrenamiento es un run de MLflow con los parametros, la version de datos y el commit
  de Git; se repite con un solo comando (`train.py` o `mlflow run` con el `MLproject`).
- Tiempo disponible: minutos en una laptop con los 24 meses (9.04 millones de viajes).

## Fuentes de datos

- Viajes mensuales del [portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/)
  (`datos_abiertos_AAAA_MM.csv`): identificador, fechas y estaciones de inicio y fin.
- Nomenclatura de estaciones (`nomenclatura_2026_09.csv`): identificador, nombre,
  coordenadas y capacidad.
- Calendario oficial de dias de descanso obligatorio en Mexico (LFT, art. 74).
- A futuro: clima (lluvia), eventos masivos, disponibilidad en tiempo real si se publica.

## Recoleccion de datos

- El portal publica un CSV por mes; la ingesta (`dataset.py`) los descarga sin pasos
  manuales y los versiona con un manifiesto (numero de registros y SHA-256 por archivo).
- Rango de trabajo: 24 meses (oct-2024 a sep-2026), configurable en `config.py`.
- Datos personales (usuario, genero, anio de nacimiento) se descartan desde la lectura.
- La etiqueta de P2 se construye a partir de los viajes y la capacidad (no hay inventario).

## Variables (features)

- **Calendario:** hora del dia, dia de la semana, fin de semana, festivo, mes.
- **Estacion:** capacidad, latitud, longitud.
- **P1 (rezagos >= 24 h):** salidas hace 1, 2, 7 y 14 dias a la misma hora; llegadas de ayer
  a la misma hora; media de la misma hora en 7 dias; medias de las ultimas 24 h y 7 dias.
- **P2 (solo bloques cerrados):** salidas, llegadas y flujo neto del bloque anterior; neto
  del mismo bloque ayer y hace una semana; media de 7 dias; frecuencia de riesgo en 7 dias;
  versiones relativas a la capacidad.
- **P3:** perfiles horarios de salidas y llegadas (laboral y fin de semana), volumen diario,
  relacion fin de semana / laboral y balance neto.

## Monitoreo

- **Modelo:** MAE y WAPE semanales (P1); recall y precision por clase (P2); ARI entre
  trimestres (P3), comparados con los valores de prueba registrados en MLflow.
- **Datos:** numero de viajes por mes y estaciones nuevas o retiradas (el manifiesto lo
  hace visible); cambios de formato en los CSV (la ingesta falla con un mensaje claro).
- **Impacto:** estaciones vacias o llenas por dia, viajes perdidos y kilometros de camion
  por bicicleta reubicada.
