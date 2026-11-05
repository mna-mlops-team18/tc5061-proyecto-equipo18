from pathlib import Path

import mlflow
from mlflow import MlflowClient

REPO = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{(REPO / 'mlflow.db').as_posix()}")

NOMBRE = "mibici-p1-regresion"
LOGGED_MODEL = "models:/m-f46595d0825b4656ade7d5d4c6fa0d9c"

mv = mlflow.register_model(LOGGED_MODEL, NOMBRE)
MlflowClient().set_registered_model_alias(NOMBRE, "champion", mv.version)
print(f"Registrado: {NOMBRE} versión {mv.version}, alias champion")

destino = REPO / "api" / "model"
destino.mkdir(parents=True, exist_ok=True)
mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{NOMBRE}@champion", dst_path=str(destino)
)
print(f"Modelo exportado a {destino}")
