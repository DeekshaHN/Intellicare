"""Register the heart, diabetes and pneumonia models in MLflow.

Run from the PROJECT ROOT (the folder that contains app.py):
    set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python scripts\\register_models.py
"""
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

# Project root = parent of the scripts folder (works wherever you run it from)
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
os.chdir(ROOT)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("xai-medical-diagnosis")
client = mlflow.MlflowClient()

HEART = ROOT / "artifacts/heart/models/best_Heart_pipe.pkl"
DIAB_MODEL = ROOT / "artifacts/diabetes/models/best_Diabetes_Stack.pkl"
DIAB_IMP = ROOT / "artifacts/diabetes/preprocessors/diab_knn_imp.pkl"
DIAB_SCALER = ROOT / "artifacts/diabetes/preprocessors/diab_scaler.pkl"
PNEU = ROOT / "artifacts/pneumonia/models/best_pneumonia_densenet121.keras"

for p in [HEART, DIAB_MODEL, DIAB_IMP, DIAB_SCALER, PNEU]:
    if not p.exists():
        sys.exit(f"Missing file: {p}")


def promote(name: str) -> None:
    versions = client.search_model_versions(f"name='{name}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(name, "production", str(latest))
    print(f"  -> {name}: version {latest} now has alias 'production'")


# ---- Heart (sklearn pipeline) ----
print("Registering heart model...")
with mlflow.start_run(run_name="heart"):
    pipe = joblib.load(HEART)
    mlflow.sklearn.log_model(
        pipe, name="model", registered_model_name="heart-pipeline",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
promote("heart-pipeline")

# ---- Diabetes (stack model + imputer + scaler logged in the same run) ----
print("Registering diabetes model...")
with mlflow.start_run(run_name="diabetes"):
    stack = joblib.load(DIAB_MODEL)
    mlflow.sklearn.log_model(
        stack, name="model", registered_model_name="diabetes-stack",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    mlflow.log_artifact(str(DIAB_IMP), "preprocessors")
    mlflow.log_artifact(str(DIAB_SCALER), "preprocessors")
promote("diabetes-stack")

# ---- Pneumonia (Keras DenseNet121) ----
print("Registering pneumonia model...")
import tensorflow as tf  # imported late: slow to load
import mlflow.keras

with mlflow.start_run(run_name="pneumonia"):
    model = tf.keras.models.load_model(PNEU, compile=False)
    mlflow.keras.log_model(
        model, name="model", registered_model_name="pneumonia-densenet121"
    )
promote("pneumonia-densenet121")

print("\nDone. Open http://127.0.0.1:5000 and check the 'Models' tab.")
