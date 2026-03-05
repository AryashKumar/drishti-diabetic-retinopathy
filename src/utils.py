# src/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tempfile, shutil, os

from .inference import load_best_model, predict_image
from .config import CLASS_NAMES, LEADERBOARD, REPORT_MD
from .metrics_utils import write_comparison_report

app = FastAPI(title="Diabetic Retinopathy Prediction API")

# Load the trained model once at startup
MODEL = load_best_model()

# Load leaderboard & get best model info
try:
    BEST_INFO = write_comparison_report(LEADERBOARD, report_path=REPORT_MD)
except Exception as e:
    print(f"[WARN] Leaderboard not found or unreadable: {e}")
    BEST_INFO = {"best_model": "unknown", "val_accuracy": 0.0, "model_path": None}


@app.get("/")
def root():
    return {"status": "API running", "docs": "/docs"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a single image and return:
    - predicted class
    - probability vector
    - best model info
    """
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    # Run prediction
    predicted_class, probs = predict_image(MODEL, temp_path)
    os.remove(temp_path)

    # Prepare JSON response
    return JSONResponse({
        "predicted_class": predicted_class,
        "probabilities": probs.tolist(),
        "class_names": CLASS_NAMES,
        "best_model": BEST_INFO.get("best_model"),
        "val_accuracy": BEST_INFO.get("val_accuracy"),
        "model_path": BEST_INFO.get("model_path")
    })
