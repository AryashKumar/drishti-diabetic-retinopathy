import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from .config import CLASS_NAMES, MODELS_DIR
import pandas as pd
import os

# --- Load Best Model Metadata ---
LEADERBOARD_PATH = os.path.join(MODELS_DIR, "leaderboard.csv")

def get_best_model_info():
    if os.path.exists(LEADERBOARD_PATH):
        df = pd.read_csv(LEADERBOARD_PATH)
        # Sort by val_accuracy descending and pick the first row
        best = df.sort_values("val_accuracy", ascending=False).iloc[0]
        return {
            "best_model": best["model"],
            "val_accuracy": float(best["val_accuracy"]),
            "model_path": best["path"]
        }
    return {"best_model": "Unknown", "val_accuracy": None, "model_path": None}

BEST_INFO = get_best_model_info()
MODEL_PATH = BEST_INFO.get("model_path") or os.path.join(MODELS_DIR, "best_model.h5")

# --- Load Keras Model ---
MODEL = load_model(MODEL_PATH)

# --- FastAPI App ---
app = FastAPI(title="Diabetic Retinopathy API")

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(224, 224))  # adjust if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    preds = MODEL.predict(img_array)
    probs = preds[0]  # shape (5,)
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]

    # --- Top-2 Predictions ---
    top2_idx = np.argsort(probs)[-2:][::-1]  # highest -> second highest
    top2 = [{ "class": CLASS_NAMES[i], "probability": float(probs[i]) } for i in top2_idx]

    # --- Build JSON ---
    response = {
        "predicted_class": predicted_class,
        "probabilities": dict(zip(CLASS_NAMES, [float(p) for p in probs])),
        "top_2": top2,
        "best_model": BEST_INFO.get("best_model"),
        "val_accuracy": BEST_INFO.get("val_accuracy"),
        "model_path": BEST_INFO.get("model_path")
    }
    return JSONResponse(response)

@app.get("/")
async def root():
    return {"message": "Diabetic Retinopathy API is running"}
