# src/inference.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .config import BEST_MODEL, IMAGE_SIZE, CLASS_NAMES   # CLASS_NAMES = ['0','1','2','3','4']

# -------------------------
# Load the saved model
# -------------------------
def load_best_model():
    if not os.path.exists(BEST_MODEL):
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL}")
    print(f"[INFO] Loading model from {BEST_MODEL}")
    model = tf.keras.models.load_model(BEST_MODEL)
    return model

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)   # (224, 224)
    arr = img_to_array(img) / 255.0                    # normalize
    arr = np.expand_dims(arr, axis=0)                  # add batch dim
    return arr

# -------------------------
# Run inference & display
# -------------------------
def predict_image(model, img_path):
    x = preprocess_image(img_path)
    preds = model.predict(x)
    probs = preds[0]
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx] if CLASS_NAMES else str(predicted_idx)

    print("\n[DEBUG] Raw prediction vector:", probs)
    print(f"[INFO] Predicted Class: {predicted_class} (Confidence: {probs[predicted_idx]:.4f})")

    return predicted_class, probs

def predict_single(img_path):
    """Predict on one image and return dict with class & probs"""
    model = load_best_model()
    predicted_class, probs = predict_image(model, img_path)
    return {"class": predicted_class, "probs": probs.tolist()}

def predict_pair(left_path, right_path):
    """Predict on left/right images and return dict for both."""
    model = load_best_model()
    return {
        "left": predict_single(left_path),
        "right": predict_single(right_path)
    }

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image for inference")
    args = parser.parse_args()

    # Load and predict
    model = load_best_model()
    predict_image(model, args.image)
