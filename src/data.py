# src/data.py
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import TRAIN_DIR, LABELS_CSV

# ---------------------------
# Helper: Match images with available extension
# ---------------------------
def build_filename(image_id: str) -> str:
    """Return correct image file path by checking multiple extensions."""
    for ext in [".jpeg", ".jpg", ".png"]:
        path = os.path.join(TRAIN_DIR, image_id + ext)
        if os.path.exists(path):
            return path
    return None


# ---------------------------
# Load data & clean
# ---------------------------
def load_data():
    df = pd.read_csv(LABELS_CSV)
    print("Initial shape:", df.shape)

    # Ensure expected columns
    if "image" not in df.columns or "level" not in df.columns:
        raise ValueError("trainLabels.csv must have 'image' and 'level' columns")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["image"])
    print(f"Removed duplicates: {before - len(df)}")

    # Map filenames
    df["filename"] = df["image"].apply(build_filename)
    missing = df["filename"].isna().sum()
    print(f"Missing after filename match: {missing}")

    # Drop rows with missing images
    df = df.dropna(subset=["filename"])

    # Add patient id (grouping by everything before _left/_right if exists)
    df["patient_id"] = df["image"].apply(lambda x: x.split("_")[0])

    print("Final shape after cleaning:", df.shape)
    print("Class distribution:\n", df["level"].value_counts())

    return df


# ---------------------------
# Stratified split
# ---------------------------
def stratified_split(df, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(df, df["level"], groups=df["patient_id"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        return train_df, val_df


# ---------------------------
# Generators
# ---------------------------
def create_generators(train_df, val_df, img_size=(224, 224), batch_size=32):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # ✅ Convert labels to string
    train_df["level"] = train_df["level"].astype(str)
    val_df["level"] = val_df["level"].astype(str)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="level",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="level",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen



# ---------------------------
# Master function for training scripts
# ---------------------------
def load_data_ready(debug=True):
    df = load_data()
    if len(df) == 0:
        raise ValueError("No images found after cleaning. Please check dataset path & extensions.")

    train_df, val_df = stratified_split(df)

    train_gen, val_gen = create_generators(train_df, val_df)

    # Class weights (to handle imbalance)
    class_counts = train_df["level"].value_counts().to_dict()
    max_count = max(class_counts.values())
    class_weights = {int(cls): max_count / count for cls, count in class_counts.items()}

    if debug:
        print("\n[DEBUG] === Generator Sanity Check ===")
        xb, yb = next(train_gen)
        print("[DEBUG] Train batch X shape:", xb.shape)
        print("[DEBUG] Train batch Y shape:", yb.shape)
        print("[DEBUG] dtype:", xb.dtype)
        print("[DEBUG] min/max pixel:", xb.min(), xb.max())

        xv, yv = next(val_gen)
        print("[DEBUG] Val batch X shape:", xv.shape)
        print("[DEBUG] Val batch Y shape:", yv.shape)
        print("=====================================\n")

    return df, train_df, val_df, train_gen, val_gen, class_weights

# Debug run
if __name__ == "__main__":
    df, train_df, val_df, train_gen, val_gen, class_weights = load_data_ready()
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Class weights:", class_weights)
