import os
import cv2
import pandas as pd
from .config import TRAIN_DIR, LABELS_CSV, ENABLE_OUTLIER_SCREEN, DARK_MEAN_THRESHOLD

def read_labels(csv_path=LABELS_CSV):
    df = pd.read_csv(csv_path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    assert "image" in df.columns and "level" in df.columns, "CSV must have 'image' and 'level' columns"
    # add extension if missing
    if not df["image"].str.endswith((".jpg", ".jpeg", ".png")).any():
        df["filename"] = df["image"].astype(str) + ".jpeg"
    else:
        df["filename"] = df["image"]
    # basic checks
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    # drop rows missing label or image name
    df = df.dropna(subset=["image", "level"]).copy()
    # duplicates on filename+level (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["filename", "level"])
    print(f"Removed duplicates: {before - len(df)}")
    return df

def match_existing_files(df, image_dir=TRAIN_DIR):
    df["exists"] = df["filename"].apply(lambda f: os.path.exists(os.path.join(image_dir, f)))
    missing = df.loc[~df["exists"]]
    if not missing.empty:
        print(f"WARNING: {len(missing)} labels have no corresponding image file. They will be dropped.")
    df = df.loc[df["exists"]].drop(columns=["exists"])
    return df

def add_patient_id(df):
    # patient_id assumed as token before first underscore e.g., "10_left"
    df["patient_id"] = df["image"].astype(str).str.split("_").str[0]
    return df

def screen_dark_images(df, image_dir=TRAIN_DIR, threshold=DARK_MEAN_THRESHOLD):
    """Flag extremely dark/blank images (simple heuristic)."""
    flags = []
    sample = df.sample(n=min(2000, len(df)), random_state=42)  # sample to save time
    for fname in sample["filename"]:
        p = os.path.join(image_dir, fname)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            flags.append((fname, "corrupt"))
            continue
        mean_val = float(img.mean())
        if mean_val < threshold:
            flags.append((fname, f"dark(mean={mean_val:.2f})"))
    flagged = pd.DataFrame(flags, columns=["filename", "reason"])
    if not flagged.empty:
        print(f"Outlier screening flagged {len(flagged)} images (sampled). Not auto-dropping; logging only.")
    return flagged  # for report/logging

def prepare_clean_labels():
    df = read_labels()
    df = match_existing_files(df)
    df = add_patient_id(df)
    if ENABLE_OUTLIER_SCREEN:
        flagged = screen_dark_images(df)
    else:
        flagged = pd.DataFrame(columns=["filename", "reason"])
    print("Class distribution:\n", df["level"].value_counts().sort_index())
    return df, flagged
