import os

# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "Dataset")
TRAIN_DIR    = os.path.join(DATA_DIR, "resized_train", "resized_train")
TEST_DIR     = os.path.join(DATA_DIR, "resized_train_cropped")
LABELS_CSV   = os.path.join(DATA_DIR, "trainLabels.csv")

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL   = os.path.join(MODELS_DIR, "best_model.h5")
LEADERBOARD  = os.path.join(MODELS_DIR, "leaderboard.csv")
TRAIN_CURVE  = os.path.join(MODELS_DIR, "training_curves.png")
CONF_MATRIX  = os.path.join(MODELS_DIR, "confusion_matrix.png")
REPORT_MD    = os.path.join(MODELS_DIR, "comparison_report.md")

# ---------- Training ----------
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
VAL_SIZE     = 0.20       # split size for validation
EPOCHS_STAGE1 = 1        # quick sweep
SEED         = 42
NUM_CLASSES  = 5
CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Patient-wise split avoids leakage between left/right eyes (recommended)
PATIENT_WISE_SPLIT = True

# Outlier screening (flag very dark images)
ENABLE_OUTLIER_SCREEN = True
DARK_MEAN_THRESHOLD = 5.0  # mean pixel (0-255) below this considered dark

# Architectures to try
ARCHS = [
    "EfficientNetB0",
    "ResNet50",
    "InceptionV3",
    "DenseNet121",
    "MobileNetV2",
    "VGG16",
    "Xception"
]

# Hyper-tune subset
TUNE_ARCHS = ["EfficientNetB0", "InceptionV3", "ResNet50"]
MAX_TUNER_TRIALS = 3  # per arch

print("EPOCHS_STAGE1 =", EPOCHS_STAGE1)