import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------
# Ensure directories exist
# ------------------------------
def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)


# ------------------------------
# Plot training curves
# ------------------------------
def plot_training_curves(history, out_path="training_curves.png", title="Training Curves"):
    hist = history.history

    acc = hist.get("accuracy", [])
    val_acc = hist.get("val_accuracy", [])
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------
# Evaluate model + confusion matrix
# ------------------------------
def evaluate_and_plots(model, val_gen, out_conf="confusion_matrix.png"):
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_conf)
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# ------------------------------
# Append experiment results
# ------------------------------
def append_leaderboard(csv_path, row_dict):
    df_new = pd.DataFrame([row_dict])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)


# ------------------------------
# Write comparison report
# ------------------------------
def write_comparison_report(leaderboard_path, report_path="comparison_report.md"):
    if not os.path.exists(leaderboard_path):
        raise FileNotFoundError("Leaderboard file not found.")

    df = pd.read_csv(leaderboard_path)

    best = df.sort_values("val_accuracy", ascending=False).iloc[0]

    report = f"""
# Model Comparison Report

## Best Model
- Model: {best['model']}
- Validation Accuracy: {best['val_accuracy']}
- Model Path: {best['path']}

## Leaderboard
{df.to_markdown(index=False)}
"""

    with open(report_path, "w") as f:
        f.write(report)

    return {
        "best_model": best["model"],
        "val_accuracy": float(best["val_accuracy"]),
        "model_path": best["path"]
    }