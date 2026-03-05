import os
import pandas as pd
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from .config import (
    MODELS_DIR, BEST_MODEL, LEADERBOARD, TRAIN_CURVE, CONF_MATRIX, REPORT_MD,
    ARCHS, TUNE_ARCHS, MAX_TUNER_TRIALS, EPOCHS_STAGE1
)
from .data import load_data_ready
from .models_zoo import build_model
from .utils import ensure_dirs, plot_training_curves, evaluate_and_plots, append_leaderboard, write_comparison_report

from .config import EPOCHS_STAGE1
print(f"[DEBUG] Using EPOCHS_STAGE1 = {EPOCHS_STAGE1}")
# ------------------------------
# Hyperparameter tuning wrapper
# ------------------------------
def tune_model(arch_name, train_gen, val_gen):
    def model_builder(hp):
        units = hp.Int("units", 128, 512, step=128)
        dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
        lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4])
        return build_model(
            arch_name,
            trainable_backbone=False,
            units=units,
            dropout=dropout,
            lr=lr
        )

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=MAX_TUNER_TRIALS,
        overwrite=True,
        directory=os.path.join(MODELS_DIR, "tuner_logs"),
        project_name=f"{arch_name}_tuning"
    )

    tuner.search(train_gen, validation_data=val_gen, epochs=EPOCHS_STAGE1, verbose=1)
    best = tuner.get_best_models(1)[0]
    return best


# ------------------------------
# Training + Comparison
# ------------------------------
def train_and_compare():
    ensure_dirs(MODELS_DIR)

    # Load/clean/split/generators + class weights
    df, train_df, val_df, train_gen, val_gen, class_weights = load_data_ready()
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # 1) Quick sweep across architectures
    for arch in ARCHS:
        print(f"\n=== Stage1: {arch} ===")
        try:
            model = build_model(arch, trainable_backbone=False, units=256, dropout=0.3, lr=1e-3)
            print("DEBUG: Model input shape =", model.input_shape)
        except Exception as e:
            print(f"[ERROR] Failed to build {arch}: {e}")
            continue

        ckpt_path = os.path.join(MODELS_DIR, f"{arch}_stage1.h5")
        cbs = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
            ModelCheckpoint(ckpt_path, save_best_only=True)
        ]

        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_STAGE1,
            class_weight=class_weights,
            verbose=1,
            callbacks=cbs
        )

        val_acc = max(hist.history.get("val_accuracy", [0.0]))
        append_leaderboard(LEADERBOARD, {
            "model": arch,
            "phase": "stage1",
            "val_accuracy": float(val_acc),
            "path": ckpt_path
        })

    # 2) Hyper-tune subset
    for arch in TUNE_ARCHS:
        print(f"\n=== Tuning: {arch} ===")
        try:
            tuned = tune_model(arch, train_gen, val_gen)
            tuned_path = os.path.join(MODELS_DIR, f"{arch}_tuned.h5")
            tuned.save(tuned_path)
            eval_res = tuned.evaluate(val_gen, verbose=0)
            val_acc = float(eval_res[1])
            append_leaderboard(LEADERBOARD, {
                "model": arch,
                "phase": "tuned",
                "val_accuracy": val_acc,
                "path": tuned_path
            })
        except Exception as e:
            print(f"[ERROR] Hyper-tuning {arch} failed: {e}")
            continue

    # 3) Select best by val_accuracy
    lb = pd.read_csv(LEADERBOARD).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    best_row = lb.iloc[0]
    print("\n=== BEST MODEL SELECTED ===")
    print(best_row)

    import shutil
    shutil.copyfile(best_row["path"], BEST_MODEL)

    # 4) Evaluate best model
    from tensorflow.keras.models import load_model
    best = load_model(BEST_MODEL)
    evaluate_and_plots(best, val_gen, out_conf=CONF_MATRIX)

    # 5) Write comparison report
    write_comparison_report(LEADERBOARD, REPORT_MD)

    # 6) Training curve placeholder
    class H: history = {"accuracy": [], "val_accuracy": []}
    plot_training_curves(H, out_path=TRAIN_CURVE, title="Training Curves (see individual runs)")

    print(f"\nArtifacts:")
    print(f"- Best model:        {BEST_MODEL}")
    print(f"- Leaderboard CSV:   {LEADERBOARD}")
    print(f"- Confusion Matrix:  {CONF_MATRIX}")
    print(f"- Report (Markdown): {REPORT_MD}")


# ------------------------------
# Main entrypoint
# ------------------------------
if __name__ == "__main__":
    # 🔹 Step 1: Debug check on ALL models before training
    print(">>> Running model debug checks...")
    for arch in ARCHS:
        try:
            print(f"\n[DEBUG] Testing {arch} build...")
            _ = build_model(arch, debug=True)   # forward pass with dummy data
            print(f"[DEBUG] {arch} build OK ✅")
        except Exception as e:
            print(f"[ERROR] {arch} build failed: {e}")

    print("\n>>> Debug build phase complete!\n")

    # 🔹 Step 2: Sanity check generators
    print(">>> Checking data generators...")
    df, train_df, val_df, train_gen, val_gen, class_weights = load_data_ready()
    batch_x, batch_y = next(iter(train_gen))
    print(f"[DEBUG] Train batch X shape: {batch_x.shape}, dtype={batch_x.dtype}, "
          f"min={batch_x.min()}, max={batch_x.max()}")
    print(f"[DEBUG] Train batch Y shape: {batch_y.shape}, sample labels={batch_y[0]}")

    batch_xv, batch_yv = next(iter(val_gen))
    print(f"[DEBUG] Val batch X shape: {batch_xv.shape}, dtype={batch_xv.dtype}, "
          f"min={batch_xv.min()}, max={batch_xv.max()}")
    print(f"[DEBUG] Val batch Y shape: {batch_yv.shape}")

    print(">>> Generator sanity check complete!\n")

    # 🔹 Step 3: Start full training pipeline
    print(">>> Starting training and comparison...")
    train_and_compare()

 