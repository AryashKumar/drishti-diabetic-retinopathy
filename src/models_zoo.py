import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50, InceptionV3, DenseNet121,
    MobileNetV2, VGG16, Xception
)
from .config import IMAGE_SIZE, NUM_CLASSES

ARCH_MAP = {
    "EfficientNetB0": EfficientNetB0,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "DenseNet121": DenseNet121,
    "MobileNetV2": MobileNetV2,
    "VGG16": VGG16,
    "Xception": Xception
}

def build_model(
    arch_name,
    trainable_backbone=False,
    units=256,
    dropout=0.3,
    lr=1e-3,
    debug=True
):
    if arch_name not in ARCH_MAP:
        raise ValueError(f"Unknown architecture: {arch_name}")

    backbone_cls = ARCH_MAP[arch_name]

    # Explicitly force RGB input
    input_shape = (*IMAGE_SIZE, 3)

    if debug:
        print(f"\n[DEBUG] Building model: {arch_name}")
        print(f"[DEBUG] Input shape set to: {input_shape}")

    # Use backbone input directly to avoid channel mismatch
    weights = "imagenet"
    if arch_name == "EfficientNetB0":
        weights = None
    backbone = backbone_cls(
    weights=weights,
    include_top=False,
    input_shape=input_shape)
    backbone.trainable = trainable_backbone

    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    m = models.Model(inputs=backbone.input, outputs=outputs)

    m.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    if debug:
        # Print model summary
        m.summary()

        # Debug: check first conv layer kernel shape
        first_conv = backbone.layers[1]  # usually "stem_conv" or "conv1"
        print(f"[DEBUG] First conv layer: {first_conv.name}")
        try:
            print(f"[DEBUG] First conv kernel shape: {first_conv.weights[0].shape}")
        except Exception:
            print("[DEBUG] Could not fetch first conv kernel shape.")

        # Test forward pass with dummy input
        dummy_x = np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)
        dummy_y = m.predict(dummy_x)
        print(f"[DEBUG] Dummy prediction shape: {dummy_y.shape}")
        print(f"[DEBUG] Dummy prediction sample: {dummy_y[0]}")

    return m
