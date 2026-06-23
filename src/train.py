"""Train a CNN digit classifier and evaluate on train / validation / test sets."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from PIL import Image

_IMG_SIZE = 28
_RANDOM_STATE = 42


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled digit images from class subdirectories (0–9).

    Args:
        data_dir: Root directory containing subdirectories named ``0``–``9``.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``(N, 28, 28, 1)`` with values
        in ``[0, 1]`` and ``y`` has shape ``(N,)`` with integer class labels.

    Raises:
        ValueError: If no labeled images are found under ``data_dir``.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label in range(10):
        class_dir = Path(data_dir) / str(label)
        files = list(class_dir.glob("*.png")) if class_dir.exists() else []
        for f in files:
            img = Image.open(f).convert("L").resize((_IMG_SIZE, _IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            X_list.append(arr)
            y_list.append(label)

    if not X_list:
        raise ValueError(f"No labeled images found under {data_dir}")

    X = np.stack(X_list)[..., np.newaxis]  # (N, 28, 28, 1)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def build_model() -> keras.Model:
    """Return a small CNN for 10-class digit classification.

    Returns:
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(_IMG_SIZE, _IMG_SIZE, 1)),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    data_dir: str | Path = "data",
    model_path: str | Path = "models/digit_cnn.h5",
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, float]:
    """Train, validate, and test the CNN; save the best model.

    Args:
        data_dir: Root directory with labeled digit subdirectories.
        model_path: File path for the saved ``.h5`` model.
        epochs: Maximum training epochs (early stopping may end earlier).
        batch_size: Mini-batch size.

    Returns:
        Dictionary with keys ``train_acc``, ``val_acc``, ``test_acc``.
    """
    print("Loading dataset …")
    X, y = load_dataset(data_dir)
    print(f"  {len(X)} samples, class distribution: {np.bincount(y)}")

    # 70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=_RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=_RANDOM_STATE
    )

    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", save_best_only=True
        ),
    ]

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Validation report (used for hyper-parameter tuning) ---
    val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    print("\n=== Validation report ===")
    print(classification_report(y_val, val_pred, digits=4))

    # --- Test report (final, reported once) ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print("=== Test report ===")
    print(classification_report(y_test, test_pred, digits=4))
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    train_acc = max(history.history["accuracy"])
    val_acc = max(history.history["val_accuracy"])
    return {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train digit CNN.")
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="models/digit_cnn.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tf.random.set_seed(_RANDOM_STATE)
    metrics = train(args.data, args.model, args.epochs, args.batch_size)
    print(f"\nSummary: {metrics}")
