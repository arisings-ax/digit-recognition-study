"""Train a handwritten digit classifier and save it to disk.

Dataset: MNIST (28x28 grayscale images of digits 0-9), fetched via
scikit-learn's `fetch_openml`. MNIST is much closer in style to real
freehand mouse/pen drawings than the tiny 8x8 `load_digits` dataset, so a
model trained on it generalizes far better to hand-drawn input.

The first run downloads and caches the dataset (~50 MB) under the
scikit-learn data directory; subsequent runs reuse the cache.
"""

import os
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.joblib")

# Using a subset of MNIST keeps training time reasonable while still giving
# far better real-world accuracy than the 8x8 dataset.
TRAIN_SIZE = 15000
TEST_SIZE = 3000


def main():
    print("Fetching MNIST (this may take a while on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float64) / 255.0  # scale pixel values to 0-1
    y = mnist.target.astype(int)

    # Shuffle and take a stratified subset for faster training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        random_state=42, stratify=y,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=60,
        early_stopping=True,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
