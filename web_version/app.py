"""Flask backend for the browser-based digit recognizer.

Serves a single page with a drawing <canvas>; the frontend POSTs the
drawing as a PNG data URL to /predict, which runs the same MNIST-style
preprocessing as the desktop app and returns the model's prediction.

Requires digit_model.joblib in this folder (copy it from
../desktop_version/digit_model.joblib, or train a fresh one with the
same recipe as ../desktop_version/train_model.py).
"""

import base64
import io
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.joblib")

app = Flask(__name__)
model = None


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert a black-on-white drawing into a 28x28 MNIST-style feature
    vector. Mirrors desktop_version/draw_app.py's preprocess() exactly:
    invert -> crop to ink bbox -> resize longer side to 20px -> paste
    centered by center of mass on a 28x28 canvas -> slight blur -> 0-1.
    """
    img = image.convert("L")
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox is None:
        return np.zeros((1, 784))
    img = img.crop(bbox)

    w, h = img.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=0)
    arr_small = np.array(img, dtype=np.float64)
    ys, xs = np.nonzero(arr_small > 10)
    if len(xs) > 0:
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = new_w / 2, new_h / 2
    paste_x = round(14 - cx)
    paste_y = round(14 - cy)
    canvas28.paste(img, (paste_x, paste_y))

    canvas28 = canvas28.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(canvas28, dtype=np.float64) / 255.0
    return arr.flatten().reshape(1, -1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.json.get("image", "")
    header, _, encoded = data_url.partition(",")
    if not encoded:
        return jsonify(error="No image data received"), 400

    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes))

    features = preprocess(image)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction] * 100)

    return jsonify(prediction=prediction, confidence=confidence)


def main():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Copy it from ../desktop_version/digit_model.joblib "
            "or run ../desktop_version/train_model.py and copy the result here."
        )
    model = joblib.load(MODEL_PATH)
    app.run(debug=True)


if __name__ == "__main__":
    main()
