# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The web half of the digit-recognition-study project: a Flask backend serving a single-page app with a browser `<canvas>` for drawing digits, which POSTs the drawing to a `/predict` endpoint for server-side inference with a scikit-learn model.

The desktop version lives alongside this one in `../desktop_version/` (see its own `CLAUDE.md`); the two are independent deployments and don't share code, only the same MNIST-trained-model concept and preprocessing recipe.

## Commands

Dependencies are listed in `requirements.txt` (`flask`, `numpy`, `scikit-learn`, `joblib`, `Pillow`); install with `pip install -r requirements.txt`.

Run from inside `web_version/`:

```bash
# Requires digit_model.joblib to already exist in this folder (see below)
python app.py
```

This starts the Flask dev server on `http://127.0.0.1:5000` in debug mode. There are no lint or test commands configured.

## Architecture

- **`app.py`** — Flask app. `GET /` renders `templates/index.html`; `POST /predict` accepts `{"image": "<PNG data URL>"}`, runs `preprocess()`, and returns `{"prediction": int, "confidence": float}` (confidence as a 0–100 percentage from `model.predict_proba`). Loads `digit_model.joblib` once at startup in `main()` and raises `FileNotFoundError` with instructions if it's missing.
- **`templates/index.html`** — Single-page frontend: a 280x280 `<canvas>` for mouse/touch drawing (white background, black 20px strokes), a Predict button that serializes the canvas via `toDataURL("image/png")` and POSTs it as JSON to `/predict`, and a Clear button. Pure vanilla JS, no build step.
- **`app.py`'s `preprocess()`** — Mirrors `../desktop_version/draw_app.py`'s `preprocess()` exactly: convert to grayscale → invert (canvas is black-on-white; MNIST is white-on-black) → crop to ink bounding box → resize longer side to 20px → paste onto a blank 28x28 canvas centered by **center of mass** of the ink → slight Gaussian blur → normalize to 0–1. This is a deliberate port, not a shared import — if prediction accuracy needs tuning, check this function against the desktop version's rather than reinventing it.
- **`digit_model.joblib`** — Trained model artifact (an `MLPClassifier` trained on MNIST), gitignored (not committed). Not generated in this folder — copy it from `../desktop_version/digit_model.joblib` after running `../desktop_version/train_model.py`, since both versions use the same model format and recipe.
