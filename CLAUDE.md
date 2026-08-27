# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal, two-script handwritten digit recognizer: train an MLP classifier on MNIST, then draw a digit in a Tkinter GUI and have it predict what you drew. No package manager config, no tests, no build system — just plain Python scripts run directly.

## Commands

There is no requirements/pyproject file; dependencies (`numpy`, `scikit-learn`, `joblib`, `Pillow`) must already be present in whatever Python environment is used (the original author used an Anaconda env named `lab`, referenced in `Run Digit Recognizer.bat`).

```bash
# Train the model (fetches MNIST via fetch_openml on first run, ~50MB, then caches it)
python train_model.py

# Launch the drawing GUI (requires digit_model.joblib to already exist)
python draw_app.py
```

There are no lint or test commands configured in this repo.

## Architecture

- **`train_model.py`** — Fetches MNIST (`mnist_784` via `fetch_openml`), takes a stratified subset (15,000 train / 3,000 test — kept small for reasonable training time), trains an `MLPClassifier(256, 128)`, prints accuracy/classification report, and dumps the fitted model to `digit_model.joblib` via `joblib`. Deliberately uses MNIST rather than sklearn's built-in 8x8 `load_digits`, since MNIST's 28x28 resolution matches real freehand mouse input much better.
- **`draw_app.py`** — Tkinter GUI with a canvas for mouse-drawn digits. Maintains two parallel representations of the drawing: the on-screen `tk.Canvas` strokes, and an offscreen `PIL.Image` (`ImageDraw`) kept in sync stroke-by-stroke, which is what's actually fed to the model.
- **`digit_model.joblib`** — Committed, trained model artifact loaded directly by `draw_app.py`. Regenerate by rerunning `train_model.py` if the training code changes.
- **`Run Digit Recognizer.bat`** — Double-click launcher for Windows that calls a hardcoded `pythonw.exe` path inside a specific Anaconda env; will need editing if the environment path differs on another machine.

### The preprocessing pipeline is the crux of this app

`DigitRecognizerApp.preprocess()` in `draw_app.py` converts the raw drawing into an MNIST-compatible 28x28 feature vector, and mirrors MNIST's own construction process closely — this matters because the model is only as good as the preprocessing that adapts freehand input to what it was trained on:

1. Invert colors (canvas is black-on-white; MNIST is white-on-black).
2. Crop to the bounding box of the ink (blank canvas → all-zero vector, skips prediction).
3. Resize so the longer side is 20px, preserving aspect ratio.
4. Paste onto a blank 28x28 canvas, centered by **center of mass** of the ink (not just the bounding box center) — this specifically is what makes off-center or oddly-scaled strokes classify well.
5. Apply a slight Gaussian blur (anti-aliasing, matching MNIST's softer strokes) and normalize to 0–1.

When touching prediction accuracy issues, look here first before touching the model architecture in `train_model.py`.

### Headless/no-console robustness

`draw_app.py` is designed to also run via `pythonw.exe` (no attached console, e.g. double-clicked from the `.bat` file), where `sys.stdout`/`sys.stderr` are `None`. It guards against that at import time, and reports uncaught exceptions via `messagebox` (both at startup and via `root.report_callback_exception` for errors raised inside button callbacks) rather than letting them fail silently.
