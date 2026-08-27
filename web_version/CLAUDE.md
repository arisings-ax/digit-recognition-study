# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Status: not yet implemented.** This folder is a placeholder for the web version of the digit-recognition-study project — no code exists here yet. This file records the intended architecture so implementation work starts from a consistent plan rather than being decided ad hoc.

The desktop version lives alongside this one in `../desktop_version/` (see its own `CLAUDE.md`) and is fully working; the two are independent deployments and will not share code, only the same MNIST-trained-model concept.

## Intended architecture

Server-side inference, decided over the client-side (TensorFlow.js) alternative specifically so the existing scikit-learn model can be reused as-is:

- **Backend**: Flask or FastAPI. Loads a trained `digit_model.joblib` (an `MLPClassifier` trained on MNIST — same model format as the desktop version, either the same artifact or a fresh training run using the same recipe as `../desktop_version/train_model.py`) and exposes a prediction endpoint that accepts a drawn-digit image and returns the predicted digit + confidence.
- **Frontend**: a browser `<canvas>` for mouse/touch drawing, which serializes the drawing (e.g. as a PNG data URL) and POSTs it to the backend's prediction endpoint.
- **Preprocessing**: the desktop version's `preprocess()` in `../desktop_version/draw_app.py` is the reference implementation for turning a raw drawing into an MNIST-compatible 28x28 feature vector (invert → crop to ink bbox → resize longer side to 20px → center by center-of-mass on a 28x28 canvas → slight Gaussian blur → normalize to 0–1). Whichever side of the request (browser JS vs. Python backend) ends up owning this step, port the same logic rather than reinventing it — the center-of-mass centering step in particular is what makes the model tolerant of off-center or oddly-scaled input.

## Open decisions for whoever implements this

- Flask vs. FastAPI for the backend.
- Whether the model artifact is retrained fresh here or copied from `../desktop_version/digit_model.joblib`.
- Deployment target (not yet chosen).

Once real code exists in this folder, replace this section (and the "not yet implemented" status above) with actual commands (install/run/test) and update the architecture section to describe what was actually built, the way `../desktop_version/CLAUDE.md` does for the desktop app.
