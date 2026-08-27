# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A handwritten digit recognizer, split into two independent implementations that both classify MNIST-style hand-drawn digits (0-9) but ship as separate deployments with no shared code:

- **`desktop_version/`** — working Tkinter desktop app with local scikit-learn inference. See `desktop_version/CLAUDE.md`.
- **`web_version/`** — planned browser-based app with server-side inference (Flask/FastAPI backend, canvas frontend). Not yet implemented — see `web_version/CLAUDE.md` for the intended architecture.

There is no repo-wide build system, package manifest, or test suite; each subfolder is self-contained and documents its own commands in its own `CLAUDE.md`. Read the relevant subfolder's `CLAUDE.md` before working in it — this root file only covers what spans both.

## Shared concept, independent code

Both versions are meant to classify digits the same way conceptually (train an MLP on MNIST, preprocess a raw drawing into a 28x28 MNIST-compatible feature vector via crop → resize → center-of-mass centering → blur, as implemented in `desktop_version/draw_app.py`), but they are not meant to import from each other. If the web version needs the same preprocessing logic, treat `desktop_version/draw_app.py`'s `preprocess()` as the reference to port, not a module to share.
