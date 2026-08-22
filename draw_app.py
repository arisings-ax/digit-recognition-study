"""A small GUI app to draw a digit by hand with the mouse and have the
trained model recognize it.

Run `train_model.py` first to create `digit_model.joblib`.
"""

import os
import sys
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import tkinter as tk
from tkinter import messagebox

# When launched via pythonw.exe (no console attached), sys.stdout/stderr
# are None. Anything that tries to write to them -- including tkinter's
# default callback-exception handler -- would then crash with a confusing
# "NoneType object has no attribute 'write'" error, masking the real
# problem. Redirect them to a null stream so writes are simply discarded.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.joblib")
CANVAS_SIZE = 280  # on-screen drawing area (pixels)
BRUSH_RADIUS = 10


class DigitRecognizerApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        # Canvas where the user draws: black strokes on a white background
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)

        # Offscreen image kept in sync with the canvas, used for prediction
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        predict_btn = tk.Button(root, text="Predict", command=self.predict)
        predict_btn.grid(row=1, column=0, pady=(0, 10))

        clear_btn = tk.Button(root, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, pady=(0, 10))

        self.result_label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 16))
        self.result_label.grid(row=1, column=2, pady=(0, 10))

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_RADIUS * 2, fill="black",
                capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0, width=BRUSH_RADIUS * 2,
            )
        self.last_x, self.last_y = x, y

    def reset_stroke(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit (0-9)")

    def preprocess(self):
        """Convert the drawn image into a 28x28 feature vector matching the
        MNIST format (white digit on a black background, pixel values
        0-1), following the same crop-resize-center steps MNIST itself was
        built with. This centering step matters a lot: without it, a digit
        drawn off-center or at a different scale than the training data
        confuses the model badly."""
        img = ImageOps.invert(self.image)  # MNIST expects bright strokes on dark background

        # Crop to the bounding box of the ink; skip if the canvas is blank
        bbox = img.getbbox()
        if bbox is None:
            return np.zeros((1, 784))
        img = img.crop(bbox)

        # Resize so the longer side is 20px, preserving aspect ratio
        # (mirrors MNIST's own preprocessing, which centers digits in a
        # 20x20 box inside the final 28x28 canvas)
        w, h = img.size
        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Paste the resized digit onto a blank 28x28 canvas, centered by
        # the digit's center of mass rather than just its bounding box
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

    def predict(self):
        features = self.preprocess()
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.1f}%)")


def main():
    # When launched by double-clicking (e.g. via pythonw with no console
    # window attached), an unhandled exception would otherwise fail
    # silently, so report errors with a message box instead.
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"'{MODEL_PATH}' not found. Run 'python train_model.py' first."
            )
        model = joblib.load(MODEL_PATH)

        root = tk.Tk()

        # Errors raised inside button callbacks (e.g. Predict) are caught
        # by tkinter internally and never reach this function's own
        # try/except, so report them through this hook instead.
        def report_callback_exception(exc, val, tb):
            messagebox.showerror("Digit Recognizer - Error", str(val))

        root.report_callback_exception = report_callback_exception

        app = DigitRecognizerApp(root, model)
        root.mainloop()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Digit Recognizer - Error", str(e))
        raise


if __name__ == "__main__":
    main()
