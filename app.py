"""
Handwritten digit classifier web application.

Serves a drawing pad and an upload form, and returns the predicted digit
together with the full probability distribution from the selected model.
"""

import base64

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_file

from preprocess_images import preprocess_digit


# ============================================================
# Models
# ============================================================

MODELS = {
    "cnn": tf.keras.models.load_model("model/mnist_cnn.h5"),
    "linear": tf.keras.models.load_model("model/mnist_linear.h5"),
}

DEFAULT_MODEL = "cnn"


app = Flask(__name__)


# ============================================================
# Helpers
# ============================================================

def read_request_image():
    """
    Pull image bytes out of either an uploaded file or a canvas data URL.
    """

    uploaded = request.files.get("file")

    if uploaded is not None and uploaded.filename:
        return uploaded.read()

    payload = request.get_json(silent=True) or {}

    data_url = request.form.get("image") or payload.get("image")

    if data_url:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        return base64.b64decode(data_url)

    return None


def shape_for(model_name, processed):
    """
    Reshape the 28x28 input to whatever the chosen model expects.
    """

    expected = MODELS[model_name].input_shape

    flat = processed.reshape(1, -1)

    if len(expected) == 2:
        return flat

    if len(expected) == 3:
        return flat.reshape(1, 28, 28)

    return flat.reshape(1, 28, 28, 1)


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    """
    Serve the classifier interface.
    """

    return render_template("index.html")


@app.route("/models")
def models():
    """
    Report which models are available and which one is the default.
    """

    return jsonify(
        {
            "models": [
                {
                    "id": name,
                    "label": "Convolutional network" if name == "cnn" else "Linear model",
                    "parameters": int(model.count_params()),
                }
                for name, model in MODELS.items()
            ],
            "default": DEFAULT_MODEL,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a digit from an uploaded image or a canvas drawing.
    """

    image_bytes = read_request_image()

    if not image_bytes:
        return jsonify({"error": "No image was supplied."}), 400

    payload = request.get_json(silent=True) or {}

    model_name = str(
        request.form.get("model") or payload.get("model") or DEFAULT_MODEL
    ).lower()

    if model_name not in MODELS:
        model_name = DEFAULT_MODEL

    processed, padded_img = preprocess_digit(image_bytes)

    if processed is None:
        return jsonify({"error": "No digit was found in the image."}), 400

    # Keep the last preprocessed image available for download.
    cv2.imwrite("preprocessed.png", padded_img)

    _, buffer = cv2.imencode(".png", padded_img)
    preprocessed_b64 = base64.b64encode(buffer).decode("utf-8")

    predictions = MODELS[model_name].predict(
        shape_for(model_name, processed),
        verbose=0,
    )[0]

    probabilities = np.asarray(predictions, dtype="float64")

    # Models that end in a softmax already return a distribution; models that
    # return raw logits are normalised here so the response is always a
    # probability over the ten digits.
    is_distribution = (probabilities >= 0).all() and np.isclose(
        probabilities.sum(), 1.0, atol=0.02
    )

    if not is_distribution:
        shifted = probabilities - probabilities.max()
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum()

    digit = int(np.argmax(probabilities))

    return jsonify(
        {
            "prediction": digit,
            "confidence": float(probabilities[digit]),
            "probabilities": [float(p) for p in probabilities],
            "preprocessed": preprocessed_b64,
            "model": model_name,
        }
    )


@app.route("/download_preprocessed")
def download_preprocessed():
    """
    Download the most recently preprocessed 28x28 image.
    """

    return send_file("preprocessed.png", as_attachment=True)


if __name__ == "__main__":
    app.run(port=5000)
