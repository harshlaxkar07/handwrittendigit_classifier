from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import tensorflow as tf
import cv2
import base64
from preprocess_images import preprocess_digit

# load models
model_linear = tf.keras.models.load_model("model/mnist_linear.h5")
model_cnn = tf.keras.models.load_model("model/mnist_cnn.h5")

# choose which model to use
model = model_cnn

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()

    # preprocess image
    processed, padded_img = preprocess_digit(img_bytes)

    if processed is None:
        return jsonify({"error": "no digit found"}), 400

    # save preprocessed image
    cv2.imwrite("preprocessed.png", padded_img)

    # convert image to base64 for frontend
    _, buffer = cv2.imencode(".png", padded_img)
    preprocessed_b64 = base64.b64encode(buffer).decode("utf-8")

    # model prediction
    preds = model.predict(processed)
    pred = int(np.argmax(preds))

    return jsonify({
        "prediction": pred,
        "preprocessed": preprocessed_b64
    })

@app.route("/download_preprocessed")
def download_preprocessed():
    return send_file("preprocessed.png", as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000)
