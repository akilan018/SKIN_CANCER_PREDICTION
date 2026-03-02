import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

print("TensorFlow version:", tf.__version__)

model = None

# ---- Safe load for old .h5 model in TF2.15+ ----
try:
    model = tf.keras.models.load_model(
        "best_model.h5",
        compile=False,
        safe_mode=False
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model failed to load:", e)
    model = None

class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus (Normal Skin)",
    "Vascular Lesion"
]

cancer_classes = [0, 1, 4]


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img)

        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        return jsonify({
            "skin_type": class_names[class_index],
            "confidence": round(confidence, 2),
            "is_cancer": class_index in cancer_classes,
            "class_index": class_index
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)