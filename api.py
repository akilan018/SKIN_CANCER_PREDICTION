import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

print(f"Loading best_model.h5 on TF {tf.__version__} (Keras {tf.keras.__version__})")
try:
    # Now that we downgraded TF/Keras to 2.15, the original .h5 will load perfectly
    model = tf.keras.models.load_model("best_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
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
cancer_classes = [0, 1, 4]  # AKIEC, BCC, MEL

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on the server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # EXACT MATCH of new/app.py logic
        img = Image.open(file.stream).convert('RGB')
        
        # Preprocess exactly as in app.py
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized)
        
        # This was doing the critical magic in Keras 2
        arr = preprocess_input(arr)
        
        arr = np.expand_dims(arr, axis=0)

        # Inference
        pred = model.predict(arr, verbose=0)
        pred_probs = pred[0]

        print("--- PREDICTION ---")
        for name, prob in zip(class_names, pred_probs):
            print(f"  {name}: {prob*100:.2f}%")

        # Get highest prob
        class_index = int(np.argmax(pred_probs))
        confidence = float(np.max(pred_probs)) * 100
        skin_type = class_names[class_index]
        print(f"  Model chose: {skin_type} ({confidence:.2f}%)")

        is_cancer = class_index in cancer_classes

        # Do NOT force "cancer" if sum of background cancer probs > 2% 
        # (That was the bug making everything cancer.)
        # The user's code just checked cancer_prob for a warning
        cancer_prob = sum(float(pred_probs[i]) for i in cancer_classes)
        print(f"  Total Cancer probability sum: {cancer_prob*100:.2f}%")

        return jsonify({
            'skin_type': skin_type,
            'confidence': round(confidence, 2),
            'is_cancer': is_cancer,
            'class_index': class_index,
            'low_confidence': confidence < 50
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "model_loaded": model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
