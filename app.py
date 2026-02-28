import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Skin Cancer Detection", layout="centered")

st.title("🧬 AI Skin Cancer Detection System")
st.write("Upload a dermoscopy skin image for analysis")

# ===============================
# LOAD MODEL (SAFE WAY)
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

# ===============================
# CLASS NAMES
# ===============================
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus (Normal Skin)",
    "Vascular Lesion"
]

cancer_classes = [0, 1, 4]  # Cancer categories

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Choose a skin image...", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ===============================
    # PREPROCESS IMAGE
    # ===============================
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ===============================
    # PREDICT
    # ===============================
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # ===============================
    # RESULTS
    # ===============================
    st.subheader("🔍 Prediction Result")

    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if predicted_class in cancer_classes:
        st.error("⚠️ Cancer Detected - Please consult a dermatologist.")
    else:
        st.success("✅ Non-Cancerous Skin Condition")
