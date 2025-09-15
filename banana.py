import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------
# Model download & loading
# -------------------------
model_path = "my_cnn_model_v3.keras"  # Keras 3 compatible model
if not os.path.exists(model_path):
    st.info("Downloading model...")
    # Use correct Google Drive download URL
    file_id = "19ondqnTkzrM07XS1TCtLxuE44fE7BdYC"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(download_url, model_path, quiet=False)

# Load model with error handling
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------
# Class names
# -------------------------
CLASS_NAMES = [
    "Augmented Banana Black Sigatoka Disease",
    "Augmented Banana Bract Mosaic Virus Disease",
    "Augmented Banana Healthy Leaf",
    "Augmented Banana Insect Pest Disease",
    "Augmented Banana Moko Disease",
    "Augmented Banana Panama Disease",
    "Augmented Banana Yellow Sigatoka Disease"
]

# -------------------------
# Streamlit UI
# -------------------------
st.title("🍌 Banana Leaf Disease Classifier")
st.write("Upload a banana leaf image and get a disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a banana leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with error handling
    try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        st.subheader("Prediction")
        st.write(f"**Class:** {CLASS_NAMES[np.argmax(score)]}")
        st.write(f"**Confidence:** {100 * np.max(score):.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")




