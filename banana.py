import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# Config
# -----------------------------
OLD_MODEL_FILE = "my_cnn_model.keras"  # Old Keras 2 model
MODEL_FILE = "/content/drive/MyDrive/my_cnn_model_v3.keras"  # Converted Keras 3 model
GOOGLE_DRIVE_FILE_ID = "19ondqnTkzrM07XS1TCtLxuE44fE7BdYC"

CLASS_NAMES = [
    "Augmented Banana Black Sigatoka Disease",
    "Augmented Banana Bract Mosaic Virus Disease",
    "Augmented Banana Healthy Leaf",
    "Augmented Banana Insect Pest Disease",
    "Augmented Banana Moko Disease",
    "Augmented Banana Panama Disease",
    "Augmented Banana Yellow Sigatoka Disease"
]

# -----------------------------
# Mount Google Drive (if using Colab)
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')

# -----------------------------
# Download old model if it doesn't exist
# -----------------------------
if not os.path.exists(OLD_MODEL_FILE):
    st.info("Downloading old Keras 2 model...")
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, OLD_MODEL_FILE, quiet=False, fuzzy=True)

# -----------------------------
# Convert model to Keras 3 if needed
# -----------------------------
if not os.path.exists(MODEL_FILE):
    st.info("Converting model to Keras 3 format...")
    old_model = tf.keras.models.load_model(OLD_MODEL_FILE)

    new_model = Sequential()
    for layer in old_model.layers:
        if isinstance(layer, InputLayer):
            new_model.add(tf.keras.Input(shape=layer.input_shape[1:]))
        else:
            layer_config = layer.get_config()
            layer_class = layer.__class__
            new_layer = layer_class.from_config(layer_config)
            new_model.add(new_layer)

    new_model.save(MODEL_FILE)
    st.success(f"Model converted and saved to Drive: {MODEL_FILE}")

# -----------------------------
# Load Keras 3 model
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍌 Banana Leaf Disease Classifier")
st.write("Upload a banana leaf image and get a disease prediction.")

uploaded_file = st.file_uploader("Choose a banana leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[np.argmax(score)]}")
    st.write(f"**Confidence:** {100 * np.max(score):.2f}%")


