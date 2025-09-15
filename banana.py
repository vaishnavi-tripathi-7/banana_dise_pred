import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# Config: Model file and Drive ID
# -----------------------------
MODEL_FILE = "my_cnn_model_v3.keras"
GOOGLE_DRIVE_FILE_ID = "1IDw8g1dZps0LwDQBn_j_DjJLjcYywqgI"  # Your Drive file ID

# -----------------------------
# Download model if it doesn't exist
# -----------------------------
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model...")
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False, fuzzy=True)

# -----------------------------
# Load model
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_FILE)
except Exception as e:
    st.error(f"Failed to load model. Error:\n{e}")
    st.stop()

# -----------------------------
# Class names
# -----------------------------
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
# Streamlit App UI
# -----------------------------
st.title("🍌 Banana Leaf Disease Classifier")
st.write("Upload a banana leaf image and get a disease prediction.")

uploaded_file = st.file_uploader("Choose a banana leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display results
    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[np.argmax(score)]}")
    st.write(f"**Confidence:** {100 * np.max(score):.2f}%")





